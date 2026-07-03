# Deploying Lore as an always-on service

Run Lore as a background service so it stays connected to Slack (Socket Mode) across logouts
and reboots — e.g. for the duration of a hackathon judging window.

## Prerequisites
- The app installed in a venv: `python -m venv .venv && .venv/bin/pip install -e .`
- A Slack app installed to your workspace (bot + app-level tokens) — see `../manifest.yaml`.
- A local model reachable at `OLLAMA_API_BASE` (or any OpenAI-compatible endpoint).

## 1. Create the environment file (secrets — keep it out of git)
```bash
cp deploy/lore.env.example ~/projects/slack-mcp-agent/lore.env
chmod 600 ~/projects/slack-mcp-agent/lore.env
# edit it: fill in SLACK_BOT_TOKEN / SLACK_APP_TOKEN / SLACK_SIGNING_SECRET,
# OLLAMA_API_BASE, LORE_MODEL, LORE_CHANNELS
```

## 2. Install as a systemd **user** service (recommended — no root needed)
```bash
mkdir -p ~/.config/systemd/user
cp deploy/lore-slack-agent.service ~/.config/systemd/user/
loginctl enable-linger "$USER"          # start on boot + survive reboots/logout
systemctl --user daemon-reload
systemctl --user enable --now lore-slack-agent.service
```

Adjust the paths in the unit file if your checkout isn't at `~/projects/slack-mcp-agent`.

## Manage it
```bash
systemctl --user status  lore-slack-agent      # is it running?
systemctl --user restart lore-slack-agent      # restart (e.g. after a code update)
systemctl --user stop    lore-slack-agent      # stop it (after judging)
systemctl --user disable lore-slack-agent      # don't start on boot anymore
tail -f ~/projects/slack-mcp-agent/lore-app.log   # live logs
```

To fully retire it after the event:
```bash
systemctl --user disable --now lore-slack-agent
# optional: loginctl disable-linger "$USER"
```

## Alternative: system service (needs root)
Copy the unit to `/etc/systemd/system/lore-slack-agent.service`, replace `%h` with the absolute
home path, add `User=<you>`, then `sudo systemctl daemon-reload && sudo systemctl enable --now lore-slack-agent`.

## Notes
- `Restart=always` brings it back automatically if the model endpoint or network blips.
- Socket Mode needs **no public URL** — the service just needs outbound HTTPS/WebSocket.
- Run only **one** instance per app (multiple Socket Mode connections split event delivery).
