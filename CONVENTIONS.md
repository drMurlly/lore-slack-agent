# CONVENTIONS — engineering conventions for this repo

- **Every module gets a real unit test** in `tests/` exercising actual behavior (never `assert True` or import-only tests).
- **Pure, testable functions.** Keep logic in plain functions callable WITHOUT live Slack/network; inject deps (llm, rts, poster) as params so tests pass fakes.
- **No live network in tests.** Mock Slack/RTS/LLM/HTTP. Tests pass offline in <2s.
- **Only stdlib + declared deps.** Need a new dep? add it to `requirements.txt` AND import lazily; prefer `urllib` over `requests`.
- **Don't rewrite passing code.** Green module → leave it; add, don't churn.
- **Small typed functions** (dataclasses for structured data). Match existing `src/conduit/` style.
- **Fix the root cause, not the test.** If test and code disagree, reconcile by the code's intent; never weaken assertions to force green.
