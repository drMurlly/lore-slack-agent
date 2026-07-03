from collections import OrderedDict
import time


class EventDedup:
    """Idempotency guard for Slack event delivery."""

    def __init__(self, maxsize: int = 2000, ttl_s: float = 300.0) -> None:
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._maxsize = maxsize
        self._ttl_s = ttl_s

    def is_seen(self, event_id: str) -> bool:
        """Return True if this event_id was already processed; mark it seen on first call.

        An empty/None id is never a duplicate — otherwise a payload with no id (e.g. a slash
        command) would key every invocation to the same "" and silently drop them all."""
        if not event_id:
            return False
        now = time.monotonic()
        # evict expired
        expired = [k for k, t in self._seen.items() if now - t >= self._ttl_s]
        for k in expired:
            del self._seen[k]
        if event_id in self._seen:
            return True
        # evict oldest if over capacity
        if len(self._seen) >= self._maxsize:
            self._seen.popitem(last=False)
        self._seen[event_id] = now
        return False
