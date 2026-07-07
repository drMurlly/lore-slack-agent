from collections import OrderedDict
import threading
import time


class EventDedup:
    """Idempotency guard for Slack event delivery.

    Thread-safe: Bolt dispatches listeners on a shared worker pool and a single module-level
    instance is shared across them, so the whole check-and-set must be atomic. Without the lock,
    (a) one thread iterating ``_seen`` while another mutates it raises ``RuntimeError: OrderedDict
    mutated during iteration``, and (b) two threads handling the same redelivered event both see
    it as new and run the full research twice, posting a duplicate answer.
    """

    def __init__(self, maxsize: int = 2000, ttl_s: float = 300.0) -> None:
        self._seen: OrderedDict[str, float] = OrderedDict()
        self._maxsize = maxsize
        self._ttl_s = ttl_s
        self._lock = threading.Lock()

    def is_seen(self, event_id: str) -> bool:
        """Return True if this event_id was already processed; mark it seen on first call.

        An empty/None id is never a duplicate — otherwise a payload with no id (e.g. a slash
        command) would key every invocation to the same "" and silently drop them all."""
        if not event_id:
            return False
        with self._lock:
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
