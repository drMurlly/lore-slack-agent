import time

from conduit.dedup import EventDedup


def test_first_call_not_seen():
    dedup = EventDedup()
    assert dedup.is_seen("evt-1") is False


def test_second_call_seen():
    dedup = EventDedup()
    dedup.is_seen("evt-1")
    assert dedup.is_seen("evt-1") is True


def test_ttl_expiry():
    dedup = EventDedup(ttl_s=0.01)
    dedup.is_seen("evt-ttl")
    time.sleep(0.02)
    assert dedup.is_seen("evt-ttl") is False


def test_capacity_eviction():
    dedup = EventDedup(maxsize=3)
    dedup.is_seen("id-1")
    dedup.is_seen("id-2")
    dedup.is_seen("id-3")
    dedup.is_seen("id-4")
    assert len(dedup._seen) == 3


def test_different_ids_not_seen():
    dedup = EventDedup()
    assert dedup.is_seen("evt-a") is False
    assert dedup.is_seen("evt-b") is False
