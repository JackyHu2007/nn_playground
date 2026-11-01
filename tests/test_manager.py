"""SessionManager behavior."""
import time

from backend.data.types import Example2D
from backend.session.manager import SessionManager


def test_create_and_get():
    mgr = SessionManager()
    sid, sess = mgr.create()
    assert mgr.get(sid) is sess
    assert mgr.count() == 1


def test_get_unknown():
    mgr = SessionManager()
    assert mgr.get("nope") is None


def test_delete():
    mgr = SessionManager()
    sid, _ = mgr.create()
    mgr.delete(sid)
    assert mgr.get(sid) is None
    assert mgr.count() == 0


def test_custom_dataset_roundtrip():
    mgr = SessionManager()
    pts = [Example2D(0.0, 0.0, 1.0), Example2D(1.0, 1.0, -1.0)]
    did = mgr.add_custom(pts)
    got = mgr.get_custom(did)
    assert got == pts
    got.append(Example2D(9.0, 9.0, 1.0))
    assert len(mgr.get_custom(did)) == 2   # store is unaffected


def test_get_custom_unknown():
    mgr = SessionManager()
    assert mgr.get_custom("nope") is None


def test_cleanup_expires():
    mgr = SessionManager(ttl_seconds=0.01)
    sid, _ = mgr.create()
    time.sleep(0.05)
    removed = mgr.cleanup()
    assert removed >= 1
    assert mgr.get(sid) is None