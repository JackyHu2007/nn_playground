"""WebSocket protocol via TestClient."""
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _drain_initial(ws):
    """Read welcome + initial frame; return (welcome, frame)."""
    welcome = ws.receive_json()
    frame = ws.receive_json()
    return welcome, frame


def _read_frame(ws, limit=10):
    for _ in range(limit):
        msg = ws.receive_json()
        if "iter" in msg:
            return msg
    raise AssertionError("no frame received")


def test_welcome(client):
    with client.websocket_connect("/ws") as ws:
        welcome, _ = _drain_initial(ws)
        assert welcome["type"] == "welcome"
        assert "sessionId" in welcome
        assert "step" in welcome["commands"]


def test_ping_pong(client):
    with client.websocket_connect("/ws") as ws:
        _drain_initial(ws)
        ws.send_json({"cmd": "ping"})
        for _ in range(5):
            if ws.receive_json().get("type") == "pong":
                return
        pytest.fail("no pong")


def test_step(client):
    with client.websocket_connect("/ws") as ws:
        _drain_initial(ws)
        ws.send_json({"cmd": "step", "n": 1})
        frame = _read_frame(ws)
        assert frame["iter"] == 1


def test_set_learning_rate(client):
    with client.websocket_connect("/ws") as ws:
        _drain_initial(ws)
        ws.send_json({"cmd": "set", "key": "learningRate", "value": 0.05})
        for _ in range(10):
            msg = ws.receive_json()
            if "state" in msg:
                assert msg["state"]["learningRate"] == 0.05
                return
        pytest.fail("no updated frame")


def test_unknown_command(client):
    with client.websocket_connect("/ws") as ws:
        _drain_initial(ws)
        ws.send_json({"cmd": "not_a_real_command"})
        for _ in range(10):
            if ws.receive_json().get("type") == "error":
                return
        pytest.fail("no error")


def test_request_frame(client):
    with client.websocket_connect("/ws") as ws:
        _drain_initial(ws)
        ws.send_json({"cmd": "request_frame"})
        _read_frame(ws)


def test_load_state(client):
    with client.websocket_connect("/ws") as ws:
        welcome, _ = _drain_initial(ws)
        new_state = dict(welcome["state"])
        new_state["learningRate"] = 0.02
        new_state["noise"] = 10
        ws.send_json({"cmd": "load_state", "state": new_state})
        for _ in range(10):
            msg = ws.receive_json()
            if "state" in msg:
                assert msg["state"]["learningRate"] == 0.02
                assert msg["state"]["noise"] == 10
                return
        pytest.fail("no updated frame")


def test_concurrent_session_gets_new_id(client):
    """Rejoining an active session is refused; fresh id issued."""
    with client.websocket_connect("/ws") as ws1:
        w1, _ = _drain_initial(ws1)
        sid = w1["sessionId"]
        with client.websocket_connect(f"/ws?session={sid}") as ws2:
            w2 = ws2.receive_json()
            assert w2["sessionId"] != sid