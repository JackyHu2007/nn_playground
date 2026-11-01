"""REST endpoints via FastAPI TestClient."""
import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_default_state(client):
    r = client.get("/api/state/default")
    assert r.status_code == 200
    assert r.json()["problem"] == "classification"


def test_datasets(client):
    r = client.get("/api/datasets")
    assert r.status_code == 200
    datasets = r.json()["datasets"]
    linear = next(d for d in datasets if d["name"] == "linear")
    assert linear["category"] == "classification"
    assert len(linear["preview"]) == 200


def test_commands(client):
    cmds = client.get("/api/commands").json()["commands"]
    assert "step" in cmds and "play" in cmds


def test_custom_upload_ok(client):
    points = [
        {"x": 0.5, "y": 0.5, "label": 1.0},
        {"x": -0.5, "y": -0.5, "label": -1.0},
        {"x": 0.1, "y": -0.1, "label": 1.0},
    ]
    r = client.post("/api/datasets/custom", json={"points": points})
    assert r.status_code == 200
    body = r.json()
    assert body["problem"] == "classification"
    assert body["numClasses"] == 2
    assert body["numPoints"] == 3


def test_custom_empty_rejected(client):
    r = client.post("/api/datasets/custom", json={"points": []})
    assert r.status_code == 400


def test_custom_single_label_rejected(client):
    r = client.post("/api/datasets/custom",
                    json={"points": [{"x": 0.0, "y": 0.0, "label": 1.0},
                                       {"x": 1.0, "y": 1.0, "label": 1.0}]})
    assert r.status_code == 400
    assert "distinct" in r.json()["detail"].lower()


def test_custom_too_many_classes(client):
    points = [{"x": float(i), "y": 0.0, "label": float(i)} for i in range(13)]
    r = client.post("/api/datasets/custom", json={"points": points})
    assert r.status_code == 400
    assert "too many" in r.json()["detail"].lower()


def test_custom_multiclass_inferred(client):
    points = [{"x": float(i), "y": 0.0, "label": float(i)} for i in range(3)]
    body = client.post("/api/datasets/custom", json={"points": points}).json()
    assert body["problem"] == "multiclass"
    assert body["numClasses"] == 3


def test_custom_regression_inferred(client):
    points = [{"x": 0.0, "y": 0.0, "label": 0.5},
              {"x": 1.0, "y": 0.0, "label": 1.5},
              {"x": 0.0, "y": 1.0, "label": 2.5}]
    body = client.post("/api/datasets/custom", json={"points": points}).json()
    assert body["problem"] == "regression"