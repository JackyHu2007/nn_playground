"""TrainingSession end-to-end."""
import numpy as np
import pytest

from backend.session.session import TrainingSession
from backend.session.state import State


def _session(**overrides):
    s = State()
    s.num_samples = 100
    for k, v in overrides.items():
        setattr(s, k, v)
    return TrainingSession(s, density=10)


def test_constructs():
    sess = _session()
    assert sess.network is not None
    assert len(sess.train_data) > 0


def test_one_step_increments_iter():
    sess = _session()
    sess.one_step(update_boundaries=False)
    sess.one_step(update_boundaries=False)
    assert sess.iter == 2


def test_reset_reproduces_initial_weights():
    """Regression guard for the reset() seed-counter fix."""
    sess = _session()
    w_init = sess.network.weights[0].copy()
    sess.one_step(update_boundaries=False)
    assert not np.array_equal(w_init, sess.network.weights[0])
    sess.reset()
    assert sess.iter == 0
    np.testing.assert_array_equal(sess.network.weights[0], w_init)


def test_reset_keeps_weight_seed_counter():
    sess = _session()
    c0 = sess._weight_seed_counter
    sess.reset()
    assert sess._weight_seed_counter == c0


def test_regenerate_bumps_weight_seed_counter():
    sess = _session()
    c0 = sess._weight_seed_counter
    w0 = sess.network.weights[0].copy()
    x0 = np.array([[p.x, p.y] for p in sess.train_data])
    sess.regenerate()
    assert sess._weight_seed_counter == c0 + 1
    assert sess.iter == 0
    assert not np.array_equal(w0, sess.network.weights[0])
    x1 = np.array([[p.x, p.y] for p in sess.train_data])
    assert not np.array_equal(x0, x1)


def test_set_learning_rate():
    sess = _session()
    sess.set("learningRate", 0.05)
    assert sess.state.learning_rate == 0.05
    assert sess.optimizer.lr == 0.05


def test_set_dataset_regenerates():
    sess = _session()
    sess.set("dataset", "xor")
    assert sess.state.dataset == "xor"
    assert sess.iter == 0


def test_set_unknown_key_raises():
    sess = _session()
    with pytest.raises(ValueError):
        sess.set("bogus_key", 1)


def test_get_frame_minimal():
    sess = _session()
    frame = sess.get_frame(include_boundaries=False)
    for k in ("iter", "lossTrain", "lossTest", "networkShape",
              "nodeIds", "trainPoints", "weights", "biases", "state", "pyCode"):
        assert k in frame
    assert frame["iter"] == 0
    assert "outputBoundary" not in frame


def test_multiclass_frame():
    sess = _session(problem="multiclass", dataset="multi-blobs")
    frame = sess.get_frame(include_boundaries=False)
    assert frame["outputMode"] == "class"
    assert frame["numClasses"] == 3


def test_node_boundary_lookup():
    sess = _session()
    for nid in sess.network.layer_node_ids[1]:
        b = sess.get_node_boundary(nid)
        assert b is not None
        assert len(b) == sess.density


def test_unknown_node_boundary():
    sess = _session()
    assert sess.get_node_boundary("nonexistent") is None


def test_load_state_resets_iter():
    sess = _session()
    new_state = State()
    new_state.learning_rate = 0.05
    sess.load_state(new_state)
    assert sess.state.learning_rate == 0.05
    assert sess.iter == 0


def test_load_state_invalid_dataset_falls_back():
    sess = _session()
    bad = State()
    bad.dataset = "garbage-name"
    sess.load_state(bad)
    from backend.data.registry import DATASETS
    assert sess.state.dataset in DATASETS


def test_moe_frame_contains_moe_info():
    spec = {"type": "moe", "experts": 2, "top_k": 2, "soft": False,
            "router_temp": 1.0, "normalize": True, "aux_coef": 0.01,
            "expert": {"layers": [{"n": 2, "act": "relu", "bias": True}]}}
    sess = _session(layer_specs=[spec])
    frame = sess.get_frame(include_boundaries=True)
    assert "moeInfo" in frame
    info = frame["moeInfo"][0]
    assert info["numExperts"] == 2
    assert "expertNodeIds" in info
    assert "routingWeights" in info