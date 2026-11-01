"""Network forward + backward (dense / MoE / norm / residual)."""
import numpy as np
import pytest

from backend.core.errors import SQUARE
from backend.core.nn import Network, normalize_spec, expert_output_dim
from backend.core.seed import make_rng


def _dense(n=4, act="tanh", bias=True, norm="none", pos="post"):
    return {"type": "dense", "n": n, "act": act, "bias": bias,
            "norm": norm, "norm_pos": pos}


def _build(**kwargs):
    defaults = dict(
        input_ids=["x", "y"],
        layer_specs=[_dense()],
        output_size=1,
        output_activation="tanh",
        rng=make_rng("t", 0),
    )
    defaults.update(kwargs)
    return Network(**defaults)


def _moe_spec(experts=3, k=2, soft=False, aux=0.01):
    return {
        "type": "moe", "experts": experts, "top_k": k, "soft": soft,
        "router_bias": True, "router_temp": 1.0, "normalize": True,
        "aux_coef": aux,
        "expert": {"layers": [{"n": 3, "act": "relu", "bias": True}]},
    }


# ---- normalize_spec ----

def test_normalize_spec_clamps():
    s = normalize_spec({"type": "dense", "n": 100, "act": "xyz"})
    assert s["n"] == 32
    assert s["act"] == "sigmoid"
    assert s["norm"] == "none"


def test_normalize_moe_k_min_2():
    s = normalize_spec({"type": "moe", "experts": 4, "top_k": 1, "soft": False})
    assert s["top_k"] == 2


def test_normalize_moe_soft_forces_full():
    s = normalize_spec({"type": "moe", "experts": 4, "soft": True})
    assert s["top_k"] == 4


def test_normalize_moe_legacy_expert():
    s = normalize_spec({"type": "moe", "experts": 2,
                         "expert": {"n": 5, "act": "relu"}})
    assert s["expert"]["layers"][0]["n"] == 5


def test_expert_output_dim():
    assert expert_output_dim(_dense(n=7)) == 7
    s = normalize_spec({
        "type": "moe", "experts": 3,
        "expert": {"layers": [{"n": 5, "act": "relu"},
                               {"n": 2, "act": "relu"}]},
    })
    assert expert_output_dim(s) == 2


# ---- forward ----

def test_forward_dense_shape():
    net = _build()
    out = net.forward(np.random.randn(10, 2))
    assert out.shape == (10, 1)


def test_forward_all_layers():
    net = _build()
    outs = net.forward_all_layers(np.random.randn(5, 2))
    assert len(outs) == net.num_layers + 1
    assert outs[0].shape == (5, 2)
    assert outs[-1].shape == (5, 1)


def test_forward_multiclass_softmax():
    net = _build(output_size=3, is_multiclass=True)
    out = net.forward(np.random.randn(4, 2))
    assert out.shape == (4, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-9)


def test_forward_with_cache_and_probe_agree():
    net = _build()
    X = np.random.randn(6, 2)
    a = net.forward(X)
    b, _, _ = net.forward_with_cache(X)
    np.testing.assert_allclose(a, b)


# ---- backward (finite diff) ----

def test_backward_matches_finite_difference():
    net = _build(layer_specs=[_dense(n=3, act="tanh")])
    X = np.array([[0.5, -0.3], [0.1, 0.9]])
    T = np.array([0.5, -0.8])
    _, lc, oc = net.forward_with_cache(X)
    grads = net.backward(lc, oc, T, SQUARE)
    eps = 1e-5
    for i in range(net.weights[0].shape[0]):
        for j in range(net.weights[0].shape[1]):
            old = net.weights[0][i, j]
            net.weights[0][i, j] = old + eps
            lp = float(SQUARE.error(net.forward(X),
                                     T.reshape(-1, 1)).mean())
            net.weights[0][i, j] = old - eps
            lm = float(SQUARE.error(net.forward(X),
                                     T.reshape(-1, 1)).mean())
            net.weights[0][i, j] = old
            num = (lp - lm) / (2 * eps)
            ana = grads["weights"][0][i, j] / X.shape[0]
            assert abs(num - ana) < 1e-4, f"w[{i},{j}]: {num} vs {ana}"


def test_backward_multiclass_shapes():
    net = _build(output_size=3, is_multiclass=True)
    X = np.random.randn(5, 2)
    T = np.array([0, 1, 2, 0, 1])
    _, lc, oc = net.forward_with_cache(X)
    grads = net.backward(lc, oc, T, SQUARE)
    assert grads["weights"][-1].shape == net.weights[-1].shape


# ---- MoE ----

def test_moe_forward_shape():
    net = _build(layer_specs=[_moe_spec()])
    assert net.forward(np.random.randn(7, 2)).shape == (7, 1)


def test_moe_gating_normalized():
    net = _build(layer_specs=[_moe_spec(k=2)])
    _, probe = net.forward_with_probe(np.random.randn(5, 2))
    np.testing.assert_allclose(probe[0]["weights"].sum(axis=1), 1.0, rtol=1e-9)


def test_moe_soft_sums_to_one():
    net = _build(layer_specs=[_moe_spec(experts=3, soft=True)])
    _, probe = net.forward_with_probe(np.random.randn(4, 2))
    np.testing.assert_allclose(probe[0]["weights"].sum(axis=1), 1.0, rtol=1e-9)


def test_moe_backward_produces_grads():
    net = _build(layer_specs=[_moe_spec()])
    X = np.random.randn(5, 2)
    T = np.random.randn(5) * 0.5
    _, lc, oc = net.forward_with_cache(X)
    grads = net.backward(lc, oc, T, SQUARE)
    assert grads["router_weights"][0] is not None
    assert grads["expert_weights"][0] is not None


def test_moe_top1_normalized_router_grad_is_zero():
    """Documented in tutorial §11.3: k=1 + normalize ⇒ zero router gradient."""
    spec = _moe_spec(k=2, aux=0.0)
    net = _build(layer_specs=[spec])
    net._specs[0]["top_k"] = 1        # bypass clamp
    X = np.random.randn(5, 2)
    T = np.random.randn(5) * 0.5
    _, lc, oc = net.forward_with_cache(X)
    grads = net.backward(lc, oc, T, SQUARE)
    if grads["router_weights"][0] is not None:
        assert np.allclose(grads["router_weights"][0], 0.0, atol=1e-10)


# ---- normalization ----

def test_layernorm_mean_var():
    net = _build(layer_specs=[_dense(n=4, norm="layernorm", act="linear")])
    X = np.random.randn(3, 2)
    pre = net.forward_all_layers(X)[0] @ net.weights[0].T + net.biases[0]
    outs = net.forward_all_layers(X)
    h = outs[1]
    np.testing.assert_allclose(h.mean(axis=1), 0.0, atol=1e-6)
    # 精确期望值: var/(var+eps),而不是 1
    expected = pre.var(axis=1) / (pre.var(axis=1) + net.norm_eps)
    np.testing.assert_allclose(h.var(axis=1), expected, rtol=1e-6)


def test_rmsnorm_unit_rms():
    net = _build(layer_specs=[_dense(n=4, norm="rmsnorm", act="linear")])
    outs = net.forward_all_layers(np.random.randn(3, 2))
    rms = np.sqrt((outs[1] ** 2).mean(axis=1))
    np.testing.assert_allclose(rms, 1.0, atol=1e-3)


# ---- residuals ----

def test_residual_mismatch_dropped():
    net = Network(
        input_ids=["x", "y"],
        layer_specs=[_dense(n=4), _dense(n=4)],
        output_size=1, output_activation="tanh",
        rng=make_rng("r", 0), residuals=[[0, 2]],
    )
    assert net.skip_links[1] == []


def test_residual_valid_kept():
    net = Network(
        input_ids=["x", "y"],
        layer_specs=[_dense(n=2), _dense(n=2)],
        output_size=1, output_activation="tanh",
        rng=make_rng("r", 1), residuals=[[0, 2]],
    )
    assert net.skip_links[1] == [0]