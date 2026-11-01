"""compile_network_to_py consistency."""
import numpy as np
import pytest

from backend.core.compile_py import compile_network_to_py, generated_forward
from backend.core.nn import Network
from backend.core.seed import make_rng


def _net(act="tanh", norm="none", pos="post", **kwargs):
    defaults = dict(
        input_ids=["x", "y"],
        layer_specs=[{"type": "dense", "n": 3, "act": act, "bias": True,
                       "norm": norm, "norm_pos": pos}],
        output_size=1, output_activation="tanh",
        rng=make_rng("c", 0),
    )
    defaults.update(kwargs)
    return Network(**defaults)


def test_valid_python_syntax():
    compile(compile_network_to_py(_net()), "<gen>", "exec")


def test_matches_dense_tanh():
    net = _net()
    X = np.random.randn(4, 2)
    out = net.forward(X).ravel()
    fwd = generated_forward(net)
    for i in range(4):
        assert abs(fwd(float(X[i, 0]), float(X[i, 1])) - out[i]) < 1e-9


@pytest.mark.parametrize("norm", ["layernorm", "rmsnorm"])
def test_matches_norm(norm):
    net = _net(norm=norm)
    X = np.random.randn(3, 2)
    out = net.forward(X).ravel()
    fwd = generated_forward(net)
    for i in range(3):
        assert abs(fwd(float(X[i, 0]), float(X[i, 1])) - out[i]) < 1e-6


def test_matches_moe():
    spec = {"type": "moe", "experts": 2, "top_k": 2, "soft": False,
            "router_temp": 1.0, "normalize": True, "aux_coef": 0.0,
            "expert": {"layers": [{"n": 2, "act": "relu", "bias": True}]}}
    net = Network(
        input_ids=["x", "y"], layer_specs=[spec], output_size=1,
        output_activation="tanh", rng=make_rng("cmoe", 0),
    )
    X = np.random.randn(3, 2)
    out = net.forward(X).ravel()
    fwd = generated_forward(net)
    for i in range(3):
        assert abs(fwd(float(X[i, 0]), float(X[i, 1])) - out[i]) < 1e-6


def test_matches_multiclass_softmax():
    net = _net(is_multiclass=True, output_size=3, output_activation="linear")
    X = np.random.randn(3, 2)
    out = net.forward(X)
    fwd = generated_forward(net)
    for i in range(3):
        np.testing.assert_allclose(fwd(float(X[i, 0]), float(X[i, 1])),
                                    out[i], rtol=1e-6)