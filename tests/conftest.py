"""Shared fixtures."""
import numpy as np
import pytest

from backend.core.nn import Network
from backend.core.seed import make_rng
from backend.session.state import State


def _dense(n=4, act="tanh", bias=True, norm="none", pos="post"):
    return {"type": "dense", "n": n, "act": act, "bias": bias,
            "norm": norm, "norm_pos": pos}


@pytest.fixture
def simple_state():
    return State()


@pytest.fixture
def simple_network():
    return Network(
        input_ids=["x", "y"],
        layer_specs=[_dense(n=4, act="tanh")],
        output_size=1,
        output_activation="tanh",
        rng=make_rng("test", 0),
    )


@pytest.fixture
def multiclass_network():
    return Network(
        input_ids=["x", "y"],
        layer_specs=[_dense(n=4, act="relu")],
        output_size=3,
        output_activation="linear",
        is_multiclass=True,
        rng=make_rng("test_mc", 0),
    )


@pytest.fixture
def construct_input():
    def _fn(xx, yy):
        xx = np.atleast_1d(np.asarray(xx, dtype=np.float64))
        yy = np.atleast_1d(np.asarray(yy, dtype=np.float64))
        return np.stack([xx, yy], axis=1)
    return _fn