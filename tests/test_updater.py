"""apply_update: averaging, L1 death, grad clip."""
import numpy as np

from backend.core.nn import Network
from backend.core.optimizers import SGD
from backend.core.regularizations import L1
from backend.core.seed import make_rng
from backend.core.updater import apply_update


def _net():
    return Network(
        input_ids=["x", "y"],
        layer_specs=[{"type": "dense", "n": 2, "act": "linear",
                       "bias": True, "norm": "none", "norm_pos": "post"}],
        output_size=1, output_activation="linear",
        rng=make_rng("u", 0), init_zero=True,
    )


def _zero_bundle(net):
    gw = [np.zeros_like(w) if w is not None else None for w in net.weights]
    gb = [np.zeros_like(b) if b is not None else None for b in net.biases]
    gg = [np.zeros_like(g) if g is not None else None for g in net.gammas]
    gbe = [np.zeros_like(b) if b is not None else None for b in net.betas]
    grw = [np.zeros_like(w) if w is not None else None for w in net.router_weights]
    grb = [np.zeros_like(b) if b is not None else None for b in net.router_biases]
    gew, geb = [], []
    for l in range(net.num_layers):
        if net.expert_weights[l] is None:
            gew.append(None); geb.append(None)
        else:
            gew.append([[np.zeros_like(w) for w in e] for e in net.expert_weights[l]])
            geb.append([[np.zeros_like(b) for b in e] for e in net.expert_biases[l]])
    return gw, gb, gg, gbe, grw, grb, gew, geb


def _apply(net, opt, bundle, **kwargs):
    apply_update(net, opt, *bundle, **kwargs)


def test_averages_grads():
    net = _net()
    b = _zero_bundle(net)
    b[0][0][:] = 2.0        # sum=2, num_samples=2 → grad=1
    old = net.weights[0].copy()
    _apply(net, SGD(lr=1.0), b, num_samples=2,
            regularization=None, reg_rate=0.0)
    np.testing.assert_allclose(net.weights[0], old - 1.0)


def test_num_samples_zero_early_return():
    net = _net()
    old = net.weights[0].copy()
    _apply(net, SGD(lr=1.0), _zero_bundle(net), num_samples=0,
            regularization=None, reg_rate=0.0)
    np.testing.assert_array_equal(net.weights[0], old)


def test_l1_snaps_crossing_to_zero():
    net = _net()
    net.weights[0][0, 0] = 0.01
    b = _zero_bundle(net)
    b[0][0][0, 0] = -1.0        # strong negative grad → w wants positive
    _apply(net, SGD(lr=1.0), b, num_samples=1,
            regularization=L1, reg_rate=1.0)
    w = net.weights[0][0, 0]
    assert w >= 0.0
    if w == 0.0:
        assert net.dead_mask[0][0, 0]


def test_grad_clip_value():
    net = _net()
    b = _zero_bundle(net)
    b[0][0][:] = 100.0
    old = net.weights[0].copy()
    _apply(net, SGD(lr=1.0), b, num_samples=1,
            regularization=None, reg_rate=0.0,
            grad_clip_method="value", grad_clip_value=0.5)
    np.testing.assert_allclose(net.weights[0], old - 0.5, atol=1e-9)


def test_grad_clip_norm():
    net = _net()
    b = _zero_bundle(net)
    b[0][0][:] = 10.0
    old = net.weights[0].copy()
    _apply(net, SGD(lr=1.0), b, num_samples=1,
            regularization=None, reg_rate=0.0,
            grad_clip_method="norm", grad_clip_value=1.0)
    n = net.weights[0].size
    expected_step = 1.0 / np.sqrt(n)
    assert np.allclose(old - net.weights[0], expected_step, atol=1e-6)