"""Decision-boundary grid + MoE probe grids."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .nn import Network


def compute_node_outputs(network: Network, X: np.ndarray) -> dict[str, np.ndarray]:
    if X.ndim == 1:
        X = X[None, :]
    outs = network.forward_all_layers(X)
    result: dict[str, np.ndarray] = {}
    for l, ids in enumerate(network.layer_node_ids):
        layer_out = outs[l]
        for i, nid in enumerate(ids):
            result[nid] = layer_out[:, i]
    return result


def compute_boundary(
    network: Network,
    construct_input: Callable[[np.ndarray, np.ndarray], np.ndarray],
    density: int = 100,
    x_domain: tuple[float, float] = (-6.0, 6.0),
    y_domain: tuple[float, float] = (-6.0, 6.0),
) -> tuple[dict[str, np.ndarray], dict[int, dict], dict[int, dict]]:
    """Compute node boundaries + MoE routing maps + per-depth expert output grids.

    Returns:
        node_boundaries: dict[node_id] -> (D, D)
        moe_probe:       dict[layer] -> {"argmax": ..., "weights": ..., "rawProbs": ...}
        expert_outputs:  dict[layer] -> {"depth": D, "outs": [ (E, D, D, n_d) ]_d=0..D-1 }
    """
    xs = np.linspace(x_domain[0], x_domain[1], density)
    ys = np.linspace(y_domain[1], y_domain[0], density)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    X = construct_input(xx.ravel(), yy.ravel())

    out_cache, probe = network.forward_with_probe(X)

    # ---- standard node boundaries ----
    node_boundaries: dict[str, np.ndarray] = {}
    for l, ids in enumerate(network.layer_node_ids):
        layer_out = out_cache[l]
        for i, nid in enumerate(ids):
            node_boundaries[nid] = layer_out[:, i].reshape(density, density)

    # ---- MoE probe ----
    moe_probe: dict[int, dict] = {}
    for l, p in probe.items():
        E = p["weights"].shape[1]
        moe_probe[l] = {
            "argmax": p["argmax"].reshape(density, density),
            "weights": p["weights"].T.reshape(E, density, density),
            "rawProbs": p["rawProbs"].T.reshape(E, density, density),
        }

    # ---- expert output grids (per depth) ----
    expert_outputs: dict[int, dict] = {}
    for l, p in probe.items():
        E = len(p["expertOuts"])
        internal = p["expertInternalHs"]                  # [E][D+1] of (N, n_d)
        D = len(internal[0]) - 1                          # skip input
        outs_by_depth = []
        for d in range(D):
            n_out = internal[0][d + 1].shape[1]
            arr = np.stack([internal[e][d + 1] for e in range(E)], axis=0)  # (E, N, n_out)
            outs_by_depth.append(arr.reshape(E, density, density, n_out))
        expert_outputs[l] = {"depth": D, "outs": outs_by_depth}

    return node_boundaries, moe_probe, expert_outputs


def reduce_matrix(matrix: np.ndarray, factor: int) -> np.ndarray:
    h, w = matrix.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError("matrix dims must be divisible by factor")
    return matrix.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
