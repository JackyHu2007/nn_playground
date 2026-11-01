from __future__ import annotations
import numpy as np
from .nn import Network
from .optimizers import Optimizer
from .regularizations import L1, RegularizationFunction


def apply_update(
        network: Network,
        optimizer: Optimizer,
        grad_sum_w: list,
        grad_sum_b: list,
        grad_sum_gamma: list,
        grad_sum_beta: list,
        grad_sum_rw: list,
        grad_sum_rb: list,
        grad_sum_ew: list,
        grad_sum_eb: list,
        num_samples: int,
        regularization: RegularizationFunction | None,
        reg_rate: float,
        grad_clip_method: str = "none",
        grad_clip_value: float = 1.0,
) -> None:
    if num_samples <= 0:
        return
    inv_n = 1.0 / float(num_samples)

    clip_applied = False
    # =========================================================
    # Gradient Clipping
    # =========================================================
    if grad_clip_method != "none" and grad_clip_value > 0.0:
        all_grads = []

        def _collect(g_list):
            for g in g_list:
                if g is None: continue
                if isinstance(g, list):
                    _collect(g)
                else:
                    all_grads.append(g)

        _collect(grad_sum_w)
        _collect(grad_sum_b)
        _collect(grad_sum_gamma)
        _collect(grad_sum_beta)
        _collect(grad_sum_rw)
        _collect(grad_sum_rb)
        _collect(grad_sum_ew)
        _collect(grad_sum_eb)

        if all_grads:
            clip_applied = True
            # 1. First, convert all accumulated gradients into average gradients (divide by num_samples)
            for g in all_grads:
                g *= inv_n

            # 2. Perform clipping
            if grad_clip_method == "value":
                for g in all_grads:
                    np.clip(g, -grad_clip_value, grad_clip_value, out=g)
            elif grad_clip_method == "norm":
                total_norm_sq = sum(float(np.sum(g ** 2)) for g in all_grads)
                total_norm = np.sqrt(total_norm_sq)
                if total_norm > grad_clip_value:
                    clip_coef = grad_clip_value / (total_norm + 1e-6)
                    for g in all_grads:
                        g *= clip_coef

    # =========================================================

    def update_weight_params(params, grads, dead_mask, key_prefix):
        g = grads if clip_applied else grads * inv_n
        if regularization is not None and reg_rate != 0.0:
            g = g + reg_rate * regularization.der(params)
        active = ~dead_mask
        new_p = optimizer.step(params, g, active, key_prefix)
        if regularization is L1 and reg_rate != 0.0:
            crossed = active & (params * new_p < 0.0) & (params != 0.0)
            if np.any(crossed):
                new_p = np.where(crossed, 0.0, new_p)
                dead_mask |= crossed
        return new_p, dead_mask

    def update_plain_params(params, grads, key):
        g = grads if clip_applied else grads * inv_n
        active = np.ones_like(params, dtype=bool)
        return optimizer.step(params, g, active, key)

    for l in range(network.num_layers):
        # ---- dense layer params ----
        if grad_sum_w[l] is not None:
            network.weights[l], network.dead_mask[l] = update_weight_params(
                network.weights[l], grad_sum_w[l], network.dead_mask[l], ("w", l)
            )
            if network.use_bias[l]:
                network.biases[l] = update_plain_params(
                    network.biases[l], grad_sum_b[l], ("b", l)
                )
            nt = network.norm_types[l]
            if nt != "none" and network.gammas[l] is not None:
                network.gammas[l] = update_plain_params(
                    network.gammas[l], grad_sum_gamma[l], ("gamma", l)
                )
                if nt == "layernorm":
                    network.betas[l] = update_plain_params(
                        network.betas[l], grad_sum_beta[l], ("beta", l)
                    )

        # ---- MoE router + deep experts ----
        if grad_sum_rw[l] is not None:
            network.router_weights[l] = update_plain_params(
                network.router_weights[l], grad_sum_rw[l], ("rw", l)
            )
            if network._get_spec(l).get("router_bias", True):
                network.router_biases[l] = update_plain_params(
                    network.router_biases[l], grad_sum_rb[l], ("rb", l)
                )
            for e in range(len(network.expert_weights[l])):
                D = len(network.expert_weights[l][e])
                for d in range(D):
                    network.expert_weights[l][e][d], network.expert_dead_mask[l][e][d] = \
                        update_weight_params(
                            network.expert_weights[l][e][d],
                            grad_sum_ew[l][e][d],
                            network.expert_dead_mask[l][e][d],
                            ("ew", l, e, d),
                        )
                    if network.expert_use_bias[l][e][d]:
                        network.expert_biases[l][e][d] = update_plain_params(
                            network.expert_biases[l][e][d],
                            grad_sum_eb[l][e][d],
                            ("eb", l, e, d),
                        )