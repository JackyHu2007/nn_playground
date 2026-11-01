"""Neural network with dense + MoE (top-k / soft) layers.

Architecture:
    input → [dense / moe] × N_hidden → output (dense, softmax-or-activation)

Dense layer semantics are unchanged from the previous version (norm pre/post,
bias toggle, residual after activation).

MoE layer (deep experts):
    z_r = h_in @ W_r.T + b_r                     (B, E)
    g = softmax(z_r / tau)                        (B, E)
    g_masked = g * MaskTopK(g, k)  if not soft    (B, E)
    g_used = g_masked / sum(g_masked)  if normalize and not soft
    expert_e(h) = h @ W_e^1.T + b_e^1 → act → ... → h @ W_e^D.T + b_e^D → act
    h = Σ_e g_used[:, e:e+1] * expert_e(h_in)     (B, n_out)

Aux loss (Switch style):
    f_e = mask_e.mean(0)  (top-k)  or  g_e.mean(0)  (soft)
    P_e = g_e.mean(0)
    L_aux = aux_coef * E * Σ f_e * P_e
Its gradient is injected into `d_g` before the softmax backward. With top-k +
normalize and k=1, the gradient is mathematically zero (g/g cancels), which is
why the UI forbids k=1.
"""
from __future__ import annotations

import numpy as np

from .activations import ACTIVATIONS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z_max = z.max(axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / e.sum(axis=-1, keepdims=True)


def _topk_mask(g: np.ndarray, k: int) -> np.ndarray:
    """(B, E) softmax probs -> (B, E) 0/1 mask keeping the top-k entries."""
    if k >= g.shape[-1]:
        return np.ones_like(g)
    idx = np.argpartition(-g, k - 1, axis=-1)[:, :k]
    mask = np.zeros_like(g)
    np.put_along_axis(mask, idx, 1.0, axis=-1)
    return mask


_MAX_NODES = 32
_MAX_EXPERT_DEPTH = 8


def _normalize_expert_layers(raw_layers) -> list[dict]:
    """Normalize a (possibly legacy) expert layer list into the canonical form."""
    layers = []
    if isinstance(raw_layers, list) and raw_layers:
        for ls in raw_layers:
            if not isinstance(ls, dict):
                ls = {}
            act = str(ls.get("act", "relu"))
            if act not in ACTIVATIONS:
                act = "relu"
            layers.append({
                "n": max(1, min(_MAX_NODES, int(ls.get("n", 4)))),
                "act": act,
                "bias": bool(ls.get("bias", True)),
            })
    if not layers:
        layers = [{"n": 4, "act": "relu", "bias": True}]
    if len(layers) > _MAX_EXPERT_DEPTH:
        layers = layers[:_MAX_EXPERT_DEPTH]
    return layers


def normalize_spec(spec: dict) -> dict:
    """Clamp a layer spec into a safe, fully-populated dict. Never raises.

    Accepts legacy ``expert: {n, act, bias}`` and migrates it to
    ``expert: {layers: [{n, act, bias}, ...]}``.
    """
    t = str(spec.get("type", "dense"))
    if t == "moe":
        E = max(2, min(16, int(spec.get("experts", 4))))
        soft = bool(spec.get("soft", False))
        if soft:
            k = E
        else:
            k = max(2, min(E, int(spec.get("top_k", 2))))
        exp = spec.get("expert", {}) or {}
        if "layers" in exp:
            raw = exp["layers"]
        else:
            # legacy single-layer format
            raw = [{
                "n": exp.get("n", 4),
                "act": exp.get("act", "relu"),
                "bias": exp.get("bias", True),
            }]
        return {
            "type": "moe",
            "experts": E,
            "top_k": k,
            "soft": soft,
            "router_bias": bool(spec.get("router_bias", True)),
            "router_temp": max(0.01, min(10.0, float(spec.get("router_temp", 1.0)))),
            "normalize": bool(spec.get("normalize", True)),
            "aux_coef": max(0.0, min(1.0, float(spec.get("aux_coef", 0.01)))),
            "expert": {"layers": _normalize_expert_layers(raw)},
        }
    # dense (default)
    norm = spec.get("norm", "none")
    pos = spec.get("norm_pos", "post")
    act = str(spec.get("act", "sigmoid"))
    if act not in ACTIVATIONS:
        act = "sigmoid"
    return {
        "type": "dense",
        "n": max(1, min(_MAX_NODES, int(spec.get("n", 4)))),
        "act": act,
        "bias": bool(spec.get("bias", True)),
        "norm": norm if norm in ("none", "layernorm", "rmsnorm") else "none",
        "norm_pos": pos if pos in ("pre", "post") else "post",
    }


def expert_output_dim(spec: dict) -> int:
    """Output width of a layer spec (used for shape bookkeeping)."""
    if spec["type"] == "dense":
        return spec["n"]
    return spec["expert"]["layers"][-1]["n"]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """Feed-forward network with dense + MoE layers and residual connections.

    Residuals are added AFTER the layer output (dense: after activation;
    MoE: after the weighted sum of experts).
    """

    def __init__(
        self,
        input_ids: list[str],
        layer_specs: list[dict],
        output_size: int,
        output_activation: str,
        output_bias: bool = True,
        is_multiclass: bool = False,
        init_zero: bool = False,
        rng: np.random.Generator | None = None,
        residuals: list[list[int]] | None = None,
        norm_eps: float = 1e-5,
    ):
        self.input_ids = list(input_ids)
        self._specs = [normalize_spec(s) for s in layer_specs]
        self._output_spec = {
            "type": "dense",
            "n": max(1, int(output_size)),
            "act": str(output_activation),
            "bias": bool(output_bias),
            "norm": "none",
            "norm_pos": "post",
        }
        self.is_multiclass = bool(is_multiclass)
        self.norm_eps = float(norm_eps)
        self.num_layers = len(self._specs) + 1

        # per-layer output sizes (includes input at index 0 and output at end)
        self.layer_sizes = [len(self.input_ids)]
        for spec in self._specs:
            self.layer_sizes.append(expert_output_dim(spec))
        self.layer_sizes.append(self._output_spec["n"])

        # activation object caches
        self.activations: list = []
        self.expert_activations: list = []
        for l in range(self.num_layers):
            spec = self._get_spec(l)
            if spec["type"] == "dense":
                self.activations.append(ACTIVATIONS[spec["act"]])
                self.expert_activations.append(None)
            else:
                self.activations.append(None)
                expert_layers = spec["expert"]["layers"]
                self.expert_activations.append([
                    [ACTIVATIONS[ls["act"]] for ls in expert_layers]
                    for _ in range(spec["experts"])
                ])

        if rng is None:
            rng = np.random.default_rng()
        if residuals is None:
            residuals = []

        N = self.num_layers
        self.weights: list = [None] * N
        self.biases: list = [None] * N
        self.dead_mask: list = [None] * N
        self.use_bias: list[bool] = [False] * N
        self.gammas: list = [None] * N
        self.betas: list = [None] * N
        self.gamma_dead: list = [None] * N
        self.beta_dead: list = [None] * N
        self.norm_types: list[str] = ["none"] * N
        self.norm_positions: list[str] = ["post"] * N

        self.router_weights: list = [None] * N
        self.router_biases: list = [None] * N
        self.router_dead_mask: list = [None] * N
        self.expert_weights: list = [None] * N
        self.expert_biases: list = [None] * N
        self.expert_use_bias: list = [None] * N
        self.expert_dead_mask: list = [None] * N

        # ---- init parameters ----
        for l in range(N):
            spec = self._get_spec(l)
            n_in = self.layer_sizes[l]

            if spec["type"] == "dense":
                n_out = spec["n"]
                w = np.zeros((n_out, n_in)) if init_zero else rng.uniform(-0.5, 0.5, (n_out, n_in))
                b = np.zeros(n_out) if init_zero else np.full(n_out, 0.1)
                self.weights[l] = w
                self.biases[l] = b
                self.dead_mask[l] = np.zeros_like(w, dtype=bool)
                self.use_bias[l] = spec["bias"]
                self.norm_types[l] = spec["norm"]
                self.norm_positions[l] = spec["norm_pos"]
                if spec["norm"] != "none":
                    dim = n_in if spec["norm_pos"] == "pre" else n_out
                    self.gammas[l] = np.ones(dim)
                    self.betas[l] = np.zeros(dim)
                    self.gamma_dead[l] = np.zeros(dim, dtype=bool)
                    self.beta_dead[l] = np.zeros(dim, dtype=bool)
            else:  # moe
                E = spec["experts"]
                expert_layers = spec["expert"]["layers"]
                D = len(expert_layers)
                w_r = np.zeros((E, n_in)) if init_zero else rng.uniform(-0.5, 0.5, (E, n_in))
                self.router_weights[l] = w_r
                self.router_dead_mask[l] = np.zeros_like(w_r, dtype=bool)
                self.router_biases[l] = (
                    np.zeros(E) if (init_zero or not spec["router_bias"])
                    else np.full(E, 0.1)
                )

                ew, eb, e_bias, e_dead = [], [], [], []
                for _ in range(E):
                    expert_w, expert_b, expert_bias_flag, expert_dead = [], [], [], []
                    prev_dim = n_in
                    for d in range(D):
                        dim = expert_layers[d]["n"]
                        w = (np.zeros((dim, prev_dim)) if init_zero
                             else rng.uniform(-0.5, 0.5, (dim, prev_dim)))
                        b = np.zeros(dim) if init_zero else np.full(dim, 0.1)
                        expert_w.append(w)
                        expert_b.append(b)
                        expert_bias_flag.append(expert_layers[d]["bias"])
                        expert_dead.append(np.zeros_like(w, dtype=bool))
                        prev_dim = dim
                    ew.append(expert_w)
                    eb.append(expert_b)
                    e_bias.append(expert_bias_flag)
                    e_dead.append(expert_dead)
                self.expert_weights[l] = ew
                self.expert_biases[l] = eb
                self.expert_use_bias[l] = e_bias
                self.expert_dead_mask[l] = e_dead

        # ---- residual links ----
        self.skip_links: list[list[int]] = [[] for _ in range(N)]
        for src, dst in residuals:
            if 0 <= src < dst <= N and self.layer_sizes[src] == self.layer_sizes[dst]:
                if self.is_multiclass and dst == N:  # no residual into softmax
                    continue
                l = dst - 1
                if src not in self.skip_links[l]:
                    self.skip_links[l].append(src)

        # ---- node ids ----
        self.layer_node_ids = [list(self.input_ids)]
        next_id = 1
        for l in range(1, len(self.layer_sizes)):
            n = self.layer_sizes[l]
            self.layer_node_ids.append([str(next_id + i) for i in range(n)])
            next_id += n

    # -------------------------------------------------------------- helpers
    def _get_spec(self, l: int) -> dict:
        if l < len(self._specs):
            return self._specs[l]
        return self._output_spec

    def _norm_forward(self, x, mode, l):
        if mode == "layernorm":
            mu = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            inv_std = 1.0 / np.sqrt(var + self.norm_eps)
            x_hat = (x - mu) * inv_std
            return x_hat * self.gammas[l] + self.betas[l], x_hat, inv_std
        if mode == "rmsnorm":
            ms = (x * x).mean(axis=-1, keepdims=True)
            inv_std = 1.0 / np.sqrt(ms + self.norm_eps)
            x_hat = x * inv_std
            return x_hat * self.gammas[l], x_hat, inv_std
        return x, None, None

    def _norm_backward(self, d_out, nc, l):
        mode = nc["mode"]
        x_hat = nc["x_hat"]
        inv_std = nc["inv_std"]
        gamma = self.gammas[l]
        grad_gamma = (d_out * x_hat).sum(axis=0)
        grad_beta = d_out.sum(axis=0) if mode == "layernorm" else np.zeros_like(self.betas[l])
        d_norm = d_out * gamma
        mean_dx = (d_norm * x_hat).mean(axis=-1, keepdims=True)
        if mode == "layernorm":
            mean_d = d_norm.mean(axis=-1, keepdims=True)
            d_x = inv_std * (d_norm - mean_d - x_hat * mean_dx)
        else:
            d_x = inv_std * (d_norm - x_hat * mean_dx)
        return d_x, grad_gamma, grad_beta

    # -------------------------------------------------------------- forward
    def _forward_dense(self, l, spec, h_prev):
        mode = self.norm_types[l]
        pos = self.norm_positions[l]
        h_in = h_prev
        if mode != "none" and pos == "pre":
            h_in, _, _ = self._norm_forward(h_prev, mode, l)
        z = h_in @ self.weights[l].T
        if self.use_bias[l]:
            z = z + self.biases[l]
        if mode != "none" and pos == "post":
            z, _, _ = self._norm_forward(z, mode, l)
        if self.is_multiclass and l == self.num_layers - 1:
            return _softmax(z)
        return self.activations[l].output(z)

    def _forward_moe_core(self, l, spec, h_prev, want_cache: bool):
        E = spec["experts"]
        tau = spec["router_temp"]
        D = len(self.expert_weights[l][0])

        z_r = h_prev @ self.router_weights[l].T
        if spec["router_bias"]:
            z_r = z_r + self.router_biases[l]
        g = _softmax(z_r / tau)

        if spec["soft"]:
            mask = np.ones_like(g)
            g_masked = g
        else:
            mask = _topk_mask(g, spec["top_k"])
            g_masked = g * mask

        if spec["soft"] or not spec["normalize"]:
            g_used = g_masked
        else:
            s = g_masked.sum(axis=-1, keepdims=True)
            s = np.where(s > 0, s, 1.0)
            g_used = g_masked / s

        expert_zs = []
        expert_outs = []
        expert_layer_inputs = []
        expert_internal_hs = []
        for e in range(E):
            h = h_prev
            zs, ins, hs = [], [], [h_prev]
            for d in range(D):
                ins.append(h)
                z = h @ self.expert_weights[l][e][d].T
                if self.expert_use_bias[l][e][d]:
                    z = z + self.expert_biases[l][e][d]
                h = self.expert_activations[l][e][d].output(z)
                zs.append(z)
                hs.append(h)
            expert_zs.append(zs)
            expert_layer_inputs.append(ins)
            expert_internal_hs.append(hs)
            expert_outs.append(h)

        combined = np.zeros_like(expert_outs[0])
        for e in range(E):
            combined = combined + g_used[:, e:e + 1] * expert_outs[e]

        if not want_cache:
            return combined, None
        cache = {
            "h_in": h_prev,
            "z_r": z_r,
            "g": g,
            "g_masked": g_masked,
            "g_used": g_used,
            "mask": mask,
            "expert_zs": expert_zs,
            "expert_layer_inputs": expert_layer_inputs,
            "expert_outs": expert_outs,
            "expert_internal_hs": expert_internal_hs,
        }
        return combined, cache

    def _forward_layer(self, l, h_prev, out_cache):
        spec = self._get_spec(l)
        if spec["type"] == "moe":
            h, _ = self._forward_moe_core(l, spec, h_prev, want_cache=False)
        else:
            h = self._forward_dense(l, spec, h_prev)
        for src in self.skip_links[l]:
            h = h + out_cache[src]
        return h

    def forward_all_layers(self, X):
        single = X.ndim == 1
        X2 = X[None, :] if single else X
        out_cache = [X2.astype(np.float64, copy=False)]
        for l in range(self.num_layers):
            out_cache.append(self._forward_layer(l, out_cache[l], out_cache))
        return [o[0] for o in out_cache] if single else out_cache

    def forward(self, X):
        return self.forward_all_layers(X)[-1]

    def _forward_dense_cached(self, l, spec, h_prev):
        mode = self.norm_types[l]
        pos = self.norm_positions[l]
        nc = {"mode": mode, "position": pos, "x_hat": None, "inv_std": None, "h_in": None}

        h_in = h_prev
        if mode != "none" and pos == "pre":
            h_in, x_hat, inv_std = self._norm_forward(h_prev, mode, l)
            nc["x_hat"], nc["inv_std"], nc["h_in"] = x_hat, inv_std, h_in

        z = h_in @ self.weights[l].T
        if self.use_bias[l]:
            z = z + self.biases[l]

        if mode != "none" and pos == "post":
            z_n, x_hat, inv_std = self._norm_forward(z, mode, l)
            nc["x_hat"], nc["inv_std"] = x_hat, inv_std
            act_input = z_n
        else:
            act_input = z

        if self.is_multiclass and l == self.num_layers - 1:
            act_cache = z
            h = _softmax(z)
        else:
            act_cache = act_input
            h = self.activations[l].output(act_input)

        return h, {
            "type": "dense",
            "h_in": h_in,
            "norm_cache": nc,
            "act_cache": act_cache,
        }

    def forward_with_cache(self, X):
        """Returns (last_layer_output, layer_cache_list, out_cache_list)."""
        single = X.ndim == 1
        X2 = X[None, :] if single else X
        out_cache = [X2.astype(np.float64, copy=False)]
        layer_cache = []

        for l in range(self.num_layers):
            spec = self._get_spec(l)
            h_prev = out_cache[l]

            if spec["type"] == "moe":
                h, lc = self._forward_moe_core(l, spec, h_prev, want_cache=True)
                lc["type"] = "moe"
            else:
                h, lc = self._forward_dense_cached(l, spec, h_prev)

            for src in self.skip_links[l]:
                h = h + out_cache[src]
            lc["h_out"] = h

            out_cache.append(h)
            layer_cache.append(lc)

        if single:
            out_cache = [o[0] for o in out_cache]
        return out_cache[-1], layer_cache, out_cache

    # -------------------------------------------------------------- backward
    def _backward_dense(self, l, spec, lc, d_h_out):
        act_cache = lc["act_cache"]
        nc = lc["norm_cache"]
        h_in = lc["h_in"]

        if self.is_multiclass and l == self.num_layers - 1:
            d_z_act = d_h_out
        else:
            d_z_act = d_h_out * self.activations[l].der(act_cache)

        mode = self.norm_types[l]
        pos = self.norm_positions[l]
        gg = np.zeros_like(self.gammas[l]) if self.gammas[l] is not None else None
        gb = np.zeros_like(self.betas[l]) if self.betas[l] is not None else None

        if mode != "none" and pos == "post":
            d_z_lin, gg, gb = self._norm_backward(d_z_act, nc, l)
        else:
            d_z_lin = d_z_act

        grad_w = d_z_lin.T @ h_in
        grad_b = d_z_lin.sum(axis=0) if self.use_bias[l] else np.zeros_like(self.biases[l])
        d_h_in = d_z_lin @ self.weights[l]

        if mode != "none" and pos == "pre":
            d_h_prev, gg, gb = self._norm_backward(d_h_in, nc, l)
        else:
            d_h_prev = d_h_in

        return d_h_prev, grad_w, grad_b, gg, gb

    def _backward_moe(self, l, spec, lc, d_h_out):
        E = spec["experts"]
        soft = spec["soft"]
        tau = spec["router_temp"]
        alpha = float(spec.get("aux_coef", 0.0))

        h_in = lc["h_in"]
        g = lc["g"]
        g_masked = lc["g_masked"]
        g_used = lc["g_used"]
        mask = lc["mask"]
        expert_zs = lc["expert_zs"]  # E × D arrays
        expert_layer_inputs = lc["expert_layer_inputs"]  # E × D arrays
        expert_outs = lc["expert_outs"]  # E arrays (B, n_out)
        batch = d_h_out.shape[0]
        D = len(expert_zs[0])

        # 1. grads for each expert output and for g_used
        d_expert_outs = [g_used[:, e:e + 1] * d_h_out for e in range(E)]
        d_g_used = np.stack(
            [(d_h_out * expert_outs[e]).sum(axis=-1) for e in range(E)], axis=-1
        )

        # 2. through normalize
        if soft or not spec["normalize"]:
            d_g_masked = d_g_used
        else:
            S = g_masked.sum(axis=-1, keepdims=True)
            S = np.where(S > 0, S, 1.0)
            proj = (d_g_used * g_used).sum(axis=-1, keepdims=True)
            d_g_masked = (d_g_used - proj) / S

        # 3. through top-k mask
        d_g = d_g_masked if soft else d_g_masked * mask

        # 4. aux loss grad (about g)
        aux_loss = 0.0
        if alpha > 0:
            f = g.mean(axis=0) if soft else mask.mean(axis=0)
            P = g.mean(axis=0)
            aux_loss = float(alpha * E * (f * P).sum())
            if soft:
                aux_grad = alpha * E * (f + P) / batch
            else:
                aux_grad = alpha * E * f / batch
            d_g = d_g + aux_grad[None, :]

        # 5. through softmax (z_r / tau)
        s = (d_g * g).sum(axis=-1, keepdims=True)
        d_z_r = g * (d_g - s) / tau

        # 6. router grads
        grad_router_w = d_z_r.T @ h_in
        grad_router_b = (
            d_z_r.sum(axis=0) if spec["router_bias"]
            else np.zeros_like(self.router_biases[l])
        )
        d_h_in = d_z_r @ self.router_weights[l]

        # 7. expert grads (deep experts: one chain per expert)
        grad_expert_w = []  # list of E lists of D arrays
        grad_expert_b = []
        for e in range(E):
            ins = expert_layer_inputs[e]  # D arrays (h_{d-1} for each d)
            zs = expert_zs[e]  # D pre-activations
            gw_e = [None] * D
            gb_e = [None] * D
            d_h = d_expert_outs[e]  # gradient at expert output
            for d in range(D - 1, -1, -1):
                d_z = d_h * self.expert_activations[l][e][d].der(zs[d])
                gw_e[d] = d_z.T @ ins[d]
                gb_e[d] = (
                    d_z.sum(axis=0) if self.expert_use_bias[l][e][d]
                    else np.zeros_like(self.expert_biases[l][e][d])
                )
                if d > 0:
                    d_h = d_z @ self.expert_weights[l][e][d]
                else:
                    d_h_in = d_h_in + d_z @ self.expert_weights[l][e][d]
            grad_expert_w.append(gw_e)
            grad_expert_b.append(gb_e)

        return d_h_in, grad_router_w, grad_router_b, grad_expert_w, grad_expert_b, aux_loss

    def backward(self, layer_cache, out_cache, targets, error_fn):
        """Returns a dict of gradients.

        Shapes:
            weights[l]         : ndarray or None
            biases[l]          : ndarray or None
            gammas[l]/betas[l] : ndarray or None
            router_weights[l]  : ndarray or None
            router_biases[l]   : ndarray or None
            expert_weights[l]  : list[E][D] of ndarray or None
            expert_biases[l]   : list[E][D] of ndarray or None
            aux_losses[l]      : float
        """
        batch = out_cache[0].shape[0]

        if self.is_multiclass:
            idx = targets.astype(int).ravel()
            onehot = np.zeros_like(out_cache[-1])
            onehot[np.arange(batch), idx] = 1.0
            d_a = out_cache[-1] - onehot
        else:
            tgt = targets.reshape(batch, -1) if targets.ndim == 1 else targets
            d_a = error_fn.der(out_cache[-1], tgt)

        N = self.num_layers
        g_w = [None] * N
        g_b = [None] * N
        g_gamma = [None] * N
        g_beta = [None] * N
        g_rw = [None] * N
        g_rb = [None] * N
        g_ew = [None] * N
        g_eb = [None] * N
        aux = [0.0] * N
        d_h_list = [np.zeros_like(o) for o in out_cache]

        for l in range(N - 1, -1, -1):
            d_h_out = d_a if l == N - 1 else d_h_list[l + 1]
            for src in self.skip_links[l]:
                d_h_list[src] += d_h_out

            spec = self._get_spec(l)
            lc = layer_cache[l]

            if spec["type"] == "moe":
                d_h_prev, rw, rb, ew, eb, al = self._backward_moe(l, spec, lc, d_h_out)
                g_rw[l], g_rb[l], g_ew[l], g_eb[l], aux[l] = rw, rb, ew, eb, al
            else:
                d_h_prev, gw, gb, gg, gbe = self._backward_dense(l, spec, lc, d_h_out)
                g_w[l], g_b[l], g_gamma[l], g_beta[l] = gw, gb, gg, gbe

            d_h_list[l] += d_h_prev

        return {
            "weights": g_w,
            "biases": g_b,
            "gammas": g_gamma,
            "betas": g_beta,
            "router_weights": g_rw,
            "router_biases": g_rb,
            "expert_weights": g_ew,
            "expert_biases": g_eb,
            "aux_losses": aux,
        }

    # -------------------------------------------------------------- loss
    def compute_loss(self, X, targets, error_fn):
        out = self.forward(X)
        if out.ndim == 1:
            out = out[None, :]
        if self.is_multiclass:
            idx = targets.astype(int).ravel()
            eps = 1e-12
            return float(-np.mean(np.log(out[np.arange(len(idx)), idx] + eps)))
        tgt = targets.reshape(out.shape) if targets.ndim == 1 else targets
        return float(error_fn.error(out, tgt).mean())

    def compute_aux_losses_from_cache(self, layer_cache) -> tuple[float, dict[int, float]]:
        """Cheap post-hoc aux loss report from a forward_with_cache() result."""
        per_layer: dict[int, float] = {}
        total = 0.0
        for l in range(self.num_layers):
            spec = self._get_spec(l)
            if spec["type"] != "moe":
                continue
            alpha = float(spec.get("aux_coef", 0.0))
            if alpha <= 0:
                continue
            lc = layer_cache[l]
            E = spec["experts"]
            g = lc["g"]
            mask = lc["mask"]
            f = g.mean(axis=0) if spec["soft"] else mask.mean(axis=0)
            P = g.mean(axis=0)
            aux = float(alpha * E * (f * P).sum())
            per_layer[l] = aux
            total += aux
        return total, per_layer

    def forward_with_probe(self, X):
        """Like forward_all_layers but also returns MoE routing info.

        Returns:
            out_cache: list of per-layer outputs (same as forward_all_layers)
            probe: dict[layer] -> {
                "argmax":           (B,),
                "weights":          (B, E),
                "rawProbs":         (B, E),
                "expertOuts":       list of E (B, n_out),
                "expertInternalHs": list of E lists of D+1 (B, n_d), # include input h0
            }
        """
        X2 = X[None, :] if X.ndim == 1 else X
        out_cache = [X2.astype(np.float64, copy=False)]
        probe: dict[int, dict] = {}

        for l in range(self.num_layers):
            spec = self._get_spec(l)
            h_prev = out_cache[l]
            if spec["type"] == "moe":
                h, lc = self._forward_moe_core(l, spec, h_prev, want_cache=True)
                probe[l] = {
                    "argmax": np.argmax(lc["g_masked"], axis=-1),
                    "weights": lc["g_used"],
                    "rawProbs": lc["g"],
                    "expertOuts": lc["expert_outs"],
                    "expertInternalHs": lc["expert_internal_hs"],
                }
            else:
                h = self._forward_dense(l, spec, h_prev)
            for src in self.skip_links[l]:
                h = h + out_cache[src]
            out_cache.append(h)

        return out_cache, probe