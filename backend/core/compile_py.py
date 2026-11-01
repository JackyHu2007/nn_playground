from __future__ import annotations
from .nn import Network

_INPUT_VAR = {
    "x": "X1", "y": "X2", "xTimesY": "X1X2",
    "xSquared": "X1_squared", "ySquared": "X2_squared",
}


def _input_var(nid: str) -> str:
    return _INPUT_VAR.get(nid, f"X_{nid}")


def _fmt(x: float) -> str:
    x = float(x)
    if x == 0.0:
        return "0.0"
    return f"{x:.12g}"


def _list_literal(values) -> str:
    return "[" + ", ".join(_fmt(float(v)) for v in values) + "]"


def _dense_linear_expr(weights, bias, use_bias: bool, in_var: str) -> str:
    """Return a list literal expression computing the pre-activation z."""
    n_out = len(weights)
    items = []
    for i in range(n_out):
        terms = []
        if use_bias:
            terms.append(_fmt(float(bias[i])))
        row = weights[i]
        for j, w in enumerate(row):
            if float(w) != 0.0:
                terms.append(f"({_fmt(w)} * {in_var}[{j}])")
        if not terms:
            terms.append("0.0")
        items.append(" + ".join(terms))
    return "[" + ", ".join(items) + "]"


def _moe_block_code(l, spec, network, input_var: str, output_var: str) -> list[str]:
    E = spec["experts"]
    tau = spec["router_temp"]
    k = spec["top_k"]
    soft = spec["soft"]
    normalize = spec["normalize"]
    expert_layers = spec["expert"]["layers"]
    D = len(expert_layers)

    lines = [f"    # Layer {l + 1} (MoE, {E} experts × {D} hidden layers)\n"]

    # router logits
    rw = network.router_weights[l]
    rb = network.router_biases[l]
    router_bias = spec["router_bias"]

    items = []
    for e in range(E):
        terms = []
        if router_bias:
            terms.append(_fmt(float(rb[e])))
        for j, w in enumerate(rw[e]):
            if float(w) != 0.0:
                terms.append(f"({_fmt(w)} * {input_var}[{j}])")
        if not terms:
            terms.append("0.0")
        items.append(" + ".join(terms))
    lines.append(f"    z_r = [{', '.join(items)}]\n")
    lines.append(f"    z_r = [v / {_fmt(tau)} for v in z_r]\n")
    lines.append("    _m = max(z_r)\n")
    lines.append("    _ex = [math.exp(v - _m) for v in z_r]\n")
    lines.append("    _s = sum(_ex)\n")
    lines.append("    g = [v / _s for v in _ex]\n")

    if not soft:
        lines.append(f"    _pairs = sorted(range({E}), key=lambda i: -g[i])[:{k}]\n")
        lines.append(f"    _mask = [0.0] * {E}\n")
        lines.append("    for _i in _pairs:\n        _mask[_i] = 1.0\n")
        lines.append("    g_masked = [g[i] * _mask[i] for i in range(" + str(E) + ")]\n")
        if normalize:
            lines.append("    _gs = sum(g_masked)\n")
            lines.append("    _gs = _gs if _gs > 0 else 1.0\n")
            lines.append("    g_used = [v / _gs for v in g_masked]\n")
        else:
            lines.append("    g_used = g_masked\n")
    else:
        lines.append("    g_used = g\n")

    # expert internal layers (shared structure, independent weights per expert)
    from .activations import ACTIVATIONS
    for e in range(E):
        cur = "prev"
        for d in range(D):
            ew = network.expert_weights[l][e][d]
            eb = network.expert_biases[l][e][d]
            e_use_b = network.expert_use_bias[l][e][d]
            expr = _dense_linear_expr(ew, eb, e_use_b, cur)
            lines.append(f"    z_e{e}_{d} = {expr}\n")
            act = ACTIVATIONS[expert_layers[d]["act"]]
            act_expr = act.compile_to_py("v")
            lines.append(f"    h_e{e}_{d} = [{act_expr} for v in z_e{e}_{d}]\n")
            cur = f"h_e{e}_{d}"
        lines.append(f"    a{e} = {cur}\n")

    n_out = expert_layers[-1]["n"]
    combine = " + ".join(f"g_used[{e}] * a{e}[i]" for e in range(E))
    lines.append(
        f"    {output_var} = [{combine} for i in range({n_out})]\n"
    )
    return lines


def compile_network_to_py(network: Network) -> str:
    input_names = [_input_var(nid) for nid in network.input_ids]
    out = ["import math\n\n"]

    needs_norm = any(nt != "none" for nt in network.norm_types)
    if needs_norm:
        out.append(
            "def _ln(z, g, b, eps):\n"
            "    n = len(z)\n"
            "    mu = sum(z) / n\n"
            "    var = sum((v - mu) ** 2 for v in z) / n\n"
            "    inv = 1.0 / math.sqrt(var + eps)\n"
            "    return [(g[i] * (z[i] - mu) * inv + b[i]) for i in range(n)]\n\n"
        )
        out.append(
            "def _rms(z, g, eps):\n"
            "    n = len(z)\n"
            "    ms = sum(v * v for v in z) / n\n"
            "    inv = 1.0 / math.sqrt(ms + eps)\n"
            "    return [g[i] * z[i] * inv for i in range(n)]\n\n"
        )

    out.append(f"def forward({', '.join(input_names)}):\n")
    out.append('    """Forward pass (supports dense + MoE layers)."""\n')
    out.append(f"    prev = [{', '.join(input_names)}]\n")
    out.append("    layers = [prev]\n")

    eps_str = repr(float(network.norm_eps))

    for l in range(network.num_layers):
        spec = network._get_spec(l)

        if spec["type"] == "moe":
            moe_lines = _moe_block_code(l, spec, network, "prev", "h_moe")
            out.extend(moe_lines)
            if network.skip_links[l]:
                parts = " + ".join(f"layers[{src}][i]" for src in network.skip_links[l])
                n_out = spec["expert"]["layers"][-1]["n"]
                out.append(
                    f"    prev = [h_moe[i] + {parts} for i in range({n_out})]\n"
                )
            else:
                out.append("    prev = h_moe\n")
            out.append("    layers.append(prev)\n")
            continue

        # ---- dense ----
        n_in = network.layer_sizes[l]
        n_out = network.layer_sizes[l + 1]
        mode = network.norm_types[l]
        pos = network.norm_positions[l]
        out.append(f"    # Layer {l + 1}\n")

        lin_input = "prev"
        if mode != "none" and pos == "pre":
            if mode == "layernorm":
                out.append(
                    f"    h_in = _ln(prev, {_list_literal(network.gammas[l])}, "
                    f"{_list_literal(network.betas[l])}, {eps_str})\n"
                )
            else:
                out.append(
                    f"    h_in = _rms(prev, {_list_literal(network.gammas[l])}, "
                    f"{eps_str})\n"
                )
            lin_input = "h_in"

        expr = _dense_linear_expr(
            network.weights[l], network.biases[l], network.use_bias[l], lin_input
        )
        out.append(f"    z = {expr}\n")

        if mode != "none" and pos == "post":
            if mode == "layernorm":
                out.append(
                    f"    z = _ln(z, {_list_literal(network.gammas[l])}, "
                    f"{_list_literal(network.betas[l])}, {eps_str})\n"
                )
            else:
                out.append(
                    f"    z = _rms(z, {_list_literal(network.gammas[l])}, "
                    f"{eps_str})\n"
                )

        if network.is_multiclass and l == network.num_layers - 1:
            out.append("    m = max(z)\n")
            out.append("    ex = [math.exp(v - m) for v in z]\n")
            out.append("    s = sum(ex)\n")
            out.append("    prev = [v / s for v in ex]\n")
        else:
            act_expr = network.activations[l].compile_to_py("v")
            out.append(f"    act_out = [{act_expr} for v in z]\n")
            if network.skip_links[l]:
                parts = " + ".join(f"layers[{src}][i]" for src in network.skip_links[l])
                out.append(
                    f"    prev = [act_out[i] + {parts} for i in range({n_out})]\n"
                )
            else:
                out.append("    prev = act_out\n")
        out.append("    layers.append(prev)\n")

    if network.is_multiclass:
        out.append("    # prev holds per-class probabilities; max(prev) is the class.\n")
        out.append("    return prev\n")
    else:
        out.append("    return prev[0]\n")
    return "".join(out)


def generated_forward(network: Network):
    ns = {}
    exec(compile_network_to_py(network), ns)
    return ns["forward"]