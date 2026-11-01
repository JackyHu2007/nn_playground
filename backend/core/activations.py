"""Activation functions.

Each function is defined with:
- output(x):   NumPy-friendly elementwise forward
- der(x):      derivative w.r.t. its input (pre-activation), NumPy-friendly
- compile_to_py(s): pure-Python (scalar) expression for code generation
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ActivationFunction:
    output: Callable[[np.ndarray], np.ndarray]
    der: Callable[[np.ndarray], np.ndarray]
    compile_to_py: Callable[[str], str]


def _as_f64(x):
    return np.asarray(x, dtype=np.float64)


# ---------- TANH ----------
TANH = ActivationFunction(
    output=lambda x: np.tanh(_as_f64(x)),
    der=lambda x: 1.0 - np.tanh(_as_f64(x)) ** 2,
    compile_to_py=lambda s: f"math.tanh({s})",
)

# ---------- RELU ----------
RELU = ActivationFunction(
    output=lambda x: np.maximum(0.0, _as_f64(x)),
    der=lambda x: (_as_f64(x) > 0).astype(np.float64),
    compile_to_py=lambda s: f"max(0.0, {s})",
)

# ---------- LEAKY RELU ----------
LEAKY_RELU = ActivationFunction(
    output=lambda x: np.where(_as_f64(x) > 0, _as_f64(x), 0.1 * _as_f64(x)),
    der=lambda x: np.where(_as_f64(x) > 0, 1.0, 0.1),
    compile_to_py=lambda s: f"max(0.0, {s}) + 0.1 * min(0.0, {s})",
)

# ---------- ELU (alpha = 0.1) ----------
def _elu_out(x):
    x = _as_f64(x)
    return np.where(x > 0, x, 0.1 * (np.exp(np.minimum(x, 0.0)) - 1.0))


def _elu_der(x):
    x = _as_f64(x)
    return np.where(x > 0, 1.0, 0.1 * np.exp(np.minimum(x, 0.0)))


ELU = ActivationFunction(
    output=_elu_out,
    der=_elu_der,
    compile_to_py=lambda s: f"({s} if {s} > 0 else 0.1 * (math.exp({s}) - 1.0))",
)

# ---------- SELU ----------
_SELU_SCALE = 1.0507009873554804934193349852946
_SELU_ALPHA = 1.6732632423543772848170429916717


def _selu_out(x):
    x = _as_f64(x)
    return _SELU_SCALE * np.where(x > 0, x, _SELU_ALPHA * (np.exp(np.minimum(x, 0.0)) - 1.0))


def _selu_der(x):
    x = _as_f64(x)
    return _SELU_SCALE * np.where(x > 0, 1.0, _SELU_ALPHA * np.exp(np.minimum(x, 0.0)))


SELU = ActivationFunction(
    output=_selu_out,
    der=_selu_der,
    compile_to_py=lambda s: (
        f"({_SELU_SCALE:.10f} * ({s} if {s} > 0 else "
        f"{_SELU_ALPHA:.10f} * (math.exp({s}) - 1.0)))"
    ),
)

# ---------- SIGMOID ----------
def _sigmoid(x):
    x = _as_f64(x)
    pos = 1.0 / (1.0 + np.exp(-np.minimum(x, 500.0)))
    neg = np.exp(np.maximum(x, -500.0)) / (1.0 + np.exp(np.maximum(x, -500.0)))
    return np.where(x >= 0, pos, neg)


SIGMOID = ActivationFunction(
    output=_sigmoid,
    der=lambda x: _sigmoid(x) * (1.0 - _sigmoid(x)),
    compile_to_py=lambda s: f"(1.0 / (1.0 + math.exp(-({s}))))",
)

# ---------- SWISH ----------
def _swish_out(x):
    x = _as_f64(x)
    return x * _sigmoid(x)


def _swish_der(x):
    x = _as_f64(x)
    s = _sigmoid(x)
    return s + x * s * (1.0 - s)


SWISH = ActivationFunction(
    output=_swish_out,
    der=_swish_der,
    compile_to_py=lambda s: f"(({s}) * (1.0 / (1.0 + math.exp(-({s})))))",
)

# ---------- GELU (tanh approximation) ----------
_GELU_C = math.sqrt(2.0 / math.pi)


def _gelu_out(x):
    x = _as_f64(x)
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x ** 3)))


def _gelu_der(x):
    x = _as_f64(x)
    inner = _GELU_C * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * _GELU_C * (1.0 + 3.0 * 0.044715 * x * x)


GELU = ActivationFunction(
    output=_gelu_out,
    der=_gelu_der,
    compile_to_py=lambda s: (
        f"(0.5 * ({s}) * (1.0 + math.tanh({_GELU_C:.10f} * "
        f"(({s}) + 0.044715 * ({s})**3))))"
    ),
)

# ---------- SOFTPLUS ----------
def _softplus_out(x):
    x = _as_f64(x)
    # max(x, 0) + log(1 + exp(-|x|))  (numerically stable)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


SOFTPLUS = ActivationFunction(
    output=_softplus_out,
    der=_sigmoid,
    compile_to_py=lambda s: f"(max({s}, 0.0) + math.log1p(math.exp(-abs({s}))))",
)

# ---------- MISH ----------
def _mish_out(x):
    x = _as_f64(x)
    sp = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    return x * np.tanh(sp)


def _mish_der(x):
    x = _as_f64(x)
    sp = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    t = np.tanh(sp)
    return t + x * (1.0 - t * t) * _sigmoid(x)


MISH = ActivationFunction(
    output=_mish_out,
    der=_mish_der,
    compile_to_py=lambda s: (
        f"(({s}) * math.tanh(max({s}, 0.0) + math.log1p(math.exp(-abs({s})))))"
    ),
)

# ---------- SINE ----------
SINE = ActivationFunction(
    output=lambda x: np.sin(_as_f64(x)),
    der=lambda x: np.cos(_as_f64(x)),
    compile_to_py=lambda s: f"math.sin({s})",
)

# ---------- ARCTAN ----------
ARCTAN = ActivationFunction(
    output=lambda x: np.arctan(_as_f64(x)),
    der=lambda x: 1.0 / (1.0 + _as_f64(x) ** 2),
    compile_to_py=lambda s: f"math.atan({s})",
)

# ---------- LINEAR ----------
LINEAR = ActivationFunction(
    output=lambda x: _as_f64(x),
    der=lambda x: np.ones_like(_as_f64(x)),
    compile_to_py=lambda s: s,
)


ACTIVATIONS: dict[str, ActivationFunction] = {
    "tanh": TANH,
    "relu": RELU,
    "leakyrelu": LEAKY_RELU,
    "elu": ELU,
    "selu": SELU,
    "sigmoid": SIGMOID,
    "swish": SWISH,
    "gelu": GELU,
    "softplus": SOFTPLUS,
    "mish": MISH,
    "sine": SINE,
    "arctan": ARCTAN,
    "linear": LINEAR,
}