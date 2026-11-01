"""Error (loss) functions. Vectorized over a batch."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ErrorFunction:
    error: Callable[[np.ndarray, np.ndarray], np.ndarray]
    der: Callable[[np.ndarray, np.ndarray], np.ndarray]


SQUARE = ErrorFunction(
    error=lambda out, target: 0.5 * (out - target) ** 2,
    der=lambda out, target: out - target,
)


# ---------------------------------------------------------------------------
# Cross-entropy (softmax + CE). `out` is the softmax probability matrix
# (N, K) and `target` is a one-hot matrix (N, K).
#
# NOTE: the *fused* softmax+CE gradient (p - onehot) wrt logits is applied
# directly inside Network.backward, so `der` here is never called by the
# training loop. It is implemented for interface completeness only.
# ---------------------------------------------------------------------------

def _ce_error(out: np.ndarray, target: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -np.sum(target * np.log(out + eps), axis=-1)


def _ce_der(out: np.ndarray, target: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -target / (out + eps)


CROSS_ENTROPY = ErrorFunction(error=_ce_error, der=_ce_der)