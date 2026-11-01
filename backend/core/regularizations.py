"""L1 / L2 regularization with vectorized output/derivative."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class RegularizationFunction:
    output: Callable[[np.ndarray], np.ndarray]
    der: Callable[[np.ndarray], np.ndarray]


L1 = RegularizationFunction(
    output=lambda w: np.abs(w),
    # subgradient at 0 is taken as 0 (same as original)
    der=lambda w: np.where(w > 0, 1.0, np.where(w < 0, -1.0, 0.0)),
)

L2 = RegularizationFunction(
    output=lambda w: 0.5 * w * w,
    der=lambda w: w.copy(),
)

REGULARIZATIONS: dict[str, RegularizationFunction | None] = {
    "none": None,
    "L1": L1,
    "L2": L2,
}