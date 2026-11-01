"""Shared types for the data layer."""
from typing import Callable, NamedTuple

import numpy as np


class Example2D(NamedTuple):
    """A single 2D example: (x, y) coordinates with a scalar label."""
    x: float
    y: float
    label: float


# A data generator takes (num_samples, noise, rng) and returns points.
# `noise` is in [0, 0.5] (matching the original `state.noise / 100`).
DataGenerator = Callable[[int, float, np.random.Generator], list[Example2D]]