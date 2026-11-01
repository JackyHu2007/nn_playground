"""Loaders for the four real-world JSON datasets.

The datasets are loaded once at import time. Train and test splits are
**guaranteed non-overlapping** by construction: for classification, each
class is divided into a front half (train) and a back half (test); for
regression, the whole list is similarly divided.

This fixes a bug in the original TS, where poroperm's train and test
were identical (both used `slice(-n)`), and porosity's were identical
too (only 85 records but `n=85`), which made "test loss" meaningless.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import Example2D


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load_json(name: str) -> list[Example2D]:
    with (DATA_DIR / name).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Example2D(d["x"], d["y"], d["label"]) for d in raw]


_SAND_SHALE: list[Example2D] = _load_json("sand-shale.json")
_PORO_PERM: list[Example2D] = _load_json("poro-perm-sands.json")
_POROSITY: list[Example2D] = _load_json("porosity.json")
_DTS: list[Example2D] = _load_json("dts.json")


def _split_by_label(data: list[Example2D]):
    return (
        [p for p in data if p.label == 1.0],
        [p for p in data if p.label == -1.0],
    )


def _classification_split(data: list[Example2D], num_samples: int,
                          which: str) -> list[Example2D]:
    """Return a non-overlapping train or test slice.

    Per class, the front half goes to train and the back half to test.
    `n_per_class = min(num_samples // 2, len(class) // 2)` so train and
    test can never share a point.
    """
    if which not in ("train", "test"):
        raise ValueError(f"which must be 'train' or 'test', got {which!r}")
    pos, neg = _split_by_label(data)
    n = num_samples // 2
    n_pos = min(n, len(pos) // 2)
    n_neg = min(n, len(neg) // 2)
    if which == "train":
        return pos[:n_pos] + neg[:n_neg]
    test_pos = pos[-n_pos:] if n_pos > 0 else []
    test_neg = neg[-n_neg:] if n_neg > 0 else []
    return test_pos + test_neg


def _regression_split(data: list[Example2D], n_formula: int,
                      which: str) -> list[Example2D]:
    """Return a non-overlapping train or test slice of a flat list."""
    if which not in ("train", "test"):
        raise ValueError(f"which must be 'train' or 'test', got {which!r}")
    n = min(n_formula, len(data) // 2)
    if n <= 0:
        return []
    return data[:n] if which == "train" else data[-n:]


# ---------------------------------------------------------------------------
# classification (JSON)
# ---------------------------------------------------------------------------

def classify_rocks_train_data(num_samples: int, noise: float,
                              rng: np.random.Generator) -> list[Example2D]:
    return _classification_split(_SAND_SHALE, num_samples, "train")


def classify_rocks_test_data(num_samples: int, noise: float,
                             rng: np.random.Generator) -> list[Example2D]:
    return _classification_split(_SAND_SHALE, num_samples, "test")


def classify_poroperm_train_data(num_samples: int, noise: float,
                                 rng: np.random.Generator) -> list[Example2D]:
    return _classification_split(_PORO_PERM, num_samples, "train")


def classify_poroperm_test_data(num_samples: int, noise: float,
                                rng: np.random.Generator) -> list[Example2D]:
    return _classification_split(_PORO_PERM, num_samples, "test")


# ---------------------------------------------------------------------------
# regression (JSON)
# ---------------------------------------------------------------------------

def regress_porosity_train_data(num_samples: int, noise: float,
                                rng: np.random.Generator) -> list[Example2D]:
    n_formula = int(85 * num_samples / 400)
    return _regression_split(_POROSITY, n_formula, "train")


def regress_porosity_test_data(num_samples: int, noise: float,
                               rng: np.random.Generator) -> list[Example2D]:
    n_formula = int(85 * num_samples / 400)
    return _regression_split(_POROSITY, n_formula, "test")


def regress_dts_train_data(num_samples: int, noise: float,
                           rng: np.random.Generator) -> list[Example2D]:
    n_formula = num_samples // 2
    return _regression_split(_DTS, n_formula, "train")


def regress_dts_test_data(num_samples: int, noise: float,
                          rng: np.random.Generator) -> list[Example2D]:
    n_formula = num_samples // 2
    return _regression_split(_DTS, n_formula, "test")