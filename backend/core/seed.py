"""Seed helpers for online data generation.

We don't attempt to replicate the JS `seedrandom` algorithm bit-for-bit.
Instead:
    hash_seed(base, k) -> 64-bit int derived deterministically from (base, k)
    make_rng(base, k)  -> np.random.Generator seeded by that int

So (base_seed, epoch_counter) fully determines the training data of that
epoch, which is what "online generation" needs for reproducibility.
"""
from __future__ import annotations

import hashlib

import numpy as np


def hash_seed(base_seed: str, counter: int = 0) -> int:
    payload = f"{base_seed}::{counter}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def make_rng(base_seed: str, counter: int = 0) -> np.random.Generator:
    return np.random.default_rng(hash_seed(base_seed, counter))