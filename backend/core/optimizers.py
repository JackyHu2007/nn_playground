"""Optimizers. All share the same interface:

    step(params, grads, active_mask, key) -> new_params

`active_mask` has the same shape as `params`; entries that are False are
LEFT UNCHANGED and their internal state is decayed with masked grads. This
lets L1-dead weights be frozen cleanly.
"""
from __future__ import annotations

from typing import Hashable

import numpy as np


class Optimizer:
    name: str = "base"

    def step(
        self,
        params: np.ndarray,
        grads: np.ndarray,
        active_mask: np.ndarray,
        key: Hashable,
    ) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def reset(self) -> None:
        pass


class SGD(Optimizer):
    name = "SGD"

    def __init__(self, lr: float):
        self.lr = float(lr)

    def step(self, params, grads, active_mask, key):
        return np.where(active_mask, params - self.lr * grads, params)


class Momentum(Optimizer):
    name = "Momentum"

    def __init__(self, lr: float, mu: float = 0.9):
        self.lr = float(lr)
        self.mu = float(mu)
        self.velocity: dict[Hashable, np.ndarray] = {}

    def step(self, params, grads, active_mask, key):
        g = np.where(active_mask, grads, 0.0)
        v = self.velocity.get(key)
        if v is None:
            v = np.zeros_like(params)
        v = self.mu * v + g
        self.velocity[key] = v
        return np.where(active_mask, params - self.lr * v, params)

    def reset(self):
        self.velocity.clear()


class RMSProp(Optimizer):
    name = "RMSProp"

    def __init__(self, lr: float, rho: float = 0.9, eps: float = 1e-8):
        self.lr = float(lr)
        self.rho = float(rho)
        self.eps = float(eps)
        self.sq_avg: dict[Hashable, np.ndarray] = {}

    def step(self, params, grads, active_mask, key):
        g = np.where(active_mask, grads, 0.0)
        s = self.sq_avg.get(key)
        if s is None:
            s = np.zeros_like(params)
        s = self.rho * s + (1.0 - self.rho) * (g * g)
        self.sq_avg[key] = s
        update = self.lr * g / (np.sqrt(s) + self.eps)
        return np.where(active_mask, params - update, params)

    def reset(self):
        self.sq_avg.clear()


class Adam(Optimizer):
    name = "Adam"

    def __init__(
        self,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m: dict[Hashable, np.ndarray] = {}
        self.v: dict[Hashable, np.ndarray] = {}
        self.t: dict[Hashable, int] = {}

    def step(self, params, grads, active_mask, key):
        g = np.where(active_mask, grads, 0.0)
        m = self.m.get(key)
        v = self.v.get(key)
        if m is None:
            m = np.zeros_like(params)
            v = np.zeros_like(params)
        m = self.beta1 * m + (1.0 - self.beta1) * g
        v = self.beta2 * v + (1.0 - self.beta2) * (g * g)
        t = self.t.get(key, 0) + 1
        self.m[key] = m
        self.v[key] = v
        self.t[key] = t
        m_hat = m / (1.0 - self.beta1 ** t)
        v_hat = v / (1.0 - self.beta2 ** t)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return np.where(active_mask, params - update, params)

    def reset(self):
        self.m.clear()
        self.v.clear()
        self.t.clear()


def make_optimizer(name: str, lr: float) -> Optimizer:
    name = (name or "SGD").lower()
    if name == "sgd":
        return SGD(lr)
    if name == "momentum":
        return Momentum(lr)
    if name == "rmsprop":
        return RMSProp(lr)
    if name == "adam":
        return Adam(lr)
    raise ValueError(f"Unknown optimizer: {name!r}")