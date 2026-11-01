"""Synthetic data generators.

All functions match the original `dataset.ts` math, but accept an explicit
`np.random.Generator` (needed for online data regeneration) instead of
relying on global `Math.random()` + `seedrandom`.
"""
from __future__ import annotations

import math

import numpy as np

from .types import Example2D


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _normal_random(mean: float, variance: float, rng: np.random.Generator) -> float:
    """Marsaglia polar method, same as the original TS implementation."""
    while True:
        v1 = 2.0 * rng.random() - 1.0
        v2 = 2.0 * rng.random() - 1.0
        s = v1 * v1 + v2 * v2
        if 0.0 < s <= 1.0:
            break
    return mean + math.sqrt(variance) * math.sqrt(-2.0 * math.log(s) / s) * v1


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def classify_linear_data(num_samples: int, noise: float,
                         rng: np.random.Generator) -> list[Example2D]:
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-5.0, 5.0)
        padding = 0.3
        x += padding if x > 0 else -padding
        y = rng.uniform(-5.0, 5.0)
        noise_x = rng.uniform(-5.0, 5.0) * noise
        noise_y = rng.uniform(-5.0, 5.0) * noise
        label = 1.0 if (x + noise_x) >= 0.0 else -1.0
        points.append(Example2D(x, y, label))
    return points


def classify_xor_data(num_samples: int, noise: float,
                      rng: np.random.Generator) -> list[Example2D]:
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-5.0, 5.0)
        padding = 0.3
        x += padding if x > 0 else -padding
        y = rng.uniform(-5.0, 5.0)
        y += padding if y > 0 else -padding
        noise_x = rng.uniform(-5.0, 5.0) * noise
        noise_y = rng.uniform(-5.0, 5.0) * noise
        label = 1.0 if (x + noise_x) * (y + noise_y) >= 0.0 else -1.0
        points.append(Example2D(x, y, label))
    return points


def classify_gauss_data(num_samples: int, noise: float,
                        rng: np.random.Generator) -> list[Example2D]:
    variance = 0.5 + (noise / 0.5) * (4.0 - 0.5)
    n = num_samples // 2
    points: list[Example2D] = []
    for _ in range(n):
        points.append(Example2D(
            _normal_random(2.0, variance, rng),
            _normal_random(2.0, variance, rng),
            1.0,
        ))
    for _ in range(n):
        points.append(Example2D(
            _normal_random(-2.0, variance, rng),
            _normal_random(-2.0, variance, rng),
            -1.0,
        ))
    return points


def classify_circle_data(num_samples: int, noise: float,
                         rng: np.random.Generator) -> list[Example2D]:
    radius = 5.0
    points: list[Example2D] = []
    n = num_samples // 2

    def label_for(x: float, y: float) -> float:
        d = math.hypot(x, y)
        return 1.0 if d < radius * 0.5 else -1.0

    for _ in range(n):
        r = rng.uniform(0.0, radius * 0.5)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        nx = rng.uniform(-radius, radius) * noise
        ny = rng.uniform(-radius, radius) * noise
        points.append(Example2D(x, y, label_for(x + nx, y + ny)))

    for _ in range(n):
        r = rng.uniform(radius * 0.7, radius)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        nx = rng.uniform(-radius, radius) * noise
        ny = rng.uniform(-radius, radius) * noise
        points.append(Example2D(x, y, label_for(x + nx, y + ny)))

    return points


def classify_spiral_data(num_samples: int, noise: float,
                         rng: np.random.Generator) -> list[Example2D]:
    points: list[Example2D] = []
    n = num_samples // 2

    def gen_spiral(delta_t: float, label: float) -> None:
        for _ in range(n):
            j = rng.uniform(0.0, float(n))
            r = j / n * 5.0
            t = 1.75 * j / n * 2.0 * math.pi + delta_t
            x = r * math.sin(t) + rng.uniform(-1.0, 1.0) * noise
            y = r * math.cos(t) + rng.uniform(-1.0, 1.0) * noise
            points.append(Example2D(x, y, label))

    gen_spiral(0.0, 1.0)
    gen_spiral(math.pi, -1.0)
    return points


def _make_moons(type_: str, n_samples: int, noise: float,
                rng: np.random.Generator) -> list[tuple[float, float]]:
    K = 2
    A = 7 * K

    def parabola(a: float, h: float, k: float, x: float) -> float:
        return a * (x - h) ** 2 + k

    if type_ == "pos":
        fn = lambda x: parabola(-A, 1.0 / 3.0, K - 1.0, x)  # noqa: E731
        x_range = (0.0, 2.0 / 3.0)
    else:
        fn = lambda x: parabola(A, 2.0 / 3.0, -1.0, x)  # noqa: E731
        x_range = (1.0 / 3.0, 1.0)

    dots: list[tuple[float, float]] = []
    for _ in range(n_samples):
        x = rng.uniform(x_range[0], x_range[1])
        radius = rng.uniform(-noise / 2.0, noise / 2.0)
        theta = rng.uniform(-math.pi, math.pi)
        dots.append((x + math.sin(theta) * radius,
                     fn(x) + math.cos(theta) * radius))
    return dots


def classify_moons_data(num_samples: int, noise: float,
                        rng: np.random.Generator) -> list[Example2D]:
    n = num_samples // 2
    noise = noise * 0.9 + 0.1
    pos = _make_moons("pos", n, noise, rng)
    neg = _make_moons("neg", n, noise, rng)
    points: list[Example2D] = []
    for x, y in pos:
        points.append(Example2D(x * 8.0 - 4.0, y * 4.5, 1.0))
    for x, y in neg:
        points.append(Example2D(x * 8.0 - 4.0, y * 4.5, -1.0))
    return points


def classify_diagonal_train_data(num_samples: int, noise: float,
                                 rng: np.random.Generator) -> list[Example2D]:
    def get_diagonal_label(x: float, y: float) -> float:
        return 1.0 if y <= -x else -1.0

    num_noise = int(num_samples * noise)
    num_neg = (num_samples - num_noise) // 2
    num_pos = num_neg

    points: list[Example2D] = []

    padding = 0.5
    for _ in range(num_neg):
        x = rng.uniform(-5.0, 5.0)
        y = rng.uniform(-x, 5.0)
        x += padding if y > -x else -padding
        points.append(Example2D(x, y, -1.0))

    padding = 2.0
    for _ in range(num_pos):
        x = rng.uniform(-5.0, 2.0)
        y = rng.uniform(-5.0, min(-x, 2.0))
        x += padding if y > -x else -padding
        points.append(Example2D(x, y, 1.0))

    for _ in range((num_noise + 1) // 2):
        x = rng.uniform(-5.0, 5.0)
        y = rng.uniform(-5.0, 5.0)
        nx = rng.uniform(-5.0, 5.0)
        ny = rng.uniform(-5.0, 5.0)
        points.append(Example2D(x, y, get_diagonal_label(x + nx, y + ny)))

    return points


def classify_diagonal_test_data(num_samples: int, noise: float,
                                rng: np.random.Generator) -> list[Example2D]:
    def get_diagonal_label(x: float, y: float) -> float:
        return 1.0 if y <= -x else -1.0

    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-5.0, 5.0)
        y = rng.uniform(-5.0, 5.0)
        nx = rng.uniform(-5.0, 5.0) * noise
        ny = rng.uniform(-5.0, 5.0) * noise
        points.append(Example2D(x, y, get_diagonal_label(x + nx, y + ny)))
    return points


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------

def regress_plane(num_samples: int, noise: float,
                  rng: np.random.Generator) -> list[Example2D]:
    radius = 6.0

    def label_for(x: float, y: float) -> float:
        return max(-1.0, min(1.0, (x + y) / 10.0))

    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-radius, radius)
        y = rng.uniform(-radius, radius)
        nx = rng.uniform(-radius, radius) * noise
        ny = rng.uniform(-radius, radius) * noise
        points.append(Example2D(x, y, label_for(x + nx, y + ny)))
    return points


_GAUSSIANS = [
    (-4.0, 2.5, 1.0), (0.0, 2.5, -1.0), (4.0, 2.5, 1.0),
    (-4.0, -2.5, -1.0), (0.0, -2.5, 1.0), (4.0, -2.5, -1.0),
]


def regress_gaussian(num_samples: int, noise: float,
                     rng: np.random.Generator) -> list[Example2D]:
    radius = 6.0

    def label_scale(d: float) -> float:
        return max(0.0, min(1.0, 1.0 - d / 2.0))

    def get_label(x: float, y: float) -> float:
        best = 0.0
        for cx, cy, sign in _GAUSSIANS:
            d = math.hypot(x - cx, y - cy)
            candidate = sign * label_scale(d)
            if abs(candidate) > abs(best):
                best = candidate
        return best

    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-radius, radius)
        y = rng.uniform(-radius, radius)
        nx = rng.uniform(-radius, radius) * noise
        ny = rng.uniform(-radius, radius) * noise
        points.append(Example2D(x, y, get_label(x + nx, y + ny)))
    return points


def classify_checkerboard_data(num_samples: int, noise: float,
                               rng: np.random.Generator) -> list[Example2D]:
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        grid_x = math.floor((x + nx) / 3.0)
        grid_y = math.floor((y + ny) / 3.0)
        label = 1.0 if (grid_x + grid_y) % 2 == 0 else -1.0
        points.append(Example2D(x, y, label))
    return points


def classify_rings_data(num_samples: int, noise: float,
                        rng: np.random.Generator) -> list[Example2D]:
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        r = math.hypot(x + nx, y + ny)
        band = math.floor(r / 1.5)
        label = 1.0 if band % 2 == 0 else -1.0
        points.append(Example2D(x, y, label))
    return points


def classify_sine_data(num_samples: int, noise: float,
                       rng: np.random.Generator) -> list[Example2D]:
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        threshold = 3.0 * math.sin(x + nx)
        label = 1.0 if (y + ny) > threshold else -1.0
        points.append(Example2D(x, y, label))
    return points


def regress_saddle(num_samples: int, noise: float,
                   rng: np.random.Generator) -> list[Example2D]:
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        z = ((x + nx) ** 2 - (y + ny) ** 2) / 36.0
        z = max(-1.0, min(1.0, z))
        points.append(Example2D(x, y, z))
    return points


def regress_ripples(num_samples: int, noise: float,
                    rng: np.random.Generator) -> list[Example2D]:
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        r = math.hypot(x + nx, y + ny)
        z = math.sin(r * 2.0)
        points.append(Example2D(x, y, z))
    return points


def regress_sine_surface(num_samples: int, noise: float,
                         rng: np.random.Generator) -> list[Example2D]:
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        vx = (x + nx) * math.pi / 3.0
        vy = (y + ny) * math.pi / 3.0
        z = math.sin(vx) * math.cos(vy)
        points.append(Example2D(x, y, z))
    return points


def classify_rose_data(num_samples: int, noise: float,
                       rng: np.random.Generator) -> list[Example2D]:
    """3-petal rose pattern: r = 4 * |cos(3*theta)|"""
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        px, py = x + nx, y + ny
        r = math.hypot(px, py)
        theta = math.atan2(py, px)
        target_r = 4.0 * abs(math.cos(3.0 * theta))

        label = 1.0 if r < target_r else -1.0
        points.append(Example2D(x, y, label))
    return points


def classify_parabola_data(num_samples: int, noise: float,
                           rng: np.random.Generator) -> list[Example2D]:
    """Parabola: y = x^2 / 4 - 3"""
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        # 阈值函数
        threshold = ((x + nx) ** 2) / 4.0 - 3.0
        label = 1.0 if (y + ny) > threshold else -1.0
        points.append(Example2D(x, y, label))
    return points


def regress_mexican_hat(num_samples: int, noise: float,
                        rng: np.random.Generator) -> list[Example2D]:
    """Ricker wavelet: z = sin(r)/r"""
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        r = math.hypot(x + nx, y + ny)
        # 限制峰值到 [-1, 1] 范围
        z = (math.sin(r * 1.5) / (r * 1.5)) if r > 1e-5 else 1.0
        points.append(Example2D(x, y, z))
    return points


def regress_egg_carton(num_samples: int, noise: float,
                       rng: np.random.Generator) -> list[Example2D]:
    """Egg carton curve: z = sin(x) * cos(y)"""
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        vx = (x + nx) * math.pi / 3.5
        vy = (y + ny) * math.pi / 3.5
        z = math.sin(vx) * math.cos(vy)
        points.append(Example2D(x, y, z))
    return points


def classify_mandala_data(num_samples: int, noise: float,
                          rng: np.random.Generator) -> list[Example2D]:
    """Mandala: r = 3 + 2*sin(8*theta)*cos(6*theta)"""
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        px, py = x + nx, y + ny
        r = math.hypot(px, py)
        theta = math.atan2(py, px)

        # Overlay two trigonometric functions of different frequencies to create a complex starry border
        target_r = 3.0 + 2.0 * math.sin(8.0 * theta) * math.cos(6.0 * theta)
        label = 1.0 if r < target_r else -1.0
        points.append(Example2D(x, y, label))
    return points


def classify_dense_spiral_data(num_samples: int, noise: float,
                               rng: np.random.Generator) -> list[Example2D]:
    """Compact double helix: twists 3.5 times"""
    points = []
    n = num_samples // 2

    def gen_spiral(delta_t: float, label: float) -> None:
        for _ in range(n):
            j = rng.uniform(0.0, float(n))
            r = j / n * 5.5
            # The number of laps increased from 1.75 to 3.5
            t = 3.5 * j / n * 2.0 * math.pi + delta_t
            x = r * math.sin(t) + rng.uniform(-1.0, 1.0) * noise
            y = r * math.cos(t) + rng.uniform(-1.0, 1.0) * noise
            points.append(Example2D(x, y, label))

    gen_spiral(0.0, 1.0)
    gen_spiral(math.pi, -1.0)
    return points


def classify_hf_checkerboard_data(num_samples: int, noise: float,
                                  rng: np.random.Generator) -> list[Example2D]:
    """High-frequency checkerboard: grid size reduced to 1.2"""
    points = []
    grid_size = 1.2
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        grid_x = math.floor((x + nx) / grid_size)
        grid_y = math.floor((y + ny) / grid_size)
        label = 1.0 if (grid_x + grid_y) % 2 == 0 else -1.0
        points.append(Example2D(x, y, label))
    return points


def classify_star_data(num_samples: int, noise: float,
                       rng: np.random.Generator) -> list[Example2D]:
    """Pentagram: Sharp corners"""
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        px, py = x + nx, y + ny
        r = math.hypot(px, py)
        theta = math.atan2(py, px)

        # Star-shaped polar coordinate equation
        a = math.pi / 5.0
        theta_mod = theta % (2 * a)
        target_r = 4.0 / math.cos(theta_mod - a)

        if r < target_r:
            label = 1.0
        else:
            label = -1.0
        points.append(Example2D(x, y, label))
    return points


def regress_rastrigin(num_samples: int, noise: float,
                      rng: np.random.Generator) -> list[Example2D]:
    """Rastrigin function: contains countless local minima"""
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        px = (x + nx) * 0.8
        py = (y + ny) * 0.8
        # Standard Rastrigin (n=2)
        z = 20.0 + (px ** 2 - 10.0 * math.cos(2.0 * math.pi * px)) + \
            (py ** 2 - 10.0 * math.cos(2.0 * math.pi * py))

        # Mapped to [-1, 1] (range is roughly 0 to 40)
        z = (z - 20.0) / 20.0
        z = max(-1.0, min(1.0, z))
        points.append(Example2D(x, y, z))
    return points


def regress_hf_sine(num_samples: int, noise: float,
                    rng: np.random.Generator) -> list[Example2D]:
    """High-frequency sine: Oscillates 6 times within the input range"""
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        vx = (x + nx) * math.pi * 2.0
        vy = (y + ny) * math.pi * 2.0
        z = math.sin(vx) * math.cos(vy)
        points.append(Example2D(x, y, z))
    return points


def regress_spiral_ramp(num_samples: int, noise: float,
                        rng: np.random.Generator) -> list[Example2D]:
    """Spiral slope: height twists continuously with the polar angle"""
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        px, py = x + nx, y + ny
        r = math.hypot(px, py)
        theta = math.atan2(py, px)

        # A surface twisted like a spiral staircase
        z = math.sin(2.0 * theta + r * 1.5)
        z = max(-1.0, min(1.0, z))
        points.append(Example2D(x, y, z))
    return points


def regress_ackley(num_samples: int, noise: float,
                   rng: np.random.Generator) -> list[Example2D]:
    """Ackley function: there's a super deep pit in the center, surrounded by flat valleys"""
    points = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise

        px = (x + nx) * 0.8
        py = (y + ny) * 0.8

        term1 = -20.0 * math.exp(-0.2 * math.sqrt(0.5 * (px ** 2 + py ** 2)))
        term2 = -math.exp(0.5 * (math.cos(2.0 * math.pi * px) + math.cos(2.0 * math.pi * py)))
        z = term1 + term2 + math.e + 20.0

        z = (z - 12.5) / 12.5
        z = max(-1.0, min(1.0, z))
        points.append(Example2D(x, y, z))
    return points


# ---------------------------------------------------------------------------
# Multi-class 2D datasets (problem="multiclass"). Labels are float(k) in
# [0, K). The training loop turns them into integer indices for softmax+CE.
# ---------------------------------------------------------------------------

def multi_blobs_data(num_samples: int, noise: float,
                     rng: np.random.Generator) -> list[Example2D]:
    """Three Gaussian clusters, evenly distributed around a circle with a radius of 4."""
    K = 3
    centers = [
        (4.0 * math.cos(2.0 * math.pi * k / K),
         4.0 * math.sin(2.0 * math.pi * k / K))
        for k in range(K)
    ]
    points: list[Example2D] = []
    for _ in range(num_samples):
        k0 = int(rng.integers(0, K))
        cx, cy = centers[k0]
        x = cx + rng.normal(0.0, 1.2)
        y = cy + rng.normal(0.0, 1.2)
        x = max(-6.0, min(6.0, x))
        y = max(-6.0, min(6.0, y))
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        d = [math.hypot(px - c[0], py - c[1]) for c in centers]
        k = int(np.argmin(d))
        points.append(Example2D(x, y, float(k)))
    return points


def multi_spirals_data(num_samples: int, noise: float,
                       rng: np.random.Generator) -> list[Example2D]:
    """3-arm spiral, 1.75 turns per arm."""
    K = 3
    n_per = max(1, num_samples // K)
    points: list[Example2D] = []
    for k in range(K):
        delta = 2.0 * math.pi * k / K
        for _ in range(n_per):
            j = rng.uniform(0.0, float(n_per))
            r = j / n_per * 5.0
            t = 1.75 * j / n_per * 2.0 * math.pi + delta
            x = r * math.sin(t) + rng.uniform(-1.0, 1.0) * noise
            y = r * math.cos(t) + rng.uniform(-1.0, 1.0) * noise
            points.append(Example2D(x, y, float(k)))
    return points


def multi_concentric_data(num_samples: int, noise: float,
                          rng: np.random.Generator) -> list[Example2D]:
    """3 concentric rings (radii 1.5 / 3.0 / 4.5), labeled from the nearest ring."""
    K = 3
    radii = [1.5, 3.0, 4.5]
    points: list[Example2D] = []
    for _ in range(num_samples):
        band = int(rng.integers(0, K))
        r = radii[band] + rng.normal(0.0, 0.35)
        r = max(0.05, r)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        x = max(-6.0, min(6.0, r * math.cos(angle)))
        y = max(-6.0, min(6.0, r * math.sin(angle)))
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        rr = math.hypot(x + nx, y + ny)
        k = int(np.argmin([abs(rr - radii[j]) for j in range(K)]))
        points.append(Example2D(x, y, float(k)))
    return points


# ---------------------------------------------------------------------------
# Multiclass generators: harder tiers
# All labels are float(k) in [0, K). Training loop converts them to int for
# softmax+CE.
# ---------------------------------------------------------------------------

def _multi_blobs_generic(num_samples, noise, rng, K, radius, spread=1.2):
    """K Gaussian blobs on a circle of given radius."""
    centers = [
        (radius * math.cos(2.0 * math.pi * k / K),
         radius * math.sin(2.0 * math.pi * k / K))
        for k in range(K)
    ]
    points: list[Example2D] = []
    for _ in range(num_samples):
        k0 = int(rng.integers(0, K))
        cx, cy = centers[k0]
        x = cx + rng.normal(0.0, spread)
        y = cy + rng.normal(0.0, spread)
        x = max(-6.0, min(6.0, x))
        y = max(-6.0, min(6.0, y))
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        k = int(np.argmin([math.hypot(px - c[0], py - c[1]) for c in centers]))
        points.append(Example2D(x, y, float(k)))
    return points


def _multi_spirals_generic(num_samples, noise, rng, K, turns, radius):
    points: list[Example2D] = []
    n_per = max(1, num_samples // K)
    for k in range(K):
        delta = 2.0 * math.pi * k / K
        for _ in range(n_per):
            j = rng.uniform(0.0, float(n_per))
            r = j / n_per * radius
            t = turns * j / n_per * 2.0 * math.pi + delta
            x = r * math.sin(t) + rng.uniform(-1.0, 1.0) * noise
            y = r * math.cos(t) + rng.uniform(-1.0, 1.0) * noise
            points.append(Example2D(x, y, float(k)))
    return points


def multi_blobs_4_data(num_samples, noise, rng):
    return _multi_blobs_generic(num_samples, noise, rng, K=4, radius=4.0, spread=1.1)


def multi_blobs_5_data(num_samples, noise, rng):
    return _multi_blobs_generic(num_samples, noise, rng, K=5, radius=4.2, spread=1.0)


def multi_blobs_8_data(num_samples, noise, rng):
    # spread is smaller so blobs stay distinct but crowded -> harder
    return _multi_blobs_generic(num_samples, noise, rng, K=8, radius=4.5, spread=0.9)


def multi_spirals_4_data(num_samples, noise, rng):
    return _multi_spirals_generic(num_samples, noise, rng, K=4, turns=2.0, radius=5.0)


def multi_spirals_5_data(num_samples, noise, rng):
    return _multi_spirals_generic(num_samples, noise, rng, K=5, turns=2.5, radius=5.2)


def multi_spirals_7_data(num_samples, noise, rng):
    return _multi_spirals_generic(num_samples, noise, rng, K=7, turns=2.8, radius=5.4)


def multi_concentric_5_data(num_samples, noise, rng):
    K = 5
    radii = [1.0, 2.2, 3.4, 4.6, 5.8]
    points: list[Example2D] = []
    for _ in range(num_samples):
        band = int(rng.integers(0, K))
        r = radii[band] + rng.normal(0.0, 0.28)
        r = max(0.05, r)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        x = max(-6.0, min(6.0, r * math.cos(angle)))
        y = max(-6.0, min(6.0, r * math.sin(angle)))
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        rr = math.hypot(x + nx, y + ny)
        k = int(np.argmin([abs(rr - radii[j]) for j in range(K)]))
        points.append(Example2D(x, y, float(k)))
    return points


def multi_moons_4_data(num_samples, noise, rng):
    """4 crescents, alternating up/down and shifted horizontally."""
    K = 4
    points: list[Example2D] = []
    n_per = max(1, num_samples // K)
    noise_eff = noise * 0.9 + 0.1
    for k in range(K):
        up = (k % 2 == 0)
        cx = -4.0 + 3.0 * k
        cy = -1.5 if up else 1.5
        for _ in range(n_per):
            t = rng.uniform(0.0, math.pi)
            px = 2.2 * math.cos(t) + cx
            py = (2.2 * math.sin(t) * (1 if up else -1)) + cy
            px += rng.uniform(-0.4, 0.4) * noise_eff
            py += rng.uniform(-0.4, 0.4) * noise_eff
            points.append(Example2D(px, py, float(k)))
    return points


def multi_rings_4_data(num_samples, noise, rng):
    """4 colored rings, classified by 'band' index (r // 1.5)."""
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        r = math.hypot(x + nx, y + ny)
        band = min(3, int(r // 1.5))
        points.append(Example2D(x, y, float(band)))
    return points


def _multi_checkerboard(num_samples, noise, rng, nx, ny, cell):
    """nx × ny colored checkerboard with K = nx*ny classes."""
    K = nx * ny
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        gx = math.floor((x + rng.uniform(-6.0, 6.0) * noise) / cell) % nx
        gy = math.floor((y + rng.uniform(-6.0, 6.0) * noise) / cell) % ny
        k = (gx + gy * nx) % K
        points.append(Example2D(x, y, float(k)))
    return points


def multi_checkerboard_4_data(num_samples, noise, rng):
    return _multi_checkerboard(num_samples, noise, rng, nx=2, ny=2, cell=3.0)


def multi_checkerboard_6_data(num_samples, noise, rng):
    return _multi_checkerboard(num_samples, noise, rng, nx=3, ny=2, cell=2.5)


def multi_hf_checkerboard_10_data(num_samples, noise, rng):
    # 5 x 2 tiny cells: extremely high frequency
    return _multi_checkerboard(num_samples, noise, rng, nx=5, ny=2, cell=1.4)


def _multi_mandala(num_samples, noise, rng, K):
    """K angular sectors, each with its own radial threshold r_k(theta)."""
    points: list[Example2D] = []
    sector = 2.0 * math.pi / K
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        r = math.hypot(px, py)
        theta = math.atan2(py, px)
        s = int(((theta + math.pi) % (2 * math.pi)) // sector)
        s = max(0, min(K - 1, s))
        # per-sector threshold + high-frequency ripple so boundaries are hard
        thresh = 3.0 + 1.5 * math.cos(s * 1.3) + \
                 0.8 * math.sin(8.0 * theta) * math.cos(6.0 * theta)
        if r < thresh:
            k = s
        else:
            k = (s + 1) % K
        points.append(Example2D(x, y, float(k)))
    return points


def multi_mandala_6_data(num_samples, noise, rng):
    return _multi_mandala(num_samples, noise, rng, K=6)


def multi_mandala_10_data(num_samples, noise, rng):
    return _multi_mandala(num_samples, noise, rng, K=10)


def _multi_rose(num_samples, noise, rng, K):
    """Rose curve with K petals; petal index -> class."""
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        r = math.hypot(px, py)
        theta = math.atan2(py, px)
        target_r = 4.5 * abs(math.cos(K * theta / 2.0))
        if r < target_r:
            # inside petal: which petal
            petal = int(((theta + math.pi) / (2.0 * math.pi / K))) % K
            k = petal
        else:
            k = (int(((theta + math.pi) / (2.0 * math.pi / K))) + K // 2) % K
        points.append(Example2D(x, y, float(k)))
    return points


def multi_rose_6_data(num_samples, noise, rng):
    return _multi_rose(num_samples, noise, rng, K=6)


def _multi_dense_spiral(num_samples, noise, rng, K, turns, radius):
    points: list[Example2D] = []
    n_per = max(1, num_samples // K)
    for k in range(K):
        delta = 2.0 * math.pi * k / K
        for _ in range(n_per):
            j = rng.uniform(0.0, float(n_per))
            r = j / n_per * radius
            t = turns * j / n_per * 2.0 * math.pi + delta
            x = r * math.sin(t) + rng.uniform(-1.0, 1.0) * noise
            y = r * math.cos(t) + rng.uniform(-1.0, 1.0) * noise
            points.append(Example2D(x, y, float(k)))
    return points


def multi_hf_spiral_8_data(num_samples, noise, rng):
    return _multi_dense_spiral(num_samples, noise, rng, K=8, turns=3.0, radius=5.5)


def multi_dense_spiral_10_data(num_samples, noise, rng):
    return _multi_dense_spiral(num_samples, noise, rng, K=10, turns=4.0, radius=5.8)

# ===========================================================================
# NEW — visually distinctive multiclass datasets
# All generators follow the same signature: (num_samples, noise, rng).
# Labels are float(k) in [0, K); the training loop converts them to int
# indices for softmax+CE.
# ===========================================================================

# --- Voronoi / cellular mosaic ---------------------------------------------
# Hard-coded seed points so the preview thumbnail is deterministic and looks
# like a designed mosaic, not random noise. `noise` only jitters the query
# point, not the seeds, so the cell layout stays recognizable.

_VORONOI_SEEDS_5 = [
    (-3.5, -2.8), (2.8, -3.2), (4.2, 1.8), (-0.5, 3.8), (-4.5, 1.5),
]
_VORONOI_SEEDS_7 = [
    (-4.5, -3.5), (0.0, -4.0), (4.2, -3.0),
    (-3.0, 0.0), (3.8, 0.5),
    (-1.2, 3.8), (2.5, 3.8),
]
_VORONOI_SEEDS_9 = [
    (-5.0, -3.5), (-2.0, -4.5), (1.0, -4.5), (4.0, -3.5),
    (-4.5, 0.0), (4.5, 0.5),
    (-4.0, 3.5), (-1.0, 4.0), (3.5, 3.5),
]


def _multi_voronoi(num_samples, noise, rng, seeds):
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        best = 0
        best_d = float("inf")
        for j, (sx, sy) in enumerate(seeds):
            d = (px - sx) ** 2 + (py - sy) ** 2
            if d < best_d:
                best_d = d
                best = j
        points.append(Example2D(x, y, float(best)))
    return points


def multi_voronoi_5_data(num_samples, noise, rng):
    return _multi_voronoi(num_samples, noise, rng, _VORONOI_SEEDS_5)


def multi_voronoi_7_data(num_samples, noise, rng):
    return _multi_voronoi(num_samples, noise, rng, _VORONOI_SEEDS_7)


def multi_voronoi_9_data(num_samples, noise, rng):
    return _multi_voronoi(num_samples, noise, rng, _VORONOI_SEEDS_9)


# --- Honeycomb (hexagonal grid) --------------------------------------------
# Pointy-top hex grid, axial coordinates. Coloring with (q - r) % K
# guarantees no two adjacent cells share a color for K in {3, 4, 6}.

def _axial_round(q: float, r: float) -> tuple[int, int]:
    s = -q - r
    rq, rr, rs = round(q), round(r), round(s)
    dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs - s)
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    return int(rq), int(rr)


def _honeycomb_axial(x: float, y: float, size: float) -> tuple[int, int]:
    q = (math.sqrt(3.0) / 3.0 * x - y / 3.0) / size
    r = (2.0 / 3.0 * y) / size
    return _axial_round(q, r)


def _multi_honeycomb(num_samples, noise, rng, K):
    points: list[Example2D] = []
    size = 2.8
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        q, r = _honeycomb_axial(x + nx, y + ny, size)
        label = (q - r) % K
        points.append(Example2D(x, y, float(label)))
    return points


def multi_honeycomb_3_data(num_samples, noise, rng):
    return _multi_honeycomb(num_samples, noise, rng, 3)


def multi_honeycomb_4_data(num_samples, noise, rng):
    return _multi_honeycomb(num_samples, noise, rng, 4)


def multi_honeycomb_6_data(num_samples, noise, rng):
    return _multi_honeycomb(num_samples, noise, rng, 6)


# --- Interference ripples --------------------------------------------------
# K point-sources arranged on a circle; field value = mean cos(freq*d + phase).
# Quantize that continuous field into K stripes. Bands curve into concentric
# interference patterns — looks like water waves.

def _multi_interference(num_samples, noise, rng, K, freq=2.0):
    sources = [
        (4.0 * math.cos(2.0 * math.pi * j / K),
         4.0 * math.sin(2.0 * math.pi * j / K))
        for j in range(K)
    ]
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        val = 0.0
        for j, (sx, sy) in enumerate(sources):
            d = math.hypot(px - sx, py - sy)
            val += math.cos(freq * d + j * 0.7)
        val /= K
        label = int((val + 1.0) * 0.5 * K)
        label = max(0, min(K - 1, label))
        points.append(Example2D(x, y, float(label)))
    return points


def multi_interference_3_data(num_samples, noise, rng):
    return _multi_interference(num_samples, noise, rng, 3)


def multi_interference_5_data(num_samples, noise, rng):
    return _multi_interference(num_samples, noise, rng, 5)


def multi_interference_7_data(num_samples, noise, rng):
    return _multi_interference(num_samples, noise, rng, 7)


# --- Galaxy arms -----------------------------------------------------------
# K = (K-1) logarithmic spiral arms + 1 background class (at the core and
# between arms). Slimmer arms than `multi-spirals` so the plot reads as a
# sparse galaxy disk, not a dense spiral.

def _multi_galaxy(num_samples, noise, rng, K, arm_width=0.7):
    arms = K - 1
    b = 0.5  # spiral tightness
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        r = math.hypot(px, py)
        if r < 0.6:
            label = K - 1
        else:
            theta = math.atan2(py, px)
            label = K - 1
            for j in range(arms):
                theta_c = 2.0 * math.pi * j / arms + b * math.log(r)
                diff = ((theta - theta_c + math.pi) % (2.0 * math.pi)) - math.pi
                width = arm_width / (1.0 + 0.25 * r)
                if abs(diff) < width:
                    label = j
                    break
        points.append(Example2D(x, y, float(label)))
    return points


def multi_galaxy_4_data(num_samples, noise, rng):
    return _multi_galaxy(num_samples, noise, rng, 4)


def multi_galaxy_5_data(num_samples, noise, rng):
    return _multi_galaxy(num_samples, noise, rng, 5)


# --- Sierpinski carpet -----------------------------------------------------
# Base-3 self-similar square. Each 3x3 block loses its center; recurse.
# The center hole at any level maps to class K-1 (background).
# 8^levels distinct sub-blocks get folded onto K classes by (code % K).

def _sierpinski_carpet_code(u: float, v: float, levels: int) -> int:
    code = 0
    for _ in range(levels):
        u = min(0.999999, max(0.0, u))
        v = min(0.999999, max(0.0, v))
        col = int(u * 3)
        row = int(v * 3)
        if col == 1 and row == 1:
            return -1
        idx = row * 3 + col
        if idx > 4:
            idx -= 1  # skip the center cell
        code = code * 8 + idx
        u = u * 3 - col
        v = v * 3 - row
    return code


def _multi_sierpinski(num_samples, noise, rng, K, levels=2):
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        u = (px + 6.0) / 12.0
        v = (py + 6.0) / 12.0
        code = _sierpinski_carpet_code(u, v, levels)
        label = K - 1 if code < 0 else (code % K)
        points.append(Example2D(x, y, float(label)))
    return points


def multi_sierpinski_4_data(num_samples, noise, rng):
    return _multi_sierpinski(num_samples, noise, rng, 4, levels=2)


def multi_sierpinski_8_data(num_samples, noise, rng):
    return _multi_sierpinski(num_samples, noise, rng, 8, levels=2)


# --- Shapes mosaic ---------------------------------------------------------
# K non-overlapping geometric figures at fixed positions. Every pixel gets
# assigned either to the shape it lies in, or (outside) to the nearest shape
# center — so the plane is fully covered by K classes, no background needed.

_SHAPES_LAYOUT = [
    # (cx, cy, size, kind)
    (-3.8, 2.6, 1.8, "circle"),
    (0.0, 2.6, 1.8, "square"),
    (3.8, 2.6, 1.8, "triangle"),
    (-3.8, -2.6, 1.8, "diamond"),
    (0.0, -2.6, 1.8, "star"),
    (3.8, -2.6, 1.8, "hexagon"),
]


def _shape_contains(kind, px, py, cx, cy, size):
    dx, dy = px - cx, py - cy
    if kind == "circle":
        return dx * dx + dy * dy < size * size
    if kind == "square":
        return abs(dx) < size and abs(dy) < size
    if kind == "triangle":
        if dy > size / 2.0:
            return False
        half_base = size * math.sqrt(3.0) / 2.0
        limit = half_base * (dy + size) / (1.5 * size)
        return abs(dx) < limit
    if kind == "diamond":
        return abs(dx) + abs(dy) < size
    if kind == "star":
        r = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        a = math.pi / 5.0
        mod = (theta + math.pi) % (2.0 * a) - a
        target = 1.5 * size / math.cos(mod)
        return r < target
    if kind == "hexagon":
        r = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        a = math.pi / 6.0
        mod = (theta % (2.0 * a)) - a
        return r < size / math.cos(mod)
    return False


def _multi_shapes(num_samples, noise, rng, n_shapes):
    layout = _SHAPES_LAYOUT[:n_shapes]
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        label = 0
        best = float("inf")
        for j, (cx, cy, sz, kind) in enumerate(layout):
            d = math.hypot(px - cx, py - cy)
            if _shape_contains(kind, px, py, cx, cy, sz):
                d -= 100.0  # strong preference for "inside a figure"
            if d < best:
                best = d
                label = j
        points.append(Example2D(x, y, float(label)))
    return points


def multi_shapes_4_data(num_samples, noise, rng):
    return _multi_shapes(num_samples, noise, rng, 4)


def multi_shapes_5_data(num_samples, noise, rng):
    return _multi_shapes(num_samples, noise, rng, 5)


def multi_shapes_6_data(num_samples, noise, rng):
    return _multi_shapes(num_samples, noise, rng, 6)


# --- Nested regular polygons ----------------------------------------------
# K layers, layer j is a regular (3+j)-gon with radius 1.6 + 1.15*j.
# From the origin outward, the first polygon whose radius covers the query
# point determines the class. Inside is class 0, outermost is class K-1.

def _poly_radius(theta, n, r_outer):
    mod = (theta % (2.0 * math.pi / n)) - math.pi / n
    return r_outer / math.cos(mod)


def _multi_nested_polygons(num_samples, noise, rng, K):
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        r = math.hypot(px, py)
        theta = math.atan2(py, px)
        label = K - 1
        for j in range(K):
            n = 3 + j
            r_limit = 1.6 + 1.15 * j
            if r < _poly_radius(theta, n, r_limit):
                label = j
                break
        points.append(Example2D(x, y, float(label)))
    return points


def multi_nested_polygons_4_data(num_samples, noise, rng):
    return _multi_nested_polygons(num_samples, noise, rng, 4)


def multi_nested_polygons_5_data(num_samples, noise, rng):
    return _multi_nested_polygons(num_samples, noise, rng, 5)


# --- Flower mandala -------------------------------------------------------
# K petals via r(θ) = 4.5·|cos(K·θ/2)|. Inside a petal: sector color.
# Outside a petal: opposite sector color. The result is a two-layer
# alternating flower (petals / sepals) with K-fold radial symmetry.

def _multi_flower(num_samples, noise, rng, K):
    points: list[Example2D] = []
    for _ in range(num_samples):
        x = rng.uniform(-6.0, 6.0)
        y = rng.uniform(-6.0, 6.0)
        nx = rng.uniform(-6.0, 6.0) * noise
        ny = rng.uniform(-6.0, 6.0) * noise
        px, py = x + nx, y + ny
        r = math.hypot(px, py)
        theta = math.atan2(py, px)
        r_flower = 4.5 * abs(math.cos(K * theta / 2.0))
        sector = int(((theta + math.pi) % (2.0 * math.pi)) / (2.0 * math.pi / K))
        sector = sector % K
        if r < r_flower:
            label = sector
        else:
            label = (sector + K // 2) % K
        points.append(Example2D(x, y, float(label)))
    return points


def multi_flower_6_data(num_samples, noise, rng):
    return _multi_flower(num_samples, noise, rng, 6)


def multi_flower_8_data(num_samples, noise, rng):
    return _multi_flower(num_samples, noise, rng, 8)