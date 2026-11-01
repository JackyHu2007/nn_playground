"""Dataset registry: name -> DatasetGenerator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import generators, loaders
from .types import DataGenerator


@dataclass
class DatasetGenerator:
    train_generator: DataGenerator
    test_generator: Optional[DataGenerator] = None
    is_probabilistic: bool = True
    #: 2 = binary classification, 0 = regression (N/A), K >= 3 = multiclass.
    num_classes: int = 2

    def __post_init__(self) -> None:
        if self.test_generator is None:
            self.test_generator = self.train_generator

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DatasetGenerator):
            return NotImplemented
        return (
            self.train_generator is other.train_generator
            and self.test_generator is other.test_generator
        )

    def __hash__(self) -> int:
        return hash((id(self.train_generator), id(self.test_generator)))


DATASETS: dict[str, DatasetGenerator] = {
    "linear": DatasetGenerator(generators.classify_linear_data),
    "diagonal": DatasetGenerator(generators.classify_diagonal_train_data,
                                 generators.classify_diagonal_test_data),
    "xor": DatasetGenerator(generators.classify_xor_data),
    "gauss": DatasetGenerator(generators.classify_gauss_data),
    "circle": DatasetGenerator(generators.classify_circle_data),
    "spiral": DatasetGenerator(generators.classify_spiral_data),
    "moons": DatasetGenerator(generators.classify_moons_data),

    "checkerboard": DatasetGenerator(generators.classify_checkerboard_data),
    "rings": DatasetGenerator(generators.classify_rings_data),
    "sine-clf": DatasetGenerator(generators.classify_sine_data),

    "rose": DatasetGenerator(generators.classify_rose_data),
    "parabola": DatasetGenerator(generators.classify_parabola_data),

    "mandala": DatasetGenerator(generators.classify_mandala_data),
    "dense-spiral": DatasetGenerator(generators.classify_dense_spiral_data),
    "hf-checkerboard": DatasetGenerator(generators.classify_hf_checkerboard_data),
    "star": DatasetGenerator(generators.classify_star_data),

    "rocks": DatasetGenerator(loaders.classify_rocks_train_data,
                              loaders.classify_rocks_test_data,
                              is_probabilistic=False),
    "poroperm": DatasetGenerator(loaders.classify_poroperm_train_data,
                                 loaders.classify_poroperm_test_data,
                                 is_probabilistic=False),
}

# ---------------------------------------------------------------------------
# Multiclass datasets. All are K >= 3, many are deliberately hard.
# ---------------------------------------------------------------------------

MULTI_DATASETS: dict[str, DatasetGenerator] = {
    # --- original 3 (kept for backward compatibility with old URLs) ---
    "multi-blobs": DatasetGenerator(generators.multi_blobs_data, num_classes=3),
    "multi-spirals": DatasetGenerator(generators.multi_spirals_data, num_classes=3),
    "multi-concentric": DatasetGenerator(generators.multi_concentric_data, num_classes=3),

    # --- moderate (4 classes) ---
    "multi-blobs-4": DatasetGenerator(generators.multi_blobs_4_data, num_classes=4),
    "multi-spirals-4": DatasetGenerator(generators.multi_spirals_4_data, num_classes=4),
    "multi-moons-4": DatasetGenerator(generators.multi_moons_4_data, num_classes=4),
    "multi-rings-4": DatasetGenerator(generators.multi_rings_4_data, num_classes=4),
    "multi-checkerboard-4": DatasetGenerator(generators.multi_checkerboard_4_data, num_classes=4),

    # --- harder (5-6 classes) ---
    "multi-blobs-5": DatasetGenerator(generators.multi_blobs_5_data, num_classes=5),
    "multi-spirals-5": DatasetGenerator(generators.multi_spirals_5_data, num_classes=5),
    "multi-concentric-5": DatasetGenerator(generators.multi_concentric_5_data, num_classes=5),
    "multi-checkerboard-6": DatasetGenerator(generators.multi_checkerboard_6_data, num_classes=6),
    "multi-mandala-6": DatasetGenerator(generators.multi_mandala_6_data, num_classes=6),
    "multi-rose-6": DatasetGenerator(generators.multi_rose_6_data, num_classes=6),

    # --- very hard (7-8 classes) ---
    "multi-spirals-7": DatasetGenerator(generators.multi_spirals_7_data, num_classes=7),
    "multi-blobs-8": DatasetGenerator(generators.multi_blobs_8_data, num_classes=8),
    "multi-hf-spiral-8": DatasetGenerator(generators.multi_hf_spiral_8_data, num_classes=8),

    # --- hell tier (10 classes) ---
    "multi-mandala-10": DatasetGenerator(generators.multi_mandala_10_data, num_classes=10),
    "multi-hf-checkerboard-10": DatasetGenerator(generators.multi_hf_checkerboard_10_data, num_classes=10),
    "multi-dense-spiral-10": DatasetGenerator(generators.multi_dense_spiral_10_data, num_classes=10),
    # --- NEW: visually distinctive datasets ---
    # Voronoi / cellular mosaic
    "multi-voronoi-5": DatasetGenerator(generators.multi_voronoi_5_data, num_classes=5),
    "multi-voronoi-7": DatasetGenerator(generators.multi_voronoi_7_data, num_classes=7),
    "multi-voronoi-9": DatasetGenerator(generators.multi_voronoi_9_data, num_classes=9),

    # Hexagonal honeycomb (adjacent cells always differ)
    "multi-honeycomb-3": DatasetGenerator(generators.multi_honeycomb_3_data, num_classes=3),
    "multi-honeycomb-4": DatasetGenerator(generators.multi_honeycomb_4_data, num_classes=4),
    "multi-honeycomb-6": DatasetGenerator(generators.multi_honeycomb_6_data, num_classes=6),

    # Interference ripples (K point sources)
    "multi-interference-3": DatasetGenerator(generators.multi_interference_3_data, num_classes=3),
    "multi-interference-5": DatasetGenerator(generators.multi_interference_5_data, num_classes=5),
    "multi-interference-7": DatasetGenerator(generators.multi_interference_7_data, num_classes=7),

    # Spiral galaxy arms (K = arms + 1 background)
    "multi-galaxy-4": DatasetGenerator(generators.multi_galaxy_4_data, num_classes=4),
    "multi-galaxy-5": DatasetGenerator(generators.multi_galaxy_5_data, num_classes=5),

    # Sierpinski carpet (fractal, 2 recursion levels)
    "multi-sierpinski-4": DatasetGenerator(generators.multi_sierpinski_4_data, num_classes=4),
    "multi-sierpinski-8": DatasetGenerator(generators.multi_sierpinski_8_data, num_classes=8),

    # Mosaic of geometric shapes
    "multi-shapes-4": DatasetGenerator(generators.multi_shapes_4_data, num_classes=4),
    "multi-shapes-5": DatasetGenerator(generators.multi_shapes_5_data, num_classes=5),
    "multi-shapes-6": DatasetGenerator(generators.multi_shapes_6_data, num_classes=6),

    # Nested regular polygons
    "multi-nested-polygons-4": DatasetGenerator(generators.multi_nested_polygons_4_data, num_classes=4),
    "multi-nested-polygons-5": DatasetGenerator(generators.multi_nested_polygons_5_data, num_classes=5),

    # Flower / mandala with K petals
    "multi-flower-6": DatasetGenerator(generators.multi_flower_6_data, num_classes=6),
    "multi-flower-8": DatasetGenerator(generators.multi_flower_8_data, num_classes=8),
}

REG_DATASETS: dict[str, DatasetGenerator] = {
    "reg-plane": DatasetGenerator(generators.regress_plane, num_classes=0),
    "reg-gauss": DatasetGenerator(generators.regress_gaussian, num_classes=0),

    "reg-saddle": DatasetGenerator(generators.regress_saddle, num_classes=0),
    "reg-ripples": DatasetGenerator(generators.regress_ripples, num_classes=0),
    "reg-sine": DatasetGenerator(generators.regress_sine_surface, num_classes=0),

    "reg-mexican": DatasetGenerator(generators.regress_mexican_hat, num_classes=0),
    "reg-egg": DatasetGenerator(generators.regress_egg_carton, num_classes=0),

    "reg-rastrigin": DatasetGenerator(generators.regress_rastrigin, num_classes=0),
    "reg-hf-sine": DatasetGenerator(generators.regress_hf_sine, num_classes=0),
    "reg-spiral-ramp": DatasetGenerator(generators.regress_spiral_ramp, num_classes=0),
    "reg-ackley": DatasetGenerator(generators.regress_ackley, num_classes=0),

    "reg-porosity": DatasetGenerator(loaders.regress_porosity_train_data,
                                     loaders.regress_porosity_test_data,
                                     is_probabilistic=False, num_classes=0),
    "reg-dts": DatasetGenerator(loaders.regress_dts_train_data,
                                loaders.regress_dts_test_data,
                                is_probabilistic=False, num_classes=0),
}


def get_key_from_value(mapping: dict[str, DatasetGenerator],
                       value: DatasetGenerator) -> Optional[str]:
    for key, v in mapping.items():
        if v == value:
            return key
    return None