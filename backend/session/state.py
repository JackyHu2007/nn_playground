from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..core.nn import normalize_spec


def _default_dense(n: int = 4, act: str = "sigmoid") -> dict:
    return {"type": "dense", "n": int(n), "act": str(act),
            "bias": True, "norm": "none", "norm_pos": "post"}


def _default_expert_layer(n: int = 4, act: str = "relu") -> dict:
    return {"n": int(n), "act": str(act), "bias": True}


def _default_moe() -> dict:
    return {
        "type": "moe", "experts": 4, "top_k": 2, "soft": False,
        "router_bias": True, "router_temp": 1.0, "normalize": True,
        "aux_coef": 0.01,
        "expert": {"layers": [_default_expert_layer()]},
    }


def _migrate_legacy(
    network_shape, hidden_activations, layer_biases,
    layer_norms, layer_norm_positions, default_act,
) -> list[dict]:
    specs = []
    for i, n in enumerate(network_shape):
        specs.append({
            "type": "dense",
            "n": int(n),
            "act": hidden_activations[i] if i < len(hidden_activations) else default_act,
            "bias": layer_biases[i] if i < len(layer_biases) else True,
            "norm": layer_norms[i] if i < len(layer_norms) else "none",
            "norm_pos": layer_norm_positions[i] if i < len(layer_norm_positions) else "post",
        })
    return specs


def _migrate_moe_expert(exp: dict) -> dict:
    """Legacy {n, act, bias} -> {layers: [{n, act, bias}]}"""
    if isinstance(exp, dict) and "layers" in exp:
        return exp
    if isinstance(exp, dict):
        return {"layers": [{
            "n": int(exp.get("n", 4)),
            "act": str(exp.get("act", "relu")),
            "bias": bool(exp.get("bias", True)),
        }]}
    return {"layers": [_default_expert_layer()]}


@dataclass
class State:
    learning_rate: float = 0.03
    regularization_rate: float = 0.01
    noise: float = 0.0
    batch_size: int = 10
    perc_train_data: float = 50.0
    activation: str = "sigmoid"
    regularization: str = "none"
    problem: str = "classification"
    init_zero: bool = False
    optimizer: str = "SGD"
    online_data: bool = False
    residuals: list[list[int]] = field(default_factory=list)
    num_samples: int = 400
    layer_specs: list[dict] = field(default_factory=lambda: [_default_dense()])

    x: bool = True
    y: bool = True
    x_times_y: bool = False
    x_squared: bool = False
    y_squared: bool = False
    dataset: str = "circle"
    reg_dataset: str = "reg-plane"
    seed: str = "0"
    show_test_data: bool = False
    discretize: bool = False
    custom_dataset_id: str | None = None

    grad_clip_method: str = "none"
    grad_clip_value: float = 1.0

    _SERIALIZE_MAP: ClassVar[dict[str, str]] = {
        "learning_rate": "learningRate",
        "regularization_rate": "regularizationRate",
        "noise": "noise",
        "batch_size": "batchSize",
        "perc_train_data": "percTrainData",
        "activation": "activation",
        "regularization": "regularization",
        "problem": "problem",
        "init_zero": "initZero",
        "optimizer": "optimizer",
        "online_data": "onlineData",
        "residuals": "residuals",
        "num_samples": "numSamples",
        "layer_specs": "layerSpecs",
        "x": "x", "y": "y", "x_times_y": "xTimesY",
        "x_squared": "xSquared", "y_squared": "ySquared",
        "dataset": "dataset", "reg_dataset": "regDataset",
        "seed": "seed",
        "show_test_data": "showTestData",
        "discretize": "discretize",
        "custom_dataset_id": "customDatasetId",
        "grad_clip_method": "gradClipMethod",
        "grad_clip_value": "gradClipValue",
    }

    def ensure_specs(self) -> None:
        if not isinstance(self.layer_specs, list) or not self.layer_specs:
            self.layer_specs = [_default_dense()]
        for s in self.layer_specs:
            if isinstance(s, dict) and s.get("type") == "moe":
                s["expert"] = _migrate_moe_expert(s.get("expert") or {})
        self.layer_specs = [normalize_spec(s) for s in self.layer_specs]

    def has_moe(self) -> bool:
        return any(s.get("type") == "moe" for s in self.layer_specs)

    @property
    def network_shape(self) -> list[int]:
        out = []
        for s in self.layer_specs:
            if s["type"] == "dense":
                out.append(s["n"])
            else:
                out.append(s["expert"]["layers"][-1]["n"])
        return out

    @property
    def hidden_activations(self) -> list[str]:
        out = []
        for s in self.layer_specs:
            if s["type"] == "dense":
                out.append(s["act"])
            else:
                out.append(s["expert"]["layers"][-1]["act"])
        return out

    @property
    def layer_biases(self) -> list[bool]:
        out = []
        for s in self.layer_specs:
            if s["type"] == "dense":
                out.append(s["bias"])
            else:
                out.append(s["expert"]["layers"][-1]["bias"])
        return out

    @property
    def layer_norms(self) -> list[str]:
        return [s["norm"] if s["type"] == "dense" else "none"
                for s in self.layer_specs]

    @property
    def layer_norm_positions(self) -> list[str]:
        return [s["norm_pos"] if s["type"] == "dense" else "post"
                for s in self.layer_specs]

    def to_dict(self) -> dict[str, Any]:
        self.ensure_specs()
        out: dict[str, Any] = {}
        for snake, camel in self._SERIALIZE_MAP.items():
            value = getattr(self, snake)
            if snake == "residuals":
                out[camel] = [list(v) for v in value]
            elif snake == "layer_specs":
                out[camel] = [
                    {
                        **{k: v for k, v in s.items() if k != "expert"},
                        **({"expert": {
                            "layers": [dict(l) for l in s["expert"]["layers"]]
                        }} if s["type"] == "moe" else {}),
                    }
                    for s in value
                ]
            elif isinstance(value, list):
                out[camel] = list(value)
            else:
                out[camel] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "State":
        st = cls()
        for snake, camel in cls._SERIALIZE_MAP.items():
            if camel in data and data[camel] is not None:
                setattr(st, snake, data[camel])

        if not isinstance(data.get("layerSpecs"), list) or not data.get("layerSpecs"):
            legacy_shape = data.get("networkShape")
            if isinstance(legacy_shape, list) and legacy_shape:
                st.layer_specs = _migrate_legacy(
                    legacy_shape,
                    data.get("hiddenActivations") or [],
                    data.get("layerBiases") or [],
                    data.get("layerNorms") or [],
                    data.get("layerNormPositions") or [],
                    data.get("activation", "sigmoid"),
                )
        st.ensure_specs()
        if not isinstance(st.residuals, list):
            st.residuals = []
        return st

    def copy(self) -> "State":
        return State.from_dict(self.to_dict())