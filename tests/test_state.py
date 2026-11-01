"""State serialization + legacy migration."""
from backend.session.state import State


def test_default_state():
    s = State()
    assert s.problem == "classification"
    assert s.learning_rate == 0.03
    assert len(s.layer_specs) == 1
    assert s.layer_specs[0]["type"] == "dense"


def test_roundtrip():
    s = State()
    s.learning_rate = 0.05
    s.noise = 12
    s.layer_specs = [{"type": "dense", "n": 5, "act": "relu", "bias": True,
                       "norm": "layernorm", "norm_pos": "pre"}]
    s.residuals = [[0, 2]]
    s2 = State.from_dict(s.to_dict())
    assert s2.learning_rate == 0.05
    assert s2.noise == 12
    assert s2.residuals == [[0, 2]]
    assert s2.layer_specs[0]["n"] == 5


def test_dict_uses_camelcase():
    d = State().to_dict()
    for k in ("learningRate", "layerSpecs", "numSamples", "gradClipMethod"):
        assert k in d
    assert "learning_rate" not in d


def test_copy_independent():
    s = State()
    s.residuals.append([0, 1])
    c = s.copy()
    c.residuals.append([1, 2])
    assert s.residuals == [[0, 1]]
    assert len(c.residuals) == 2


def test_legacy_network_shape_migration():
    legacy = {
        "networkShape": [4, 3],
        "hiddenActivations": ["relu", "tanh"],
        "layerBiases": [True, False],
        "layerNorms": ["none", "layernorm"],
        "layerNormPositions": ["post", "pre"],
    }
    s = State.from_dict(legacy)
    assert len(s.layer_specs) == 2
    assert s.layer_specs[0]["act"] == "relu"
    assert s.layer_specs[1]["norm"] == "layernorm"
    assert s.layer_specs[1]["norm_pos"] == "pre"
    assert s.layer_specs[1]["bias"] is False


def test_legacy_moe_expert_migration():
    s = State.from_dict({"layerSpecs": [{
        "type": "moe", "experts": 4,
        "expert": {"n": 6, "act": "relu"},
    }]})
    exp = s.layer_specs[0]["expert"]
    assert "layers" in exp
    assert exp["layers"][0]["n"] == 6


def test_has_moe():
    s = State()
    assert not s.has_moe()
    s.layer_specs = [
        {"type": "moe", "experts": 2,
         "expert": {"layers": [{"n": 2, "act": "relu"}]}},
    ]
    s.ensure_specs()
    assert s.has_moe()


def test_network_shape_property():
    s = State()
    s.layer_specs = [
        {"type": "dense", "n": 8, "act": "relu", "bias": True,
         "norm": "none", "norm_pos": "post"},
        {"type": "moe", "experts": 2,
         "expert": {"layers": [{"n": 5, "act": "relu"}]}},
    ]
    s.ensure_specs()
    assert s.network_shape == [8, 5]


def test_empty_specs_becomes_default():
    s = State()
    s.layer_specs = []
    s.ensure_specs()
    assert len(s.layer_specs) == 1


def test_unknown_keys_ignored():
    s = State.from_dict({"learningRate": 0.01, "bogus": "x"})
    assert s.learning_rate == 0.01