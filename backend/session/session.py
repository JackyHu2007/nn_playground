from __future__ import annotations
from typing import Any, Optional
import numpy as np

from ..core.boundary import compute_boundary, reduce_matrix
from ..core.compile_py import compile_network_to_py
from ..core.errors import SQUARE
from ..core.nn import Network
from ..core.optimizers import make_optimizer
from ..core.regularizations import REGULARIZATIONS
from ..core.seed import make_rng
from ..core.updater import apply_update
from ..data.registry import DATASETS, MULTI_DATASETS, REG_DATASETS, DatasetGenerator
from ..data.types import Example2D
from .state import State

_FEATURE_DEFS = [
    ("x", "x", lambda x, y: x),
    ("y", "y", lambda x, y: y),
    ("x_times_y", "xTimesY", lambda x, y: x * y),
    ("x_squared", "xSquared", lambda x, y: x * x),
    ("y_squared", "ySquared", lambda x, y: y * y),
]

_KEY_TO_ATTR = {
    "learningRate": "learning_rate",
    "regularizationRate": "regularization_rate",
    "noise": "noise",
    "batchSize": "batch_size",
    "percTrainData": "perc_train_data",
    "activation": "activation",
    "regularization": "regularization",
    "problem": "problem",
    "initZero": "init_zero",
    "optimizer": "optimizer",
    "onlineData": "online_data",
    "residuals": "residuals",
    "layerSpecs": "layer_specs",
    "numSamples": "num_samples",
    "x": "x", "y": "y", "xTimesY": "x_times_y",
    "xSquared": "x_squared", "ySquared": "y_squared",
    "dataset": "dataset", "regDataset": "reg_dataset",
    "seed": "seed",
    "showTestData": "show_test_data",
    "discretize": "discretize",
    "customDatasetId": "custom_dataset_id",
    "gradClipMethod": "grad_clip_method",
    "gradClipValue": "grad_clip_value",
}

_PROBLEMS = {"classification", "multiclass", "regression"}
_NUM_SAMPLES_MIN = 50
_NUM_SAMPLES_MAX = 2000


class TrainingSession:
    def __init__(self, state: State, density: int = 100, manager=None):
        self.state = state
        self.density = density
        self.manager = manager
        self.iter = 0
        self.loss_train = 0.0
        self.loss_test = 0.0
        self.aux_loss_train = 0.0
        self.optimizer = make_optimizer(state.optimizer, state.learning_rate)
        self.network: Optional[Network] = None
        self.train_data, self.test_data = [], []

        self._custom_points: list[Example2D] | None = None
        self._custom_num_classes: int = 1

        # pending grads
        self._pending_grad_w = []
        self._pending_grad_b = []
        self._pending_grad_gamma = []
        self._pending_grad_beta = []
        self._pending_grad_rw = []
        self._pending_grad_rb = []
        self._pending_grad_ew = []
        self._pending_grad_eb = []
        self._pending_count = 0

        # Weight-seed counter: identifies which initialization draw to use.
        # `reset()` keeps it constant (same architecture + same seed ⇒ same
        # initial weights, useful for A/B comparisons); `regenerate()` bumps
        # it to draw a fresh initialization
        self._weight_seed_counter = 0
        self._online_counter = 0
        self._update_count = 0

        self._boundary_full: dict = {}
        self._output_boundary = None
        self._node_boundaries: dict = {}
        self._moe_probe: dict = {}
        self._expert_outputs: dict = {}
        self._expert_boundary_cache: dict = {}
        self._expert_boundaries_small: dict = {}
        self._num_classes = 1

        self._generate_data()
        self._build_network()
        self._compute_losses()
        self._update_boundaries()

    # ------------------------------------------------------------------ data
    def _current_dataset(self) -> DatasetGenerator:
        if self._custom_points:
            pts = list(self._custom_points)
            return DatasetGenerator(
                train_generator=lambda n, noise, rng: list(pts),
                test_generator=lambda n, noise, rng: list(pts),
                is_probabilistic=False,
                num_classes=self._custom_num_classes,
            )
        if self.state.problem == "regression":
            return REG_DATASETS[self.state.reg_dataset]
        if self.state.problem == "multiclass":
            if self.state.dataset in MULTI_DATASETS:
                return MULTI_DATASETS[self.state.dataset]
            return next(iter(MULTI_DATASETS.values()))
        return DATASETS[self.state.dataset]

    def _clamp_num_samples(self) -> int:
        return max(_NUM_SAMPLES_MIN, min(_NUM_SAMPLES_MAX, int(self.state.num_samples)))

    def _generate_data(self) -> None:
        if self._custom_points:
            self._generate_custom_split()
            return
        gen = self._current_dataset()
        n_total = self._clamp_num_samples()
        num_train = int(n_total * self.state.perc_train_data / 100)
        rng = make_rng(self.state.seed, 0)
        self.train_data = list(gen.train_generator(num_train, self.state.noise / 100.0, rng))
        self.test_data = list(gen.test_generator(n_total - num_train, self.state.noise / 100.0, rng))

    def _generate_custom_split(self) -> None:
        points = list(self._custom_points or [])
        n = len(points)
        if n == 0:
            self.train_data, self.test_data = [], []
            return
        num_train = max(1, min(n - 1, int(n * self.state.perc_train_data / 100)))
        rng = make_rng(self.state.seed, 0)
        idx = rng.permutation(n)
        shuffled = [points[int(i)] for i in idx]
        self.train_data = shuffled[:num_train]
        self.test_data = shuffled[num_train:]

    def _regen_train(self) -> None:
        if self._custom_points:
            return
        gen = self._current_dataset()
        if not gen.is_probabilistic:
            return
        self._online_counter += 1
        n_total = self._clamp_num_samples()
        num_train = int(n_total * self.state.perc_train_data / 100)
        rng = make_rng(self.state.seed, self._online_counter)
        self.train_data = list(gen.train_generator(num_train, self.state.noise / 100.0, rng))

    # --------------------------------------------------------------- inputs
    def construct_input_ids(self) -> list[str]:
        ids = [input_id for attr, input_id, _ in _FEATURE_DEFS if getattr(self.state, attr)]
        if not ids:
            raise ValueError("At least one input feature must be enabled")
        return ids

    def construct_input(self, xx, yy) -> np.ndarray:
        xx_arr = np.atleast_1d(np.asarray(xx, dtype=np.float64))
        yy_arr = np.atleast_1d(np.asarray(yy, dtype=np.float64))
        cols = [fn(xx_arr, yy_arr) for attr, _, fn in _FEATURE_DEFS if getattr(self.state, attr)]
        return np.stack(cols, axis=1)

    # --------------------------------------------------------------- build
    def _build_network(self) -> None:
        self.state.ensure_specs()
        rng = make_rng(self.state.seed + "_w", self._weight_seed_counter)
        input_ids = self.construct_input_ids()

        is_multiclass = self.state.problem == "multiclass"
        if is_multiclass:
            n_out = int(self._current_dataset().num_classes)
        else:
            n_out = 1

        if is_multiclass or self.state.problem == "regression":
            out_act = "linear"
        else:
            out_act = "tanh"

        n_in = len(input_ids)
        hidden_n = list(self.state.network_shape)
        shape = [n_in] + hidden_n + [n_out]

        valid_res = []
        seen = set()
        for s, d in self.state.residuals:
            if 0 <= s < d <= len(shape) - 1 and shape[s] == shape[d]:
                if is_multiclass and d == len(shape) - 1:
                    continue
                if (s, d) not in seen:
                    valid_res.append([s, d])
                    seen.add((s, d))
        self.state.residuals = valid_res

        self.network = Network(
            input_ids=input_ids,
            layer_specs=self.state.layer_specs,
            output_size=n_out,
            output_activation=out_act,
            output_bias=True,
            is_multiclass=is_multiclass,
            init_zero=self.state.init_zero,
            rng=rng,
            residuals=valid_res,
        )

        self.optimizer.reset()

        self._pending_grad_w = [np.zeros_like(w) if w is not None else None
                                for w in self.network.weights]
        self._pending_grad_b = [np.zeros_like(b) if b is not None else None
                                for b in self.network.biases]
        self._pending_grad_gamma = [np.zeros_like(g) if g is not None else None
                                    for g in self.network.gammas]
        self._pending_grad_beta = [np.zeros_like(b) if b is not None else None
                                   for b in self.network.betas]
        self._pending_grad_rw = [np.zeros_like(w) if w is not None else None
                                 for w in self.network.router_weights]
        self._pending_grad_rb = [np.zeros_like(b) if b is not None else None
                                 for b in self.network.router_biases]
        self._pending_grad_ew = []
        self._pending_grad_eb = []
        for l in range(self.network.num_layers):
            ew = self.network.expert_weights[l]
            eb = self.network.expert_biases[l]
            if ew is None:
                self._pending_grad_ew.append(None)
                self._pending_grad_eb.append(None)
            else:
                self._pending_grad_ew.append(
                    [[np.zeros_like(w) for w in expert_w] for expert_w in ew]
                )
                self._pending_grad_eb.append(
                    [[np.zeros_like(b) for b in expert_b] for expert_b in eb]
                )
        self._pending_count = 0
        self._expert_boundary_cache = {}
        self._expert_boundaries_small = {}

    # ---------------------------------------------------------------- loss
    def _get_loss(self, data) -> float:
        if not data or self.network is None:
            return 0.0
        X = self.construct_input(
            np.array([p.x for p in data]), np.array([p.y for p in data])
        )
        T = np.array([p.label for p in data])
        if self.state.problem == "multiclass":
            p = self.network.forward(X)
            idx = T.astype(int).ravel()
            eps = 1e-12
            return float(-np.mean(np.log(p[np.arange(len(idx)), idx] + eps)))
        return float(np.mean(0.5 * (self.network.forward(X).ravel() - T) ** 2))

    def _compute_losses(self) -> None:
        self.loss_train = self._get_loss(self.train_data)
        self.loss_test = self._get_loss(self.test_data)

        self.aux_loss_train = 0.0
        if self.network and self.state.has_moe() and self.train_data:
            X = self.construct_input(
                np.array([p.x for p in self.train_data]),
                np.array([p.y for p in self.train_data]),
            )
            _, layer_cache, _ = self.network.forward_with_cache(X)
            total, _ = self.network.compute_aux_losses_from_cache(layer_cache)
            self.aux_loss_train = total

    def _update_boundaries(self) -> None:
        full, moe_probe, expert_outs = compute_boundary(
            self.network, self.construct_input, density=self.density
        )
        self._boundary_full = full
        self._moe_probe = moe_probe
        self._expert_outputs = expert_outs
        self._expert_boundary_cache = {}

        out_ids = self.network.layer_node_ids[-1]
        if self.state.problem == "multiclass":
            self._node_boundaries = {nid: reduce_matrix(m, 10) for nid, m in full.items()}
            stack = np.stack([full[nid] for nid in out_ids], axis=0)
            self._output_boundary = np.argmax(stack, axis=0).astype(np.float64)
            self._num_classes = len(out_ids)
        else:
            self._output_boundary = full[out_ids[0]]
            self._node_boundaries = {nid: reduce_matrix(m, 10) for nid, m in full.items()}
            self._num_classes = 1

        self._expert_boundaries_small = {}
        for l in range(self.network.num_layers):
            spec = self.network._get_spec(l)
            if spec.get("type") != "moe":
                continue
            entry = self._expert_outputs.get(l)
            if entry is None:
                continue
            D = entry["depth"]
            for d in range(D):
                arr = entry["outs"][d]
                n_d = arr.shape[-1]
                for e in range(spec["experts"]):
                    for i in range(n_d):
                        key = f"E:{l + 1}:{e}:{d}:{i}"
                        self._expert_boundaries_small[key] = reduce_matrix(
                            arr[e, :, :, i], 10
                        )

    def get_node_boundary(self, node_id: str):
        if node_id in self._boundary_full:
            return self._boundary_full[node_id].tolist()
        if node_id in self._expert_boundary_cache:
            return self._expert_boundary_cache[node_id].tolist()

        if not node_id.startswith("E:"):
            return None
        try:
            parts = node_id.split(":")
            if len(parts) == 5:
                _, l_str, e_str, d_str, i_str = parts
                d = int(d_str)
            elif len(parts) == 4:
                _, l_str, e_str, i_str = parts
                d = -1
            else:
                return None
            l = int(l_str) - 1
            e = int(e_str)
            i = int(i_str)
        except Exception:
            return None

        spec = self.network._get_spec(l)
        if spec.get("type") != "moe":
            return None
        entry = self._expert_outputs.get(l)
        if entry is None:
            return None
        D = entry["depth"]
        if d < 0:
            d = D - 1
        if d >= D:
            return None
        if e < 0 or e >= spec["experts"]:
            return None
        n_d = entry["outs"][d].shape[-1]
        if i < 0 or i >= n_d:
            return None

        m = entry["outs"][d][e, :, :, i]
        self._expert_boundary_cache[node_id] = m
        return m.tolist()

    # --------------------------------------------------------------- update
    def _flush_pending(self) -> None:
        if self._pending_count == 0:
            return
        apply_update(
            self.network,
            self.optimizer,
            self._pending_grad_w,
            self._pending_grad_b,
            self._pending_grad_gamma,
            self._pending_grad_beta,
            self._pending_grad_rw,
            self._pending_grad_rb,
            self._pending_grad_ew,
            self._pending_grad_eb,
            num_samples=self._pending_count,
            regularization=REGULARIZATIONS.get(self.state.regularization),
            reg_rate=self.state.regularization_rate,
            grad_clip_method=self.state.grad_clip_method,
            grad_clip_value=self.state.grad_clip_value,
        )
        for l in range(self.network.num_layers):
            if self._pending_grad_w[l] is not None:
                self._pending_grad_w[l].fill(0.0)
                self._pending_grad_b[l].fill(0.0)
            if self._pending_grad_gamma[l] is not None:
                self._pending_grad_gamma[l].fill(0.0)
                self._pending_grad_beta[l].fill(0.0)
            if self._pending_grad_rw[l] is not None:
                self._pending_grad_rw[l].fill(0.0)
                self._pending_grad_rb[l].fill(0.0)
                for e, expert_w in enumerate(self._pending_grad_ew[l]):
                    for d in range(len(expert_w)):
                        self._pending_grad_ew[l][e][d].fill(0.0)
                        self._pending_grad_eb[l][e][d].fill(0.0)
        self._pending_count = 0
        self._update_count += 1

    def one_step(self, update_boundaries: bool = True) -> None:
        self.iter += 1
        n = len(self.train_data)
        if n == 0:
            self._compute_losses()
            if update_boundaries:
                self._update_boundaries()
            return

        X = self.construct_input(
            np.array([p.x for p in self.train_data]),
            np.array([p.y for p in self.train_data]),
        )
        T = np.array([p.label for p in self.train_data])
        batch_size = max(1, self.state.batch_size)
        i = 0
        while i < n:
            need = batch_size - self._pending_count
            if need <= 0:
                self._flush_pending()
                continue
            take = min(need, n - i)
            Xb, Tb = X[i:i + take], T[i:i + take]

            _, layer_cache, out_cache = self.network.forward_with_cache(Xb)
            grads = self.network.backward(layer_cache, out_cache, Tb, SQUARE)

            for l in range(self.network.num_layers):
                if grads["weights"][l] is not None:
                    self._pending_grad_w[l] += grads["weights"][l]
                    self._pending_grad_b[l] += grads["biases"][l]
                    if grads["gammas"][l] is not None:
                        self._pending_grad_gamma[l] += grads["gammas"][l]
                        self._pending_grad_beta[l] += grads["betas"][l]
                if grads["router_weights"][l] is not None:
                    self._pending_grad_rw[l] += grads["router_weights"][l]
                    self._pending_grad_rb[l] += grads["router_biases"][l]
                    for e in range(len(grads["expert_weights"][l])):
                        D = len(grads["expert_weights"][l][e])
                        for d in range(D):
                            self._pending_grad_ew[l][e][d] += grads["expert_weights"][l][e][d]
                            self._pending_grad_eb[l][e][d] += grads["expert_biases"][l][e][d]

            self._pending_count += take
            i += take
            if self._pending_count == batch_size:
                self._flush_pending()

        if self.state.online_data:
            self._flush_pending()
            self._regen_train()

        self._compute_losses()
        if update_boundaries:
            self._update_boundaries()

    # ------------------------------------------------------------- control
    def reset(self) -> None:
        self.iter = 0
        self._build_network()
        self._compute_losses()
        self._update_boundaries()

    def regenerate(self) -> None:
        self.state.seed = f"{np.random.random():.5f}"
        self._online_counter = 0
        self._weight_seed_counter += 1
        self._generate_data()
        self.reset()

    # ---- custom datasets ----
    def _infer_problem_and_classes(self, points: list[Example2D]) -> tuple[str, int]:
        unique = sorted(set(float(p.label) for p in points))
        n = len(unique)
        if n < 2:
            raise ValueError("need at least 2 distinct labels")
        is_int = all(abs(u - round(u)) < 1e-9 for u in unique)
        if n >= 3 and is_int and min(unique) == 0.0 and max(unique) == float(n - 1):
            return "multiclass", n
        if n == 2:
            return "classification", 2
        return "regression", 1

    def set_custom_dataset_by_id(self, did: str | None) -> None:
        if not did:
            self._custom_points = None
            self._custom_num_classes = 1
            self.state.custom_dataset_id = None
            self.reset()
            return
        if self.manager is None:
            raise ValueError("SessionManager not attached to TrainingSession")
        points = self.manager.get_custom(did)
        if points is None:
            raise ValueError(f"unknown custom dataset id {did!r}")
        problem, num_classes = self._infer_problem_and_classes(points)
        self.state.problem = problem
        self.state.custom_dataset_id = did
        self._custom_points = points
        self._custom_num_classes = num_classes

        self.optimizer = make_optimizer(self.state.optimizer, self.state.learning_rate)
        self.iter = 0
        self._online_counter = 0
        self._weight_seed_counter = 0
        self._update_count = 0
        self._generate_data()
        self._build_network()
        self._compute_losses()
        self._update_boundaries()

    def load_state(self, new_state: State) -> None:
        did = new_state.custom_dataset_id
        custom_points = None
        if did and self.manager is not None:
            custom_points = self.manager.get_custom(did)
        if custom_points is None:
            new_state.custom_dataset_id = None

        if new_state.problem not in _PROBLEMS:
            new_state.problem = "classification"
        self.state = new_state
        if new_state.problem == "multiclass" and new_state.dataset not in MULTI_DATASETS:
            new_state.dataset = next(iter(MULTI_DATASETS.keys()))
        elif new_state.problem == "classification" and new_state.dataset not in DATASETS:
            new_state.dataset = next(iter(DATASETS.keys()))
        elif new_state.problem == "regression" and new_state.reg_dataset not in REG_DATASETS:
            new_state.reg_dataset = next(iter(REG_DATASETS.keys()))

        if custom_points:
            self._custom_points = custom_points
            try:
                problem, num_classes = self._infer_problem_and_classes(custom_points)
                self.state.problem = problem
                self._custom_num_classes = num_classes
            except ValueError:
                self._custom_points = None
                self._custom_num_classes = 1
        else:
            self._custom_points = None
            self._custom_num_classes = 1

        self.optimizer = make_optimizer(new_state.optimizer, new_state.learning_rate)
        self.iter = 0
        self._online_counter = 0
        self._weight_seed_counter = 0
        self._update_count = 0
        self._generate_data()
        self._build_network()
        self._compute_losses()
        self._update_boundaries()


    def set(self, key: str, value: Any) -> None:
        if key not in _KEY_TO_ATTR:
            raise ValueError(f"Unknown state key: {key!r}")
        attr = _KEY_TO_ATTR[key]

        if key == "customDatasetId":
            self.set_custom_dataset_by_id(value)
            return

        if key == "layerSpecs":
            if not isinstance(value, list):
                raise ValueError("layerSpecs must be a list")
            self.state.layer_specs = value
            self.state.ensure_specs()
            n_in = len(self.construct_input_ids())
            shape = [n_in] + self.state.network_shape + (
                [int(self._current_dataset().num_classes)]
                if self.state.problem == "multiclass" else [1]
            )
            self.state.residuals = [
                [s, d] for s, d in self.state.residuals
                if 0 <= s < d <= len(shape) - 1 and shape[s] == shape[d]
            ]
            self.reset()
            return

        if key == "learningRate":
            self.state.learning_rate = float(value)
            if hasattr(self.optimizer, "lr"):
                self.optimizer.lr = self.state.learning_rate
            return

        if key == "optimizer":
            self.state.optimizer = str(value)
            self.optimizer = make_optimizer(self.state.optimizer, self.state.learning_rate)
            return

        if key in {"batchSize", "onlineData", "showTestData", "discretize", "regularizationRate", "gradClipMethod",
                   "gradClipValue"}:
            if key == "batchSize":
                value = int(value)
            elif key in {"onlineData", "showTestData", "discretize"}:
                value = bool(value)
            elif key in {"regularizationRate", "gradClipValue"}:
                value = float(value)
            elif key == "gradClipMethod":
                value = str(value)
            setattr(self.state, attr, value)
            return

        if key == "numSamples":
            self.state.num_samples = max(_NUM_SAMPLES_MIN, min(_NUM_SAMPLES_MAX, int(value)))
            self.regenerate()
            return

        if key == "residuals":
            self.state.residuals = [[int(s), int(d)] for s, d in value]
            self.reset()
            return

        if key in {"activation", "regularization", "initZero",
                   "x", "y", "xTimesY", "xSquared", "ySquared"}:
            if key == "initZero":
                value = bool(value)
            elif key in {"x", "y", "xTimesY", "xSquared", "ySquared"}:
                value = bool(value)
            else:
                value = str(value)
            setattr(self.state, attr, value)
            self.reset()
            return

        if key in {"dataset", "regDataset", "noise", "percTrainData", "problem"}:
            if key in {"noise", "percTrainData"}:
                value = float(value)
            elif key in {"dataset", "regDataset", "problem"}:
                value = str(value)

            if key == "problem":
                if value not in _PROBLEMS:
                    raise ValueError(f"unknown problem {value!r}")
                # switching away from a custom dataset must drop it
                if self._custom_points is not None:
                    self._custom_points = None
                    self._custom_num_classes = 1
                    self.state.custom_dataset_id = None
                self.state.problem = value
                if value == "multiclass" and self.state.dataset not in MULTI_DATASETS:
                    self.state.dataset = next(iter(MULTI_DATASETS.keys()))
                elif value == "classification" and self.state.dataset not in DATASETS:
                    self.state.dataset = next(iter(DATASETS.keys()))
                self.regenerate()
                return

            if key == "dataset":
                if self.state.problem == "multiclass":
                    if value not in MULTI_DATASETS:
                        raise ValueError(f"unknown multiclass dataset {value!r}")
                else:
                    if value not in DATASETS:
                        raise ValueError(f"unknown dataset {value!r}")
                if self._custom_points is not None:
                    self._custom_points = None
                    self._custom_num_classes = 1
                    self.state.custom_dataset_id = None
                self.state.dataset = value
                self.regenerate()
                return

            if key == "regDataset":
                if value not in REG_DATASETS:
                    raise ValueError(f"unknown regression dataset {value!r}")
                if self._custom_points is not None:
                    self._custom_points = None
                    self._custom_num_classes = 1
                    self.state.custom_dataset_id = None
                self.state.reg_dataset = value
                self.regenerate()
                return

            setattr(self.state, attr, value)
            self.regenerate()
            return

    # ---------------------------------------------------------------- frame
    def get_frame(self, include_boundaries: bool = True) -> dict[str, Any]:
        is_multiclass = self.state.problem == "multiclass"
        if is_multiclass:
            output_act = "softmax"
        elif self.state.problem == "regression":
            output_act = "linear"
        else:
            output_act = "tanh"

        frame: dict[str, Any] = {
            "iter": self.iter,
            "lossTrain": self.loss_train,
            "lossTest": self.loss_test,
            "auxLossTrain": self.aux_loss_train,
            "networkShape": list(self.network.layer_sizes),
            "nodeIds": [list(ids) for ids in self.network.layer_node_ids],
            "inputIds": list(self.network.input_ids),
            "activation": self.state.activation,
            "outputActivation": output_act,
            "outputMode": "class" if is_multiclass else "value",
            "numClasses": self._num_classes if is_multiclass else 1,
            "weights": [w.tolist() if w is not None else None for w in self.network.weights],
            "biases": [b.tolist() if b is not None else None for b in self.network.biases],
            "routerWeights": [w.tolist() if w is not None else None for w in self.network.router_weights],
            "expertWeights": [
                [
                    [d_w.tolist() for d_w in e_w] for e_w in w_list
                ] if w_list is not None else None
                for w_list in self.network.expert_weights
            ],
            "gammas": [g.tolist() if g is not None else None for g in self.network.gammas],
            "betas": [b.tolist() if b is not None else None for b in self.network.betas],
            "useBias": list(self.network.use_bias),
            "normTypes": list(self.network.norm_types),
            "normPositions": list(self.network.norm_positions),
            "residuals": self.state.residuals,
            "discretize": self.state.discretize,
            "trainPoints": [[p.x, p.y, p.label] for p in self.train_data],
            "testPoints": (
                [[p.x, p.y, p.label] for p in self.test_data]
                if self.state.show_test_data else []
            ),
            "layerSpecs": self.state.to_dict().get("layerSpecs", []),
            "pyCode": compile_network_to_py(self.network),
            "state": self.state.to_dict(),
        }

        if include_boundaries and self._output_boundary is not None:
            frame["outputBoundary"] = self._output_boundary.tolist()
            frame["nodeBoundaries"] = {
                nid: m.tolist() for nid, m in self._node_boundaries.items()
            }
            if self._expert_boundaries_small:
                for nid, m in self._expert_boundaries_small.items():
                    frame["nodeBoundaries"][nid] = m.tolist()

        if self.state.has_moe():
            moe_info = []
            for l in range(self.network.num_layers):
                spec = self.network._get_spec(l)
                if spec.get("type") != "moe":
                    continue
                D = len(spec["expert"]["layers"])
                expert_node_ids = [
                    [
                        [f"E:{l + 1}:{e}:{d}:{i}"
                         for i in range(spec["expert"]["layers"][d]["n"])]
                        for d in range(D)
                    ]
                    for e in range(spec["experts"])
                ]
                info = {
                    "layer": l + 1,
                    "numExperts": spec["experts"],
                    "topK": spec["top_k"],
                    "soft": spec["soft"],
                    "routerTemp": spec["router_temp"],
                    "auxCoef": spec["aux_coef"],
                    "expertDepth": D,
                    "expertNodeIds": expert_node_ids,
                }
                if include_boundaries and l in self._moe_probe:
                    p = self._moe_probe[l]
                    argmax = p["argmax"]
                    step = self.density // 50 if self.density >= 50 else 1
                    info["routingMap"] = argmax[::step, ::step].astype(int).tolist()

                    w = p["weights"]
                    info["routingWeights"] = w.tolist()
                moe_info.append(info)
            frame["moeInfo"] = moe_info

        return frame