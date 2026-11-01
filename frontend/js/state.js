export class State {
  constructor(init = {}) {
    this.learningRate = 0.03; this.regularizationRate = 0.01; this.noise = 0;
    this.batchSize = 10; this.percTrainData = 50; this.activation = "sigmoid";
    this.regularization = "none"; this.problem = "classification"; this.initZero = false;
    this.optimizer = "SGD"; this.onlineData = false;
    this.residuals = [];
    this.numSamples = 400;
    this.layerSpecs = [{ type: "dense", n: 4, act: "sigmoid", bias: true, norm: "none", norm_pos: "post" }];
    this.x = true; this.y = true; this.xTimesY = false;
    this.xSquared = false; this.ySquared = false;
    this.dataset = "circle"; this.regDataset = "reg-plane";
    this.seed = "0"; this.showTestData = false; this.discretize = false;
    this.gradClipMethod = "none";
    this.gradClipValue = 1.0;
    Object.assign(this, init);
    if (!Array.isArray(this.layerSpecs) || this.layerSpecs.length === 0) {
      this.layerSpecs = [{ type: "dense", n: 4, act: this.activation || "sigmoid", bias: true, norm: "none", norm_pos: "post" }];
    }
    if (!Array.isArray(this.residuals)) this.residuals = [];
    this.ensureSpecs();
  }

  // ---- derived read-only views (use LAST expert layer) ----
  get networkShape() {
    return this.layerSpecs.map(s =>
      s.type === "moe" ? s.expert.layers[s.expert.layers.length - 1].n : s.n
    );
  }
  get hiddenActivations() {
    return this.layerSpecs.map(s =>
      s.type === "moe" ? s.expert.layers[s.expert.layers.length - 1].act : s.act
    );
  }
  get layerBiases() {
    return this.layerSpecs.map(s =>
      s.type === "moe" ? s.expert.layers[s.expert.layers.length - 1].bias : s.bias
    );
  }
  get layerNorms() {
    return this.layerSpecs.map(s => s.type === "moe" ? "none" : s.norm);
  }
  get layerNormPositions() {
    return this.layerSpecs.map(s => s.type === "moe" ? "post" : s.norm_pos);
  }

  hasMoE() { return this.layerSpecs.some(s => s.type === "moe"); }

  moeLayerIndices() {
    return this.layerSpecs.map((s, i) => s.type === "moe" ? i : -1).filter(i => i >= 0);
  }

  ensureSpecs() {
    this.layerSpecs = this.layerSpecs.map(s => {
      if (s.type === "moe") {
        const E = Math.max(2, Math.min(16, parseInt(s.experts ?? 4, 10)));
        const soft = !!s.soft;
        const k = soft ? E : Math.max(2, Math.min(E, parseInt(s.top_k ?? 2, 10)));
        let rawLayers;
        if (s.expert && Array.isArray(s.expert.layers) && s.expert.layers.length > 0) {
          rawLayers = s.expert.layers;
        } else if (s.expert && typeof s.expert === "object") {
          rawLayers = [{ n: s.expert.n ?? 4, act: s.expert.act || "relu", bias: s.expert.bias !== false }];
        } else {
          rawLayers = [{ n: 4, act: "relu", bias: true }];
        }
        const layers = rawLayers.slice(0, 8).map(ls => ({
          n: Math.max(1, Math.min(32, parseInt(ls.n ?? 4, 10))),
          act: ls.act || "relu",
          bias: ls.bias !== false,
        }));
        return {
          type: "moe", experts: E, top_k: k, soft,
          router_bias: s.router_bias !== false,
          router_temp: Math.max(0.01, Math.min(10, parseFloat(s.router_temp ?? 1.0))),
          normalize: s.normalize !== false,
          aux_coef: Math.max(0, Math.min(1, parseFloat(s.aux_coef ?? 0.01))),
          expert: { layers },
        };
      }
      return {
        type: "dense",
        n: Math.max(1, Math.min(32, parseInt(s.n ?? 4, 10))),
        act: s.act || "sigmoid",
        bias: s.bias !== false,
        norm: ["none", "layernorm", "rmsnorm"].includes(s.norm) ? s.norm : "none",
        norm_pos: ["pre", "post"].includes(s.norm_pos) ? s.norm_pos : "post",
      };
    });
  }

  static fromDict(dict) { return new State(dict); }

  static fromHash() {
    const h = location.hash.slice(1);
    if (!h) return null;
    const map = {};
    for (const kv of h.split("&")) {
      const eq = kv.indexOf("="); if (eq < 0) continue;
      map[decodeURIComponent(kv.slice(0, eq))] = decodeURIComponent(kv.slice(eq + 1));
    }
    const ints = ["batchSize", "noise", "percTrainData", "numSamples"];
    const floats = ["learningRate", "regularizationRate", "gradClipValue"];
    const bools = ["x", "y", "xTimesY", "xSquared", "ySquared", "initZero",
                   "onlineData", "showTestData", "discretize"];
    for (const k of ints) if (k in map) map[k] = parseInt(map[k], 10);
    for (const k of floats) if (k in map) map[k] = parseFloat(map[k]);
    for (const k of bools) if (k in map) map[k] = map[k] === "true";

    if ("layerSpecs" in map) {
      try { map.layerSpecs = JSON.parse(map.layerSpecs); }
      catch (e) { delete map.layerSpecs; }
    }
    if (!map.layerSpecs && "networkShape" in map) {
      const shape = map.networkShape ? map.networkShape.split(",").map(Number) : [4];
      const acts = map.hiddenActivations ? map.hiddenActivations.split(",") : [];
      const biases = map.layerBiases ? map.layerBiases.split(",").map(v => v === "true") : [];
      const norms = map.layerNorms ? map.layerNorms.split(",") : [];
      const normPos = map.layerNormPositions ? map.layerNormPositions.split(",") : [];
      const defAct = map.activation || "sigmoid";
      map.layerSpecs = shape.map((n, i) => ({
        type: "dense", n,
        act: acts[i] || defAct,
        bias: biases[i] !== false,
        norm: norms[i] || "none",
        norm_pos: normPos[i] || "post",
      }));
      delete map.networkShape;
      delete map.hiddenActivations;
      delete map.layerBiases;
      delete map.layerNorms;
      delete map.layerNormPositions;
    }
    if ("residuals" in map) {
      try { map.residuals = JSON.parse(decodeURIComponent(map.residuals)); }
      catch (e) { map.residuals = []; }
    }
    return new State(map);
  }

  toHash() {
    const parts = [];
    for (const [k, v] of Object.entries(this)) {
      if (v === null || v === undefined) continue;
      let s;
      if (k === "layerSpecs" || k === "residuals") s = JSON.stringify(v);
      else if (Array.isArray(v)) s = v.join(",");
      else s = String(v);
      parts.push(`${k}=${encodeURIComponent(s)}`);
    }
    return parts.join("&");
  }

  writeHash() {
    const next = "#" + this.toHash();
    if (location.hash !== next) history.replaceState(null, "", next);
  }

  toDictWire() {
    const out = {};
    for (const [k, v] of Object.entries(this)) out[k] = v;
    return out;
  }
}