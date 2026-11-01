import { Api } from "./api.js";
import { State } from "./state.js";
import { HeatMap } from "./charts/heatmap.js";
import { NetworkView, INPUTS } from "./charts/network.js";
import { DatasetThumbs } from "./ui/dataset_thumbs.js";
import { classColor, colorFor, fmtLoss, fmtInt } from "./utils.js";
import { LineChart } from "./charts/linechart.js";

const api = new Api();
const heatmap = new HeatMap(document.getElementById("heatmap"), 260, 100);
const networkView = new NetworkView(document.getElementById("network"));
const linechart = new LineChart(document.getElementById("linechart"));

const zoomLayer = document.getElementById("node-zoom-layer");
const classLegend = document.getElementById("class-legend");

const SVG_PLAY = `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" style="display: block;"><path d="M8 5v14l11-7z"/></svg>`;
const SVG_PAUSE = `<svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" style="display: block;"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>`;

const MAX_HIDDEN_LAYERS = 12;
const MAX_NODES = 32;
const MAX_EXPERT_DEPTH = 8;

const datasetThumbs = new DatasetThumbs(
  document.getElementById("synthetic-datasets"),
  document.getElementById("real-datasets"),
  (name, category) => {
    if (category === "regression") sendSet("regDataset", name);
    else sendSet("dataset", name);
  }
);

// ---- tutorial link: carry current state via URL hash ----
const tutorialLink = document.getElementById("tutorial-link");
if (tutorialLink) {
  tutorialLink.addEventListener("click", (e) => {
    if (location.hash.length > 1) {
      e.preventDefault();
      location.href = "/tutorial.html" + location.hash;
    }
  });
}

// ---- drawers ----
const toggleConfigBtn = document.getElementById("toggle-config-btn");
const closeConfigBtn = document.getElementById("close-config-btn");
const configDrawer = document.getElementById("config-drawer");
if (toggleConfigBtn && configDrawer) {
  toggleConfigBtn.addEventListener("click", () => configDrawer.classList.toggle("open"));
}
if (closeConfigBtn && configDrawer) {
  closeConfigBtn.addEventListener("click", () => configDrawer.classList.remove("open"));
}
const toggleDataBtn = document.getElementById("toggle-data-btn");
const closeDataBtn = document.getElementById("close-data-btn");
const dataDrawer = document.getElementById("data-drawer");
if (toggleDataBtn && dataDrawer) {
  toggleDataBtn.addEventListener("click", () => dataDrawer.classList.toggle("open"));
}
if (closeDataBtn && dataDrawer) {
  closeDataBtn.addEventListener("click", () => dataDrawer.classList.remove("open"));
}

let state = State.fromHash() || new State();
let sentInitialState = false, currentFrame = null, lastNetworkSignature = "";
let residualUIEnabled = false;
let pendingCustomDatasetCheck = false;

// multi-layer zoom pending aggregation
const pendingKeys = new Set();
const activeInputs = new Set();
const PENDING_TTL_MS = 1500;

const els = {
  playPause: document.getElementById("play-pause-button"), reset: document.getElementById("reset-button"),
  step: document.getElementById("next-step-button"), iter: document.getElementById("iter-number"),
  lossTrain: document.getElementById("loss-train"), lossTest: document.getElementById("loss-test"),
  learningRate: document.getElementById("learningRate"), activations: document.getElementById("activations"),
  regularizations: document.getElementById("regularizations"), regularRate: document.getElementById("regularRate"),
  problem: document.getElementById("problem"),
  optimizer: document.getElementById("optimizer"),
  gradClipMethod: document.getElementById("gradClipMethod"),
  gradClipValue: document.getElementById("gradClipValue"),
  percTrainData: document.getElementById("percTrainData"), noise: document.getElementById("noise"),
  batchSize: document.getElementById("batchSize"), onlineData: document.getElementById("onlineData"),
  numSamples: document.getElementById("numSamples"),
  useResidual: document.getElementById("useResidual"), showTestData: document.getElementById("show-test-data"),
  discretize: document.getElementById("discretize"), regen: document.getElementById("data-regen-button"),
  pyCode: document.getElementById("network-as-python"), connStatus: document.getElementById("conn-status"),
  copyPyBtn: document.getElementById("copy-py-btn"),
  resControls: document.getElementById("residual-controls"), resPanel: document.getElementById("residual-panel"),
  showResPanel: document.getElementById("show-res-panel"), resSrc: document.getElementById("res-src"),
  resDst: document.getElementById("res-dst"), btnAddRes: document.getElementById("btn-add-res"),
  btnCancelRes: document.getElementById("btn-cancel-res"), residualList: document.getElementById("residual-list"),
  perLayerConfig: document.getElementById("per-layer-config"),
  addLayerBtn: document.getElementById("add-layer-btn"),
  addMoeBtn: document.getElementById("add-moe-btn"),
  expertMapsContainer: document.getElementById("expert-maps-container"),
};

const ACTIVATION_OPTIONS = [
  ["linear", "Linear"], ["sigmoid", "Sigmoid"], ["tanh", "Tanh"],
  ["relu", "ReLU"], ["leakyrelu", "Leaky ReLU"], ["elu", "ELU"],
  ["selu", "SELU"], ["swish", "Swish"], ["gelu", "GELU"],
  ["softplus", "Softplus"], ["mish", "Mish"], ["sine", "Sine"], ["arctan", "ArcTan"],
];
const NORM_OPTIONS = [["none", "None"], ["layernorm", "LayerNorm"], ["rmsnorm", "RMSNorm"]];
const NORM_POSITION_OPTIONS = [["pre", "Pre"], ["post", "Post"]];

function shallowEqual(a, b) {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!shallowEqual(a[i], b[i])) return false;
    return true;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a), kb = Object.keys(b);
    if (ka.length !== kb.length) return false;
    for (const k of ka) if (!shallowEqual(a[k], b[k])) return false;
    return true;
  }
  return false;
}

function setValueIfIdle(el, key, value) {
  if (!el || activeInputs.has(key)) return;
  const str = String(value);
  if (el.value !== str) el.value = str;
}
function setCheckedIfIdle(el, key, checked) {
  if (!el || activeInputs.has(key)) return;
  if (el.checked !== !!checked) el.checked = !!checked;
}
function setLabelIfIdle(key, sel, value) {
  if (activeInputs.has(key)) return;
  const el = document.querySelector(sel);
  if (el && el.textContent !== String(value)) el.textContent = String(value);
}
function isPlaying() { return els.playPause.classList.contains("playing"); }

// "Discretize output" is only meaningful for continuous outputs.
// Hide it when the network produces a class index (multiclass).
function updateDiscretizeVisibility(isMulticlass) {
  const label = els.discretize
    ? els.discretize.closest(".ui-discretize")
    : null;
  if (!label) return;
  label.style.display = isMulticlass ? "none" : "";
}

// initial sync — URL hash may already be multiclass before the first frame
updateDiscretizeVisibility(state.problem === "multiclass");

function applyStateToUI(s) {
  setValueIfIdle(els.learningRate, "learningRate", s.learningRate);
  setValueIfIdle(els.activations, "activation", s.activation);
  setValueIfIdle(els.regularizations, "regularization", s.regularization);
  setValueIfIdle(els.regularRate, "regularizationRate", s.regularizationRate);
  setValueIfIdle(els.problem, "problem", s.problem);
  setValueIfIdle(els.optimizer, "optimizer", s.optimizer);
    setValueIfIdle(els.gradClipMethod, "gradClipMethod", s.gradClipMethod);

  if (els.gradClipValue && !activeInputs.has("gradClipValue")) {
    const targetVal = parseFloat(s.gradClipValue);

    const options = Array.from(els.gradClipValue.options);
    const matched = options.find(o => parseFloat(o.value) === targetVal);
    if (matched) {
      els.gradClipValue.value = matched.value;
    }
    els.gradClipValue.style.display = (s.gradClipMethod === "none") ? "none" : "inline-block";
  }
  setValueIfIdle(els.numSamples, "numSamples", s.numSamples);
  setValueIfIdle(els.percTrainData, "percTrainData", s.percTrainData);
  setValueIfIdle(els.noise, "noise", s.noise);
  setValueIfIdle(els.batchSize, "batchSize", s.batchSize);
  setCheckedIfIdle(els.onlineData, "onlineData", s.onlineData);
  setCheckedIfIdle(els.showTestData, "showTestData", s.showTestData);
  setCheckedIfIdle(els.discretize, "discretize", s.discretize);
  setLabelIfIdle("numSamples", 'label[for="numSamples"] .value', s.numSamples);
  setLabelIfIdle("percTrainData", 'label[for="percTrainData"] .value', s.percTrainData);
  setLabelIfIdle("noise", 'label[for="noise"] .value', s.noise);
  setLabelIfIdle("batchSize", 'label[for="batchSize"] .value', s.batchSize);

  const residuals = s.residuals || [];
  if (residuals.length > 0) residualUIEnabled = true;
  if (els.useResidual.checked !== residualUIEnabled) els.useResidual.checked = residualUIEnabled;
  els.resControls.style.display = residualUIEnabled ? "block" : "none";

  renderResidualList(residuals, s.networkShape);
  renderPerLayerConfig(s);
}

function renderResidualList(residuals, networkShape) {
  const list = els.residualList;
  if (!list) return;
  const empty = !residuals || residuals.length === 0;
  if (empty) {
    if (!residualUIEnabled) { if (list.childElementCount !== 0) list.innerHTML = ""; return; }
    if (list.childElementCount === 1 && list.firstElementChild.classList.contains("residual-empty")) return;
    list.innerHTML = "";
    list.dataset.sig = "";
    const hint = document.createElement("div");
    hint.className = "residual-empty";
    hint.textContent = "No skip connections yet — click “+ Add Skip Connection” above.";
    list.appendChild(hint);
    return;
  }
  const sig = JSON.stringify(residuals) + "|" + JSON.stringify(networkShape);
  if (list.dataset.sig === sig) return;
  list.dataset.sig = sig;
  list.innerHTML = "";
  const numHidden = networkShape.length;
  residuals.forEach(([src, dst]) => {
    const item = document.createElement("div");
    item.className = "residual-item";
    const span = document.createElement("span");
    const srcLabel = src === 0 ? "Input" : `Layer ${src}`;
    const dstLabel = dst === numHidden + 1 ? "Output" : `Layer ${dst}`;
    span.textContent = `${srcLabel} → ${dstLabel}`;
    const btn = document.createElement("button");
    btn.className = "residual-del";
    btn.title = "Remove this skip connection";
    btn.textContent = "×";
    btn.addEventListener("click", () => {
      const newRess = (state.residuals || []).filter(r => !(r[0] === src && r[1] === dst));
      sendLayerSpecs(state.layerSpecs, { residuals: newRess });
      linechart.reset();
    });
    item.appendChild(span); item.appendChild(btn);
    list.appendChild(item);
  });
}

// ---- per-layer config ----

function _deleteLayer(idx) {
  const specs = [...state.layerSpecs];
  specs.splice(idx, 1);
  if (specs.length === 0) {
    specs.push({ type: "dense", n: 4, act: state.activation, bias: true, norm: "none", norm_pos: "post" });
  }
  const residuals = (state.residuals || [])
    .filter(([s, d]) => s !== idx && d !== idx)
    .map(([s, d]) => [s > idx ? s - 1 : s, d > idx ? d - 1 : d]);
  sendLayerSpecs(specs, { residuals });
  linechart.reset();
  hideZoom();
}

function _patchSpec(idx, patch) {
  const specs = state.layerSpecs.map((s, i) => i === idx ? { ...s, ...patch } : s);
  sendLayerSpecs(specs);
  linechart.reset();
}

function _patchExpertLayer(idx, layerIdx, patch) {
  const specs = state.layerSpecs.map((s, i) => {
    if (i !== idx) return s;
    const layers = s.expert.layers.map((ls, j) => j === layerIdx ? { ...ls, ...patch } : ls);
    return { ...s, expert: { ...s.expert, layers } };
  });
  sendLayerSpecs(specs);
  linechart.reset();
}

function _addExpertLayer(idx) {
  const specs = state.layerSpecs.map((s, i) => {
    if (i !== idx) return s;
    if (s.expert.layers.length >= MAX_EXPERT_DEPTH) return s;
    const last = s.expert.layers[s.expert.layers.length - 1];
    const layers = [...s.expert.layers, {
      n: Math.max(1, Math.min(MAX_NODES, last.n)),
      act: last.act, bias: last.bias,
    }];
    return { ...s, expert: { ...s.expert, layers } };
  });
  sendLayerSpecs(specs);
  linechart.reset();
}

function _removeExpertLayer(idx, layerIdx) {
  const specs = state.layerSpecs.map((s, i) => {
    if (i !== idx) return s;
    if (s.expert.layers.length <= 1) return s;
    const layers = s.expert.layers.filter((_, j) => j !== layerIdx);
    return { ...s, expert: { ...s.expert, layers } };
  });
  sendLayerSpecs(specs);
  linechart.reset();
}

function renderPerLayerConfig(s) {
  const container = els.perLayerConfig;
  if (!container) return;
  const sig = JSON.stringify(s.layerSpecs);
  if (container.dataset.sig === sig) return;
  container.dataset.sig = sig;
  container.innerHTML = "";

  s.layerSpecs.forEach((spec, idx) => {
    const row = document.createElement("div");
    row.className = `per-layer-block per-layer-${spec.type}`;

    // header row: label + delete
    const header = document.createElement("div");
    header.className = "per-layer-header";
    const title = document.createElement("span");
    title.className = "per-layer-title";
    title.textContent = spec.type === "moe"
      ? `L${idx + 1} · MoE (D=${spec.expert.layers.length})`
      : `L${idx + 1} · ${spec.n}n`;
    header.appendChild(title);
    const del = document.createElement("button");
    del.className = "layer-delete-btn";
    del.title = "Delete this layer";
    del.textContent = "×";
    del.addEventListener("click", () => _deleteLayer(idx));
    header.appendChild(del);
    row.appendChild(header);

    if (spec.type === "dense") {
      const grid = document.createElement("div");
      grid.className = "per-layer-grid";

      const nLabel = document.createElement("label");
      nLabel.textContent = "N";
      const nInput = document.createElement("input");
      nInput.type = "number"; nInput.min = "1"; nInput.max = String(MAX_NODES); nInput.value = spec.n;
      nInput.addEventListener("change", e => _patchSpec(idx, { n: parseInt(e.target.value, 10) }));
      grid.appendChild(nLabel); grid.appendChild(nInput);

      const actLabel = document.createElement("label");
      actLabel.textContent = "Act";
      const actSel = makeSelect(ACTIVATION_OPTIONS, spec.act, v => _patchSpec(idx, { act: v }));
      grid.appendChild(actLabel); grid.appendChild(actSel);

      const normLabel = document.createElement("label");
      normLabel.textContent = "Norm";
      const normSel = makeSelect(NORM_OPTIONS, spec.norm, v => _patchSpec(idx, { norm: v }));
      grid.appendChild(normLabel); grid.appendChild(normSel);

      const posLabel = document.createElement("label");
      posLabel.textContent = "Pos";
      const posSel = makeSelect(NORM_POSITION_OPTIONS, spec.norm_pos, v => _patchSpec(idx, { norm_pos: v }));
      posSel.disabled = spec.norm === "none";
      grid.appendChild(posLabel); grid.appendChild(posSel);

      const biasLabel = document.createElement("label");
      biasLabel.className = "checkbox-cell";
      const biasCb = document.createElement("input");
      biasCb.type = "checkbox"; biasCb.checked = spec.bias;
      biasCb.addEventListener("change", e => _patchSpec(idx, { bias: e.target.checked }));
      biasLabel.appendChild(biasCb); biasLabel.appendChild(document.createTextNode("Bias"));
      grid.appendChild(biasLabel); grid.appendChild(document.createElement("span"));

      row.appendChild(grid);
    } else {
      // MoE block
      const grid = document.createElement("div");
      grid.className = "per-layer-grid";

      const softLabel = document.createElement("label");
      softLabel.textContent = "Type";
      const softSel = makeSelect(
        [["false", "Top-k"], ["true", "Soft"]], String(spec.soft),
        v => _patchSpec(idx, { soft: v === "true" })
      );
      grid.appendChild(softLabel); grid.appendChild(softSel);

      const eLabel = document.createElement("label");
      eLabel.textContent = "Experts";
      const eInput = document.createElement("input");
      eInput.type = "number"; eInput.min = "2"; eInput.max = "16"; eInput.value = spec.experts;
      eInput.addEventListener("change", e => _patchSpec(idx, { experts: parseInt(e.target.value, 10) }));
      grid.appendChild(eLabel); grid.appendChild(eInput);

      const kLabel = document.createElement("label");
      kLabel.textContent = "Top-k";
      const kInput = document.createElement("input");
      kInput.type = "number"; kInput.min = "2"; kInput.max = String(spec.experts); kInput.value = spec.top_k;
      kInput.disabled = spec.soft;
      kInput.addEventListener("change", e => _patchSpec(idx, { top_k: parseInt(e.target.value, 10) }));
      grid.appendChild(kLabel); grid.appendChild(kInput);

      const tempLabel = document.createElement("label");
      tempLabel.textContent = "Temp";
      const tempInput = document.createElement("input");
      tempInput.type = "number"; tempInput.min = "0.01"; tempInput.max = "10"; tempInput.step = "0.1";
      tempInput.value = spec.router_temp;
      tempInput.addEventListener("change", e => _patchSpec(idx, { router_temp: parseFloat(e.target.value) }));
      grid.appendChild(tempLabel); grid.appendChild(tempInput);

      const auxLabel = document.createElement("label");
      auxLabel.textContent = "Aux";
      const auxInput = document.createElement("input");
      auxInput.type = "number"; auxInput.min = "0"; auxInput.max = "1"; auxInput.step = "0.01";
      auxInput.value = spec.aux_coef;
      auxInput.addEventListener("change", e => _patchSpec(idx, { aux_coef: parseFloat(e.target.value) }));
      grid.appendChild(auxLabel); grid.appendChild(auxInput);

      const rbLabel = document.createElement("label");
      rbLabel.className = "checkbox-cell";
      const rbCb = document.createElement("input");
      rbCb.type = "checkbox"; rbCb.checked = spec.router_bias;
      rbCb.addEventListener("change", e => _patchSpec(idx, { router_bias: e.target.checked }));
      rbLabel.appendChild(rbCb); rbLabel.appendChild(document.createTextNode("Router bias"));
      grid.appendChild(rbLabel);

      const normLabel2 = document.createElement("label");
      normLabel2.className = "checkbox-cell";
      const normCb = document.createElement("input");
      normCb.type = "checkbox"; normCb.checked = spec.normalize;
      normCb.disabled = spec.soft;
      normCb.addEventListener("change", e => _patchSpec(idx, { normalize: e.target.checked }));
      normLabel2.appendChild(normCb); normLabel2.appendChild(document.createTextNode("Normalize"));
      grid.appendChild(normLabel2);

      row.appendChild(grid);

      // Expert layers (shared structure)
      const expHeader = document.createElement("div");
      expHeader.className = "per-layer-subheader";
      expHeader.textContent = `Expert depth: ${spec.expert.layers.length} (shared structure)`;
      const addExpBtn = document.createElement("button");
      addExpBtn.className = "add-btn";
      addExpBtn.style.marginLeft = "8px";
      addExpBtn.style.padding = "2px 8px";
      addExpBtn.textContent = "+ layer";
      addExpBtn.disabled = spec.expert.layers.length >= MAX_EXPERT_DEPTH;
      addExpBtn.style.opacity = addExpBtn.disabled ? "0.4" : "1";
      addExpBtn.addEventListener("click", () => _addExpertLayer(idx));
      expHeader.appendChild(addExpBtn);
      row.appendChild(expHeader);

      const expList = document.createElement("div");
      expList.className = "expert-layer-list";
      spec.expert.layers.forEach((ls, li) => {
        const liRow = document.createElement("div");
        liRow.className = "expert-layer-row";

        const tag = document.createElement("span");
        tag.className = "expert-layer-tag";
        tag.textContent = `D${li + 1}`;
        liRow.appendChild(tag);

        const nLbl = document.createElement("label"); nLbl.textContent = "N";
        const nIn = document.createElement("input");
        nIn.type = "number"; nIn.min = "1"; nIn.max = String(MAX_NODES); nIn.value = ls.n;
        nIn.addEventListener("change", e => _patchExpertLayer(idx, li, { n: parseInt(e.target.value, 10) }));
        liRow.appendChild(nLbl); liRow.appendChild(nIn);

        const aLbl = document.createElement("label"); aLbl.textContent = "Act";
        const aSel = makeSelect(ACTIVATION_OPTIONS, ls.act, v => _patchExpertLayer(idx, li, { act: v }));
        liRow.appendChild(aLbl); liRow.appendChild(aSel);

        const bLbl = document.createElement("label"); bLbl.className = "checkbox-cell";
        const bCb = document.createElement("input");
        bCb.type = "checkbox"; bCb.checked = ls.bias;
        bCb.addEventListener("change", e => _patchExpertLayer(idx, li, { bias: e.target.checked }));
        bLbl.appendChild(bCb); bLbl.appendChild(document.createTextNode("Bias"));
        liRow.appendChild(bLbl);

        const rmBtn = document.createElement("button");
        rmBtn.className = "layer-delete-btn";
        rmBtn.textContent = "×";
        rmBtn.title = "Remove this expert layer";
        rmBtn.disabled = spec.expert.layers.length <= 1;
        rmBtn.style.opacity = rmBtn.disabled ? "0.4" : "1";
        rmBtn.addEventListener("click", () => _removeExpertLayer(idx, li));
        liRow.appendChild(rmBtn);

        expList.appendChild(liRow);
      });
      row.appendChild(expList);
    }

    container.appendChild(row);
  });
}

function makeSelect(options, value, onChange) {
  const sel = document.createElement("select");
  for (const [v, label] of options) {
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = label;
    sel.appendChild(opt);
  }
  sel.value = value;
  sel.addEventListener("change", e => onChange(e.target.value));
  return sel;
}

// ---- sending state ----

function sendSet(key, value) {
  state[key] = value;
  state.writeHash();
  pendingKeys.add(key);
  setTimeout(() => pendingKeys.delete(key), PENDING_TTL_MS);
  api.send({ cmd: "set", key, value });
}

function sendLayerSpecs(specs, extras = {}) {
  state.layerSpecs = specs;
  state.ensureSpecs();
  if (extras.residuals) state.residuals = extras.residuals;
  state.writeHash();
  pendingKeys.add("layerSpecs");
  setTimeout(() => pendingKeys.delete("layerSpecs"), PENDING_TTL_MS);
  api.send({ cmd: "set", key: "layerSpecs", value: state.layerSpecs });
  if (extras.residuals) {
    api.send({ cmd: "set", key: "residuals", value: state.residuals });
  }
}

// ---- zoom layer ----

function hideZoom() { if (zoomLayer) zoomLayer.innerHTML = ""; }

function _fmtZoomTitle(nodeId) {
  if (nodeId.startsWith("E:")) {
    const parts = nodeId.split(":");
    if (parts.length === 5) {
      const [, l, e, d, i] = parts;
      return `L${l} · Expert ${Number(e) + 1} · D${Number(d) + 1} · N${Number(i) + 1}`;
    }
    if (parts.length === 4) {
      const [, l, e, i] = parts;
      return `L${l} · Expert ${Number(e) + 1} · N${Number(i) + 1}`;
    }
  }
  return `Node ${nodeId}`;
}

function showZoom(nodeId, boundary, titleOverride) {
  if (!zoomLayer || !boundary) return;
  const coord = networkView.node2coord[nodeId];
  if (!coord) return;

  const SIZE = 200;
  const boxW = SIZE + 20;
  const boxH = SIZE + 44;

  zoomLayer.innerHTML = "";
  const box = document.createElement("div");
  box.className = "node-zoom";

  const title = document.createElement("div");
  title.className = "node-zoom-title";
  title.textContent = titleOverride || _fmtZoomTitle(nodeId);
  box.appendChild(title);

  const H = boundary.length;
  const W = boundary[0].length;

  const canvas = document.createElement("canvas");
  canvas.width = W; canvas.height = H;
  canvas.style.width = `${SIZE}px`; canvas.style.height = `${SIZE}px`;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(W, H);

  const isRouter = nodeId.startsWith("moe_router");
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {

      const v = isRouter ? boundary[j][i] : boundary[i][j];
      let rgb;
      if (isRouter) {

        rgb = [
          Math.round(240 - v * 106),
          Math.round(242 - v * 202),
          Math.round(244 - v * 22)
        ];
      } else {
        rgb = colorFor(v);
      }
      const idx = (j * W + i) * 4;
      img.data[idx] = rgb[0]; img.data[idx + 1] = rgb[1]; img.data[idx + 2] = rgb[2]; img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  box.appendChild(canvas);

  const closeBtn = document.createElement("button");
  closeBtn.className = "node-zoom-close";
  closeBtn.textContent = "×";
  closeBtn.addEventListener("click", hideZoom);
  box.appendChild(closeBtn);

  zoomLayer.appendChild(box);

  const Wc = networkView.container.clientWidth;
  const Hc = networkView.container.clientHeight;
  const rect = networkView.rectSize;

  let left = coord.cx + rect / 2 + 12;
  if (left + boxW > Wc) left = coord.cx - rect / 2 - 12 - boxW;
  left = Math.max(4, left);
  let top = coord.cy - boxH / 2;
  top = Math.max(4, Math.min(Hc - boxH - 4, top));
  box.style.left = `${left}px`;
  box.style.top = `${top}px`;
}

networkView.onNodeClick = (nodeId, directBoundary) => {
  if (isPlaying()) return;
  if (directBoundary) {

    const parts = nodeId.split("_");
    const layerIdx = parseInt(parts[2], 10);
    const eIdx = parseInt(parts[3], 10);
    showZoom(nodeId, directBoundary, `L${layerIdx} · Router R${eIdx + 1}`);
  } else {
    api.send({ cmd: "request_node_boundary", node_id: nodeId });
  }
};

networkView.onChangeExpertSelection = (layerIdx, expertIdx) => {
  networkView.setSelectedExpert(layerIdx, expertIdx);
  if (currentFrame) {
    networkView.build(currentFrame, state);
    networkView.update(currentFrame);
    lastNetworkSignature = _networkSignature(currentFrame);
    renderExpertMaps(currentFrame);
  }
};

// ---- play/pause ----

els.playPause.addEventListener("click", () => {
  if (isPlaying()) {
    api.send({ cmd: "pause" });
    els.playPause.classList.remove("playing");
    els.playPause.innerHTML = SVG_PLAY;
  } else {
    api.send({ cmd: "play", fps: 30 });
    els.playPause.classList.add("playing");
    els.playPause.innerHTML = SVG_PAUSE;
    hideZoom();
  }
});

els.reset.addEventListener("click", () => {
  api.send({ cmd: "reset" });
  els.playPause.classList.remove("playing");
  els.playPause.innerHTML = SVG_PLAY;
  linechart.reset();
  hideZoom();
});

els.step.addEventListener("click", () => {
  api.send({ cmd: "pause" });
  els.playPause.classList.remove("playing");
  els.playPause.innerHTML = SVG_PLAY;
  api.send({ cmd: "step", n: 1 });
  hideZoom();
});

els.learningRate.addEventListener("change", e => sendSet("learningRate", parseFloat(e.target.value)));
els.activations.addEventListener("change", e => sendSet("activation", e.target.value));
els.regularizations.addEventListener("change", e => sendSet("regularization", e.target.value));
els.regularRate.addEventListener("change", e => sendSet("regularizationRate", parseFloat(e.target.value)));
els.problem.addEventListener("change", e => {
  hideZoom();
  sendSet("problem", e.target.value);
  updateDiscretizeVisibility(e.target.value === "multiclass");
});

els.optimizer.addEventListener("change", e => {
  const opt = e.target.value; sendSet("optimizer", opt);
  let newLr = opt === "Momentum" ? 0.01 : (opt === "RMSProp" || opt === "Adam" ? 0.003 : 0.03);
  if (opt !== "SGD") { sendSet("learningRate", newLr); els.learningRate.value = String(newLr); }
});

els.gradClipMethod.addEventListener("change", e => {
  const method = e.target.value;
  sendSet("gradClipMethod", method);
  if (els.gradClipValue) {
    els.gradClipValue.style.display = (method === "none") ? "none" : "inline-block";
  }
});

els.gradClipValue.addEventListener("change", e => {
  sendSet("gradClipValue", parseFloat(e.target.value));
});

function bindRange(el, key, labelSel, parse) {
  el.addEventListener("input", e => {
    activeInputs.add(key);
    if (labelSel) {
      const lbl = document.querySelector(labelSel);
      if (lbl && lbl.textContent !== e.target.value) lbl.textContent = e.target.value;
    }
  });
  el.addEventListener("change", e => {
    activeInputs.delete(key);
    sendSet(key, parse(e.target.value));
    hideZoom();
  });
  el.addEventListener("blur", () => activeInputs.delete(key));
}

bindRange(els.numSamples, "numSamples", 'label[for="numSamples"] .value', v => parseInt(v, 10));
bindRange(els.percTrainData, "percTrainData", 'label[for="percTrainData"] .value', v => parseInt(v, 10));
bindRange(els.noise, "noise", 'label[for="noise"] .value', v => parseInt(v, 10));
bindRange(els.batchSize, "batchSize", 'label[for="batchSize"] .value', v => parseInt(v, 10));

els.onlineData.addEventListener("change", e => sendSet("onlineData", e.target.checked));
els.showTestData.addEventListener("change", e => sendSet("showTestData", e.target.checked));
els.discretize.addEventListener("change", e => sendSet("discretize", e.target.checked));
els.regen.addEventListener("click", () => { api.send({ cmd: "regenerate" }); linechart.reset(); hideZoom(); });

// ---- add layer / MoE block ----

if (els.addLayerBtn) els.addLayerBtn.addEventListener("click", () => {
  if (state.layerSpecs.length >= MAX_HIDDEN_LAYERS) return;
  const specs = [...state.layerSpecs, {
    type: "dense", n: 4, act: state.activation, bias: true, norm: "none", norm_pos: "post",
  }];
  sendLayerSpecs(specs);
  linechart.reset();
  hideZoom();
});

if (els.addMoeBtn) els.addMoeBtn.addEventListener("click", () => {
  if (state.layerSpecs.length >= MAX_HIDDEN_LAYERS) return;
  const specs = [...state.layerSpecs, {
    type: "moe", experts: 4, top_k: 2, soft: false,
    router_bias: true, router_temp: 1.0, normalize: true, aux_coef: 0.01,
    expert: { layers: [{ n: 4, act: "relu", bias: true }] },
  }];
  sendLayerSpecs(specs);
  linechart.reset();
  hideZoom();
});

// ---- residual UI ----

els.useResidual.addEventListener("change", e => {
  residualUIEnabled = e.target.checked;
  els.resControls.style.display = residualUIEnabled ? "block" : "none";
  if (!residualUIEnabled) {
    sendLayerSpecs(state.layerSpecs, { residuals: [] });
    els.resPanel.style.display = "none";
    linechart.reset();
  } else if ((state.residuals || []).length === 0) {
    els.showResPanel.click();
  }
});

function getLayerDim(idx, s, inputCount) {
  const numHidden = s.layerSpecs.length;
  if (idx === 0) return inputCount;
  if (idx === numHidden + 1) return 1;
  return s.networkShape[idx - 1];
}

els.showResPanel.addEventListener("click", () => {
  const s = state;
  const numHidden = s.layerSpecs.length;
  const inputCount = Object.keys(INPUTS).filter(id => s[id]).length;
  els.resSrc.innerHTML = ""; els.resDst.innerHTML = "";
  for (let i = 0; i <= numHidden; i++) {
    const opt = document.createElement("option");
    opt.value = i; opt.textContent = i === 0 ? "Input" : `Layer ${i}`;
    els.resSrc.appendChild(opt);
  }
  for (let i = 1; i <= numHidden + 1; i++) {
    const opt = document.createElement("option");
    opt.value = i; opt.textContent = i === numHidden + 1 ? "Output" : `Layer ${i}`;
    els.resDst.appendChild(opt);
  }
  const refreshDst = () => {
    const src = parseInt(els.resSrc.value, 10);
    const srcDim = getLayerDim(src, s, inputCount);
    let firstValid = null;
    Array.from(els.resDst.options).forEach(opt => {
      const dst = parseInt(opt.value, 10);
      const ok = dst > src && getLayerDim(dst, s, inputCount) === srcDim;
      opt.disabled = !ok;
      if (ok && firstValid === null) firstValid = opt.value;
    });
    if (firstValid !== null) els.resDst.value = firstValid;
    els.btnAddRes.disabled = firstValid === null;
    els.btnAddRes.style.opacity = firstValid === null ? "0.5" : "1";
    els.btnAddRes.style.cursor = firstValid === null ? "not-allowed" : "pointer";
  };
  els.resSrc.onchange = refreshDst;
  if (numHidden >= 1) {
    let picked = false;
    for (let i = 0; i <= numHidden; i++) {
      const dim = getLayerDim(i, s, inputCount);
      for (let j = i + 1; j <= numHidden + 1; j++) {
        if (getLayerDim(j, s, inputCount) === dim) { els.resSrc.value = String(i); picked = true; break; }
      }
      if (picked) break;
    }
  }
  refreshDst();
  els.resPanel.style.display = "block";
});

els.btnCancelRes.addEventListener("click", () => { els.resPanel.style.display = "none"; });

els.btnAddRes.addEventListener("click", () => {
  const src = parseInt(els.resSrc.value, 10);
  const dst = parseInt(els.resDst.value, 10);
  const s = state;
  const inputCount = Object.keys(INPUTS).filter(id => s[id]).length;
  if (src >= dst) { alert("Source layer must be before destination layer."); return; }
  if (getLayerDim(src, s, inputCount) !== getLayerDim(dst, s, inputCount)) {
    alert(`Dimension mismatch: ${getLayerDim(src, s, inputCount)} vs ${getLayerDim(dst, s, inputCount)} neurons.`);
    return;
  }
  if ((s.residuals || []).some(r => r[0] === src && r[1] === dst)) {
    alert("This skip connection already exists.");
    return;
  }
  sendLayerSpecs(state.layerSpecs, { residuals: [...(s.residuals || []), [src, dst]] });
  els.resPanel.style.display = "none";
  linechart.reset();
});

networkView.onRemoveResidual = (src, dst) => {
  const newRess = (state.residuals || []).filter(r => !(r[0] === src && r[1] === dst));
  sendLayerSpecs(state.layerSpecs, { residuals: newRess });
  linechart.reset();
};

networkView.onToggleFeature = (featureId) => { sendSet(featureId, !state[featureId]); linechart.reset(); hideZoom(); };

networkView.onNodeHover = (nodeId) => {
  if (!currentFrame) return;
  if (nodeId && currentFrame.nodeBoundaries && currentFrame.nodeBoundaries[nodeId]) {
    // reserved for hover preview
  } else {
    if (currentFrame.outputBoundary) {
      heatmap.drawBoundary(
        currentFrame.outputBoundary, state.discretize,
        currentFrame.outputMode || "value", currentFrame.numClasses || 1,
      );
    }
  }
};

// ---- legend ----

let lastLegendSig = "";
function updateLegend(isMC, numClasses) {
  const sig = `${isMC}|${numClasses}`;
  if (sig === lastLegendSig) return;
  lastLegendSig = sig;
  const cm = document.querySelector(".colormap");
  if (isMC) {
    if (cm) cm.style.display = "none";
    if (classLegend) {
      classLegend.style.display = "flex";
      classLegend.innerHTML = "";
      for (let k = 0; k < numClasses; k++) {
        const [r, g, b] = classColor(k);
        const item = document.createElement("div");
        item.className = "class-legend-item";
        const sw = document.createElement("span");
        sw.className = "swatch";
        sw.style.background = `rgb(${r},${g},${b})`;
        item.appendChild(sw);
        item.appendChild(document.createTextNode(`C${k}`));
        classLegend.appendChild(item);
      }
    }
  } else {
    if (cm) cm.style.display = "";
    if (classLegend) classLegend.style.display = "none";
  }
}

// ---- expert selection maps (per MoE layer, under its column) ----

function renderExpertMaps(frame) {
  const container = els.expertMapsContainer;
  if (!container) return;
  const moeInfo = frame.moeInfo || [];
  if (moeInfo.length === 0) {
    container.innerHTML = "";
    container.dataset.sig = "";
    return;
  }

  const sig = moeInfo.map(m => `L${m.layer}:D${m.expertDepth || 1}`).join(",");
  if (container.dataset.sig !== sig) {
    container.dataset.sig = sig;
    container.innerHTML = "";
    moeInfo.forEach(info => {
      const wrap = document.createElement("div");
      wrap.className = "expert-map";
      wrap.dataset.layer = String(info.layer);
      const title = document.createElement("div");
      title.className = "expert-map-title";
      title.textContent = `L${info.layer} routing (D=${info.expertDepth || 1})`;
      const canvas = document.createElement("canvas");
      canvas.width = 80; canvas.height = 80;
      canvas.style.width = "80px"; canvas.style.height = "80px";
      wrap.appendChild(title);
      wrap.appendChild(canvas);
      container.appendChild(wrap);
    });
  }

  moeInfo.forEach(info => {
    const wrap = container.querySelector(`.expert-map[data-layer="${info.layer}"]`);
    if (!wrap) return;
    const canvas = wrap.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    const selected = networkView.getSelectedExpert(info.layer);
    const isSoft = !!info.soft;
    const weights = info.routingWeights;   // E x H x W
    if (!weights || !weights[selected]) return;
    const W = weights[selected][0].length;
    const H = weights[selected].length;

    const img = ctx.createImageData(W, H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const v = weights[selected][j][i];
        const idx = (j * W + i) * 4;
        if (isSoft) {
          img.data[idx]     = Math.round(255 * (1 - v));
          img.data[idx + 1] = Math.round(255 * (1 - v * 0.7));
          img.data[idx + 2] = 255;
        } else {
          const on = v > 1e-6;
          img.data[idx]     = on ? 50 : 255;
          img.data[idx + 1] = on ? 100 : 255;
          img.data[idx + 2] = on ? 220 : 255;
        }
        img.data[idx + 3] = 255;
      }
    }
    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, 80, 80);
    ctx.drawImage(off, 0, 0, 80, 80);
  });
}

// ---- frame handling ----

function _networkSignature(frame) {
  const expertSig = (frame.moeInfo || []).map(m =>
    `${m.layer}:${networkView.getSelectedExpert(m.layer)}:${m.expertDepth || 1}`
  ).join("|");
  return JSON.stringify(frame.nodeIds) + JSON.stringify(frame.inputIds)
    + JSON.stringify(frame.residuals)
    + JSON.stringify(state.layerSpecs)
    + "|" + expertSig;
}

function onFrame(frame) {
  currentFrame = frame;
  const serverState = State.fromDict(frame.state);
  if (pendingCustomDatasetCheck) {
    pendingCustomDatasetCheck = false;
    if (!serverState.customDatasetId) {
      showCopyToast("Custom dataset not found on server — reverted to default dataset", true);
    } else {
      showCopyToast("Configuration loaded (with custom dataset)");
    }
  }

  for (const k of Array.from(pendingKeys)) {
    if (shallowEqual(serverState[k], state[k])) pendingKeys.delete(k);
  }
  const merged = State.fromDict(frame.state);
  for (const k of pendingKeys) merged[k] = state[k];
  state = merged;
  state.ensureSpecs();

  applyStateToUI(state);

  els.iter.textContent = fmtInt(frame.iter);
  els.lossTrain.textContent = fmtLoss(frame.lossTrain);
  els.lossTest.textContent = fmtLoss(frame.lossTest);

  if (frame.auxLossTrain > 0 && els.lossTrain) {
    els.lossTrain.title = `Aux: ${frame.auxLossTrain.toFixed(4)}`;
  }

  if (frame.iter === 0) linechart.reset();
  linechart.append(frame.lossTrain, frame.lossTest);

  const isMC = frame.outputMode === "class";
  const numClasses = frame.numClasses || 1;
  const pointColorFn = isMC ? (l) => classColor(Math.round(l)) : (l) => colorFor(l);

  heatmap.setTrainPoints(frame.trainPoints || [], pointColorFn);
  heatmap.setTestPoints(frame.testPoints || [], pointColorFn);
  if (frame.outputBoundary) {
    heatmap.drawBoundary(
      frame.outputBoundary, state.discretize,
      frame.outputMode || "value", numClasses,
    );
  }
  updateLegend(isMC, numClasses);
  updateDiscretizeVisibility(isMC);

  const signature = _networkSignature(frame);
  if (signature !== lastNetworkSignature) {
    networkView.build(frame, state);
    lastNetworkSignature = signature;
    hideZoom();
  }
  networkView.update(frame);

  renderExpertMaps(frame);

  const selectedName = state.problem === "regression" ? state.regDataset : state.dataset;
  datasetThumbs.setSelected(selectedName, state.problem);

  if (frame.pyCode && window.Prism) {
    els.pyCode.textContent = frame.pyCode;
    Prism.highlightElement(els.pyCode);
  }
}

api.on("open", () => { els.connStatus.textContent = "connected"; els.connStatus.className = "connected"; });
api.on("close", () => {
  els.connStatus.textContent = "disconnected — retrying…";
  els.connStatus.className = "disconnected";
  els.playPause.classList.remove("playing");
  els.playPause.innerHTML = SVG_PLAY;
});
api.on("welcome", (welcome) => {
  const welcomeState = State.fromDict(welcome.state);
  if (!residualUIEnabled && welcomeState.residuals && welcomeState.residuals.length > 0) {
    residualUIEnabled = true;
  }
  applyStateToUI(welcomeState);
  updateDiscretizeVisibility(welcomeState.problem === "multiclass");
  datasetThumbs.load();
  if (!sentInitialState && location.hash.length > 1) {
    sentInitialState = true;
    api.send({ cmd: "load_state", state: State.fromHash().toDictWire() });
  } else { sentInitialState = true; }
});

api.on("frame", onFrame);
api.on("node_boundary", (msg) => {
  if (isPlaying()) return;
  showZoom(msg.nodeId, msg.boundary);
});

// ---- custom dataset upload ----
const filesInput = document.getElementById("files");
const customPreviewBox = document.getElementById("custom-preview-box");

function drawCustomPreview(points, problem) {
  if (!customPreviewBox) return;
  customPreviewBox.innerHTML = "";
  const canvas = document.createElement("canvas");
  canvas.width = 60;
  canvas.height = 60;
  canvas.style.width = "100%";
  canvas.style.height = "100%";
  customPreviewBox.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  const isMC = problem === "multiclass";
  points.forEach(p => {
    const [r, g, b] = isMC ? classColor(Math.round(p[2])) : colorFor(p[2]);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    const x = 60 * (p[0] + 6) / 12;
    const y = 60 * (-p[1] + 6) / 12;
    ctx.fillRect(x, y, 2, 2);
  });
}

async function handleCustomFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".json")) {
    showCopyToast("Only .json files are supported", true);
    return;
  }

  let raw;
  try {
    raw = JSON.parse(await file.text());
  } catch (_) {
    showCopyToast("Invalid JSON file", true);
    return;
  }

  let points;
  if (Array.isArray(raw)) points = raw;
  else if (raw && Array.isArray(raw.points)) points = raw.points;
  else {
    showCopyToast("Expected [{x,y,label},...] or {points: [...]}", true);
    return;
  }

  const cleaned = [];
  for (const p of points) {
    if (typeof p.x !== "number" || typeof p.y !== "number" || typeof p.label !== "number") {
      showCopyToast("Each point must have numeric x, y, label", true);
      return;
    }
    cleaned.push({ x: p.x, y: p.y, label: p.label });
  }
  if (cleaned.length < 2) {
    showCopyToast("Need at least 2 points", true);
    return;
  }

  showCopyToast(`Uploading ${cleaned.length} points…`);

  let resp;
  try {
    resp = await fetch("/api/datasets/custom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points: cleaned }),
    });
  } catch (_) {
    showCopyToast("Upload failed: network error", true);
    return;
  }

  if (!resp.ok) {
    let msg = `Upload failed (${resp.status})`;
    try {
      const err = await resp.json();
      if (err && err.detail) msg = `Upload failed: ${err.detail}`;
    } catch (_) {}
    showCopyToast(msg, true);
    return;
  }

  const data = await resp.json();
  // `customDatasetId` 的 set 处理里会自动推断 problem / numClasses
  sendSet("customDatasetId", data.datasetId);
  // 本地 UI 立刻刷新一次问题类型
  els.problem.value = data.problem;
  updateDiscretizeVisibility(data.problem === "multiclass");
  drawCustomPreview(data.preview, data.problem);
  showCopyToast(
    `Loaded ${data.numPoints} points (${data.problem}, K=${data.numClasses})`
  );
}

if (filesInput) {
  filesInput.addEventListener("change", e => {
    const file = e.target.files && e.target.files[0];
    if (file) handleCustomFile(file);
    e.target.value = "";  // allow re-uploading the same file
  });
}

// On welcome: if a custom dataset is already attached, show its preview
// (the server may have it from a previous upload); if not, leave box empty.
api.on("welcome", (welcome) => {
  if (welcome.state && welcome.state.customDatasetId && customPreviewBox) {
    // The preview itself is only known client-side, so we just clear the
    // box and rely on the user to re-upload if they want the thumbnail.
    customPreviewBox.innerHTML = "";
    customPreviewBox.title = "Custom dataset attached (preview unavailable after reconnect)";
  }
});

// ---- copy Python code ----
let _toastTimer = null;
function showCopyToast(msg, isError = false) {
  let toast = document.getElementById("copy-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "copy-toast";
    toast.className = "copy-toast";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.toggle("error", !!isError);
  // force reflow so the transition plays even on rapid clicks
  void toast.offsetWidth;
  toast.classList.add("show");
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toast.classList.remove("show"), 1800);
}

async function copyPyCode() {
  if (!els.pyCode) return;
  const text = els.pyCode.textContent || "";
  if (!text.trim()) {
    showCopyToast("Nothing to copy yet", true);
    return;
  }

  let ok = false;
  // preferred path: async Clipboard API (requires secure context)
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      ok = true;
    }
  } catch (_) { ok = false; }

  // fallback: legacy execCommand for http:// LAN deployments
  if (!ok) {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      ok = document.execCommand("copy");
      document.body.removeChild(ta);
    } catch (_) { ok = false; }
  }

  if (ok) {
    showCopyToast("Python code copied to clipboard");
    if (els.copyPyBtn) {
      els.copyPyBtn.classList.add("copied");
      setTimeout(() => els.copyPyBtn.classList.remove("copied"), 1200);
    }
  } else {
    showCopyToast("Copy failed — please select manually", true);
  }
}

if (els.copyPyBtn) {
  els.copyPyBtn.addEventListener("click", copyPyCode);
}

// ---- configuration save / import ----
const saveConfigBtn = document.getElementById("save-config-btn");
const importConfigBtn = document.getElementById("import-config-btn");
const importConfigInput = document.getElementById("import-config-input");

if (saveConfigBtn) {
  saveConfigBtn.addEventListener("click", () => {
    const payload = {
      app: "nn-playground",
      version: 1,
      savedAt: new Date().toISOString(),
      state: state.toDictWire(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const a = document.createElement("a");
    a.href = url;
    a.download = `nn-playground-${ts}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showCopyToast("Configuration saved to file");
  });
}

async function handleImportConfigFile(file) {
  let raw;
  try {
    raw = JSON.parse(await file.text());
  } catch (_) {
    showCopyToast("Invalid JSON file", true);
    return;
  }

  let incoming;
  if (raw && typeof raw === "object" && raw.state && typeof raw.state === "object") {
    incoming = raw.state;
  } else if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    incoming = raw;   // bare-state fallback
  } else {
    showCopyToast("Unrecognized file format", true);
    return;
  }

  let imported;
  try {
    imported = State.fromDict(incoming);
    imported.ensureSpecs();
  } catch (_) {
    showCopyToast("Failed to parse configuration", true);
    return;
  }

  if (!api.ws || api.ws.readyState !== WebSocket.OPEN) {
    showCopyToast("Not connected — cannot import right now", true);
    return;
  }

  const hadCustom = !!imported.customDatasetId;
  pendingKeys.clear();
  api.send({ cmd: "load_state", state: imported.toDictWire() });
  state = imported;
  state.writeHash();
  linechart.reset();
  hideZoom();
  pendingCustomDatasetCheck = hadCustom;
  if (!hadCustom) {
    showCopyToast("Configuration loaded");
  }
}

if (importConfigBtn && importConfigInput) {
  importConfigBtn.addEventListener("click", () => importConfigInput.click());
  importConfigInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) handleImportConfigFile(file);
    e.target.value = "";
  });
}

api.connect();