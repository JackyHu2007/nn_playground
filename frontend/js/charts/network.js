import { colorFor } from "../utils.js";

const RECT_SIZE = 44;
const MIN_RECT_SIZE = 16;
const MAX_SPACING = 50;
const MIN_GAP = 2;
const MIN_VERT_SPACING = 26;
const MIN_HORIZ_SPACING = 40;
const TOP_PAD = 180;
const BOTTOM_PAD = 50;
const MIN_HEIGHT = 520;
const BASE_WIDTH = 900;

export const INPUTS = {
  x: { label: "X₁" }, y: { label: "X₂" }, xTimesY: { label: "X₁X₂" },
  xSquared: { label: "X₁²" }, ySquared: { label: "X₂²" },
};

export class NetworkView {
  constructor(container) {
    this.container = container;
    this.svg = d3.select(container).select("svg");
    this.width = BASE_WIDTH;
    this.height = MIN_HEIGHT;
    this.linkWidthScale = d3.scaleLinear().domain([0, 5]).range([1, 8]).clamp(true);
    this.node2coord = {};
    this.rectSize = RECT_SIZE;
    this.spacing = MAX_SPACING;
    this.onNodeHover = null;
    this.onNodeClick = null;
    this.onToggleFeature = null;
    this.onRemoveResidual = null;
    this.onChangeExpertSelection = null;

    this._imageCache = new Map();
    this._selectedExperts = {};

    this.offCanvas = document.createElement("canvas");
    this.offCanvas.width = 160; this.offCanvas.height = 160;
    this.offCtx = this.offCanvas.getContext("2d");
    this.tempCanvas = document.createElement("canvas");
    this.tempCanvas.width = 10; this.tempCanvas.height = 10;
    this.tempCtx = this.tempCanvas.getContext("2d");
    this._routerImageCache = new Map();
  }

  getSelectedExpert(layerIdx) { return this._selectedExperts[layerIdx] ?? 0; }
  setSelectedExpert(layerIdx, e) { this._selectedExperts[layerIdx] = e; }

  _isMoE(state, layerIdx) {
    const specs = state.layerSpecs || [];
    return layerIdx >= 1 && layerIdx <= specs.length && specs[layerIdx - 1]?.type === "moe";
  }

  build(frame, state) {
    const svg = this.svg;
    svg.selectAll("g").remove();
    d3.select(this.container)
      .selectAll(".plus-minus-neurons, .layer-badge, .moe-expert-selector, .moe-agg-badge, .router-badge")
      .remove();
    if (!frame || !frame.nodeIds) return;

    this._imageCache.clear();
    this._routerImageCache.clear();
    const nodeIds = frame.nodeIds;
    const numLayers = nodeIds.length;
    this.node2coord = {};

    const moeByLayer = {};
    (frame.moeInfo || []).forEach(info => { moeByLayer[info.layer] = info; });

    const allInputIds = Object.keys(INPUTS);

    const columns = [];
    columns.push({ type: "input", ids: allInputIds, layerIdx: 0 });

    for (let l = 1; l < numLayers - 1; l++) {
      const currentLayer = nodeIds[l];
      const isMoE = this._isMoE(state, l);

      if (isMoE) {
        const moeInfo = moeByLayer[l];
        const E = moeInfo?.numExperts ?? state.layerSpecs[l - 1].experts;
        const selected = Math.min(this.getSelectedExpert(l), E - 1);
        this._selectedExperts[l] = selected;

        const rIds = Array.from({length: E}, (_, e) => `moe_router_${l}_${e}`);
        columns.push({ type: "router", ids: rIds, layerIdx: l, E, selected });

        const byDepth = moeInfo?.expertNodeIds?.[selected];
        const depthList = byDepth || [currentLayer];
        for (let d = 0; d < depthList.length; d++) {
          columns.push({ type: "moe_depth", ids: depthList[d], layerIdx: l, depth: d, E, selected, totalD: depthList.length });
        }

        columns.push({ type: "moe_agg", ids: currentLayer, layerIdx: l });
      } else {
        columns.push({ type: "dense", ids: currentLayer, layerIdx: l });
      }
    }

    columns.push({ type: "output", ids: nodeIds[numLayers - 1], layerIdx: numLayers - 1 });

    // Size calculation
    let maxLen = 1;
    columns.forEach(col => { if (col.ids.length > maxLen) maxLen = col.ids.length; });
    const needBody = (maxLen - 1) * MIN_VERT_SPACING + RECT_SIZE;
    this.height = Math.max(MIN_HEIGHT, TOP_PAD + BOTTOM_PAD + needBody);
    this.container.style.height = `${this.height}px`;
    this.svg.style("height", `${this.height}px`);

    const availH = this.height - TOP_PAD - BOTTOM_PAD;
    this.spacing = maxLen > 1 ? Math.min(MAX_SPACING, availH / (maxLen - 1)) : MAX_SPACING;
    this.rectSize = Math.max(MIN_RECT_SIZE, Math.min(RECT_SIZE, this.spacing - MIN_GAP));

    const getY = (i, len) => {
      const totalH = (len - 1) * this.spacing;
      const startY = TOP_PAD + (availH - totalH) / 2;
      return startY + i * this.spacing;
    };

    // Horizontal layout
    const inputX = 50;
    const numCols = columns.length;
    let spacingX = (BASE_WIDTH - 100) / (numCols - 1);
    if (spacingX < MIN_HORIZ_SPACING) {
      spacingX = MIN_HORIZ_SPACING;
      this.width = inputX + spacingX * (numCols - 1) + 50;
    } else {
      this.width = BASE_WIDTH;
    }
    this.svg.style("width", `${this.width}px`);
    this.container.style.width = `${this.width}px`;
    this.svg.attr("width", this.width);

    columns.forEach((col, c) => {
      const cx = inputX + c * spacingX;
      col.cx = cx;
      col.ids.forEach((id, i) => {
        this.node2coord[id] = { cx, cy: getY(i, col.ids.length) };
      });
    });

    // Tags and Selectors
    for (let l = 1; l < numLayers - 1; l++) {
      const denseCol = columns.find(c => c.type === "dense" && c.layerIdx === l);
      const d0Col = columns.find(c => c.type === "moe_depth" && c.layerIdx === l && c.depth === 0);
      const routerCol = columns.find(c => c.type === "router" && c.layerIdx === l);
      const aggCol = columns.find(c => c.type === "moe_agg" && c.layerIdx === l);

      if (denseCol || d0Col) {
        const targetCol = d0Col || denseCol;
        const topNodeY = getY(0, targetCol.ids.length);
        this._addLayerBadge(targetCol.cx, topNodeY - (this.rectSize / 2 + 110), l - 1, state);
      }
      if (routerCol) {
        const topNodeY = getY(0, routerCol.ids.length);
        d3.select(this.container).append("div")
          .attr("class", "router-badge")
          .style("left", `${routerCol.cx}px`)
          .style("top", `${Math.max(4, topNodeY - this.rectSize / 2 - 20)}px`)
          .text("Router");
      }
      if (d0Col) {
        const topNodeY = getY(0, d0Col.ids.length);
        this._addExpertSelector(d0Col.cx, topNodeY - (this.rectSize / 2 + 40), l, d0Col.E, d0Col.selected);
      }
      if (aggCol) {
        d3.select(this.container).append("div")
          .attr("class", "moe-agg-badge")
          .style("left", `${aggCol.cx}px`)
          .style("top", `${getY(0, aggCol.ids.length) - this.rectSize / 2 - 20}px`)
          .text("AGG");
      }
    }

    const linkContainer = svg.append("g").attr("class", "links");
    const moeLinkContainer = svg.append("g").attr("class", "moe-links");
    const skipLinkContainer = svg.append("g").attr("class", "skip-links");

    // The large arc layer is below the node layer to make sure it’s not blocked.
    const routerCurveContainer = svg.append("g").attr("class", "router-curves");
    const shadowContainer = svg.append("g").attr("class", "shadows");
    const nodeContainer = svg.append("g").attr("class", "nodes")

    for (let c = 1; c < numCols; c++) {
      const prevCol = columns[c - 1];
      const currCol = columns[c];

      // Find the actual previous layer's output column of the current layer
      const prevLayerIdx = currCol.layerIdx - 1;
      let truePrevCol = prevCol;
      if (currCol.type === "router" || (currCol.type === "moe_depth" && currCol.depth === 0)) {
        for (let k = c - 1; k >= 0; k--) {
          if (columns[k].layerIdx === prevLayerIdx) {
            truePrevCol = columns[k];
            break;
          }
        }
      }

      if (currCol.type === "router") {
        // 1. The previous layer -> Router layer
        const rWeights = frame.routerWeights?.[currCol.layerIdx - 1] || [];
        truePrevCol.ids.forEach((srcId, j) => {
          currCol.ids.forEach((destId, e) => {
            const w = rWeights[e]?.[j] ?? 0.5;
            this._drawMoeLine(srcId, destId, w, false, moeLinkContainer, 0.8);
          });
        });

      } else if (currCol.type === "moe_depth" && currCol.depth === 0) {
        // 2. Previous layer -> D1 Expert
        const eWeights = frame.expertWeights?.[currCol.layerIdx - 1]?.[currCol.selected]?.[0];
        truePrevCol.ids.forEach((srcId, j) => {
          currCol.ids.forEach((destId, i) => {
            const w = eWeights ? eWeights[i][j] : 0.5;
            this._drawMoeLine(srcId, destId, w, false, moeLinkContainer, 0.25);
          });
        });

      } else if (currCol.type === "moe_depth") {
        // 3. D(n-1) -> D(n)
        const eWeights = frame.expertWeights?.[currCol.layerIdx - 1]?.[currCol.selected]?.[currCol.depth];
        prevCol.ids.forEach((srcId, j) => {
          currCol.ids.forEach((destId, i) => {
            const w = eWeights ? eWeights[i][j] : 0.5;
            this._drawMoeLine(srcId, destId, w, false, moeLinkContainer);
          });
        });

      } else if (currCol.type === "moe_agg") {
        // 4. D(last) -> AGG
        prevCol.ids.forEach((srcId, i) => {
          const destId = currCol.ids[i];
          if (destId) this._drawMoeAggLine(srcId, destId, moeLinkContainer);
        });

        // 5. Router -> AGG
        const routerCol = columns.find(cc => cc.type === "router" && cc.layerIdx === currCol.layerIdx);
        if (routerCol) {
          const topR = this.node2coord[routerCol.ids[0]];
          const topAgg = this.node2coord[currCol.ids[0]];
          this._drawRouterToAggCurve(routerCol.cx, topR.cy, topAgg.cx, topAgg.cy, routerCurveContainer);
        }

      } else {
        // 6. Common Dense / Output
        truePrevCol.ids.forEach((srcId, j) => {
          currCol.ids.forEach((destId, i) => {
            this._drawLink(srcId, destId, currCol.layerIdx - 1, i, j, linkContainer);
          });
        });
      }
    }

    if (frame.residuals) {
      frame.residuals.forEach(([srcL, dstL]) => {
        if (srcL >= dstL) return;
        const srcLayer = nodeIds[srcL];
        const dstLayer = nodeIds[dstL];
        if (!srcLayer || !dstLayer || srcLayer.length !== dstLayer.length) return;
        srcLayer.forEach((srcId, idx) => {
          this._drawSkipLink(srcId, dstLayer[idx], srcL, idx, dstL, idx, skipLinkContainer);
        });
      });
    }

    columns.forEach(col => {
      col.ids.forEach((id, i) => {
        const cy = getY(i, col.ids.length);

        if (col.type === "input") {
          this._drawNode(col.cx, cy, id, true, nodeContainer, shadowContainer, !!state[id], false, 0, 0, 0);
        } else if (col.type === "router") {
          const eIdx = i;
          const isSelected = (eIdx === col.selected);
          const weights = frame.moeInfo?.find(m => m.layer === col.layerIdx)?.routingWeights?.[eIdx];
          this._drawRouterNode(col.cx, cy, id, eIdx, weights, isSelected, nodeContainer);
        } else if (col.type === "moe_depth") {
          const labelDepth = col.totalD > 1 ? col.depth + 1 : 0;
          const nodeG = this._drawNode(col.cx, cy, id, false, nodeContainer, shadowContainer, true, true, col.E, col.selected, labelDepth);

          nodeG.attr("transform", `translate(${col.cx},${cy}) scale(0.6) rotate(-15)`)
            .style("opacity", 0)
            .transition()
            .delay(col.depth * 50)
            .duration(450)
            .ease(d3.easeElasticOut.amplitude(1.2).period(0.4))
            .attr("transform", `translate(${col.cx},${cy}) scale(1) rotate(0)`)
            .style("opacity", 1);
        } else {
          this._drawNode(col.cx, cy, id, false, nodeContainer, shadowContainer, true, false, 0, 0, 0);
        }
      });
    });
  }

  _drawRouterNode(cx, cy, nodeId, eIdx, routingWeights, isSelected, container) {
    const g = container.append("g")
      .attr("class", `node router-node ${isSelected ? 'active' : ''}`)
      .attr("transform", `translate(${cx},${cy})`)
      .attr("data-node-id", nodeId);

    g.append("rect")
      .attr("x", -this.rectSize / 2).attr("y", -this.rectSize / 2)
      .attr("width", this.rectSize).attr("height", this.rectSize).attr("rx", 4)
      .attr("class", "node-bg")
      .attr("fill", "#f4f5f8");

    if (routingWeights) {
      this._routerImageCache.set(nodeId, this._hashFloat2D(routingWeights));
      g.append("image")
        .attr("class", "node-image")
        .attr("x", -this.rectSize / 2 + 1).attr("y", -this.rectSize / 2 + 1)
        .attr("width", this.rectSize - 2).attr("height", this.rectSize - 2)
        .attr("href", this._makeRouterDataUrl(routingWeights))
        .style("image-rendering", "auto")
        .style("pointer-events", "none");
    }

    g.append("rect")
      .attr("x", -this.rectSize / 2).attr("y", -this.rectSize / 2)
      .attr("width", this.rectSize).attr("height", this.rectSize).attr("rx", 4)
      .attr("fill", "none")
      .attr("stroke", isSelected ? "#8a3ca0" : "#aaa")
      .attr("stroke-width", isSelected ? 2.5 : 1.5);

    g.append("text")
      .attr("x", 0).attr("y", this.rectSize / 2 + 10)
      .attr("text-anchor", "middle")
      .style("font-size", "9px")
      .style("fill", isSelected ? "#8a3ca0" : "#888")
      .style("font-weight", isSelected ? "bold" : "normal")
      .text(`R${eIdx + 1}`);

    // Click to enlarge
    g.style("cursor", "pointer")
      .on("click", () => {
        if (this.onNodeClick) this.onNodeClick(nodeId, routingWeights);
      });
  }

    _makeRouterDataUrl(weights) {
    const H = weights.length;
    const W = weights[0].length;
    const img = this.tempCtx.createImageData(W, H);
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const v = weights[j][i];
        const idx = (j * W + i) * 4;
        img.data[idx]     = Math.round(240 - v * 94);
        img.data[idx + 1] = Math.round(242 - v * 191);
        img.data[idx + 2] = Math.round(244 - v * 9);
        img.data[idx + 3] = 255;
      }
    }
    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    off.getContext("2d").putImageData(img, 0, 0);
    return off.toDataURL();
  }

  _hashFloat2D(arr) {
    let h = 0;
    for (let j = 0; j < arr.length; j++) {
      const row = arr[j];
      for (let i = 0; i < row.length; i++) {
        h = (h * 31 + Math.round(row[i] * 10000)) | 0;
      }
    }
    return h;
  }


  _cssColorMoe(v) {
    if (!Number.isFinite(v)) v = 0;
    const t = Math.max(-1, Math.min(1, v));
    if (t >= 0) return `rgba(162, 58, 189, ${0.15 + 0.85 * t})`;
    else return `rgba(80, 30, 110, ${0.15 + 0.85 * (-t)})`;
  }

  _drawMoeLine(srcId, destId, weight, isRouterIn, container, forceOpacity) {
    const src = this.node2coord[srcId], dest = this.node2coord[destId];
    if (!src || !dest) return;

    const absW = Math.abs(weight);
    if (absW < 0.08 && forceOpacity === undefined) return;

    let sx = src.cx, sy = src.cy;
    let dx = dest.cx, dy = dest.cy;

    if (srcId.startsWith("moe_router")) sy += this.rectSize / 2;
    else sx += this.rectSize / 2;

    if (destId.startsWith("moe_agg")) dy -= this.rectSize / 2;
    else if (destId.match(/^E:/)) dy -= this.rectSize / 2;
    else dx -= this.rectSize / 2;

    let cp1x, cp1y;
    let cp2x, cp2y;
    if (Math.abs(src.cx - dest.cx) > 20) {
      cp1x = sx + 40; cp1y = sy;
      cp2x = dx - 40; cp2y = dy;
    } else {
      cp1x = sx; cp1y = sy + 20;
      cp2x = dx; cp2y = dy - 20;
    }

    const path = `M${sx},${sy} C${cp1x},${cp1y} ${cp2x},${cp2y} ${dx},${dy}`;
    container.append("path")
      .attr("d", path).attr("class", "link moe-link").attr("fill", "none")
      .attr("stroke", this._cssColorMoe(weight))
      .attr("stroke-width", Math.max(0.5, this.linkWidthScale(absW * 2)))
      .style("opacity", forceOpacity !== undefined ? forceOpacity : (isRouterIn ? 0.6 : 0.9))
      .attr("marker-end", "url(#markerArrow)");
  }

  _drawMoeAggLine(srcId, destId, container) {
    const src = this.node2coord[srcId], dest = this.node2coord[destId];
    if (!src || !dest) return;

    const sx = src.cx + this.rectSize / 2;
    const sy = src.cy;
    const dx = dest.cx - this.rectSize / 2;
    const dy = dest.cy;

    const cp1x = sx + (dx - sx) * 0.3;
    const cp1y = sy - 15;
    const cp2x = dx - (dx - sx) * 0.3;
    const cp2y = dy - 15;

    const path = `M${sx},${sy} C${cp1x},${cp1y} ${cp2x},${cp2y} ${dx},${dy}`;

    container.append("path")
      .attr("d", path).attr("class", "link moe-link").attr("fill", "none")
      .attr("stroke", "#8a3ca0").attr("stroke-width", 1.5)
      .style("opacity", 0.35)
      .attr("marker-end", "url(#markerArrow)");
  }

    _drawRouterToAggCurve(routerCx, topNodeCy, aggCx, topAggCy, container) {
    const sx = routerCx;
    const sy = topNodeCy - this.rectSize / 2;

    const dx = aggCx;
    const dy = topAggCy - this.rectSize / 2 - 6;

    const cp1x = sx + (dx - sx) * 0.2;
    const cp1y = sy - 90;
    const cp2x = dx - (dx - sx) * 0.2;
    const cp2y = dy - 90;

    const path = `M${sx},${sy} C${cp1x},${cp1y} ${cp2x},${cp2y} ${dx},${dy}`;

    const softPurple = "#9c7ab8";

    container.append("path")
      .attr("d", path).attr("fill", "none")
      .attr("stroke", softPurple).attr("stroke-width", 8)
      .style("opacity", 0.12).style("pointer-events", "none");

    container.append("path")
      .attr("d", path).attr("fill", "none")
      .attr("stroke", softPurple).attr("stroke-width", 2.5)
      .style("opacity", 0.85).style("pointer-events", "none")
      .attr("marker-end", "url(#markerArrow)");
  }

  _drawLink(srcId, destId, layerIdx, dstIdx, srcIdx, container) {
    const src = this.node2coord[srcId], dest = this.node2coord[destId];
    if (!src || !dest) return;
    const sx = src.cx + this.rectSize / 2 - 1, sy = src.cy;
    const dx = dest.cx - this.rectSize / 2 + 1, dy = dest.cy;
    const path = `M${sx},${sy} C${sx + 40},${sy} ${dx - 40},${dy} ${dx},${dy}`;
    container.append("path").datum({ srcId, destId, layerIdx, dstIdx, srcIdx })
      .attr("d", path).attr("class", "link").attr("fill", "none")
      .attr("stroke", "#ccc").attr("stroke-width", 1).style("opacity", 0.6)
      .attr("marker-end", "url(#markerArrow)");
  }

  _drawNode(cx, cy, nodeId, isInput, container, shadowContainer, isActive,
            isMoE, numExperts, selectedExpert, depth) {
    if (isMoE && numExperts > 1 && shadowContainer) {
      for (let i = numExperts - 1; i >= 1; i--) {
        const off = i * 4.5;
        shadowContainer.append("rect")
          .attr("x", cx - this.rectSize / 2 + off)
          .attr("y", cy - this.rectSize / 2 - off)
          .attr("width", this.rectSize).attr("height", this.rectSize)
          .attr("rx", 4)
          .attr("class", "moe-stack-shadow")
          .attr("fill", "#eef0f3")
          .attr("stroke", "#b0b8c4")
          .attr("stroke-width", "1")
          .attr("stroke-dasharray", "2 2")
          .style("pointer-events", "none");
      }
    }

    const g = container.append("g")
      .attr("class", `node ${isActive ? "active" : "inactive"}`)
      .attr("transform", `translate(${cx},${cy})`)
      .attr("data-node-id", nodeId);

    g.append("rect")
      .attr("x", -this.rectSize / 2).attr("y", -this.rectSize / 2)
      .attr("width", this.rectSize).attr("height", this.rectSize).attr("rx", 4)
      .attr("class", "node-bg");

    g.append("image")
      .attr("class", "node-image")
      .attr("x", -this.rectSize / 2 + 1).attr("y", -this.rectSize / 2 + 1)
      .attr("width", this.rectSize - 2).attr("height", this.rectSize - 2)
      .style("image-rendering", "auto")
      .style("pointer-events", "none");

    g.append("rect")
      .attr("x", -this.rectSize / 2).attr("y", -this.rectSize / 2)
      .attr("width", this.rectSize).attr("height", this.rectSize).attr("rx", 4)
      .attr("class", isMoE ? "moe-active-border" : "node-border");

    if (isMoE && depth > 0) {
      g.append("text")
        .attr("class", "node-depth-label")
        .attr("x", 0)
        .attr("y", this.rectSize / 2 + 10)
        .attr("text-anchor", "middle")
        .style("font-size", "9px")
        .style("fill", "#8a3ca0")
        .style("font-weight", "600")
        .text(`D${depth}`);
    }

    if (isInput) {
      const label = INPUTS[nodeId] ? INPUTS[nodeId].label : nodeId;
      g.append("text").attr("x", -this.rectSize / 2 - 10).attr("y", 0).text(label);
      g.style("cursor", "pointer")
        .on("click", () => { if (this.onToggleFeature) this.onToggleFeature(nodeId); });
    } else {
      g.on("mouseenter", () => { if (this.onNodeHover) this.onNodeHover(nodeId); })
        .on("mouseleave", () => { if (this.onNodeHover) this.onNodeHover(null); })
        .style("cursor", "pointer")
        .on("click", () => { if (this.onNodeClick) this.onNodeClick(nodeId); });
    }
    return g;
  }

  _addExpertSelector(x, topY, layerIdx, E, selected) {
    const wrap = d3.select(this.container).append("div")
      .attr("class", "moe-expert-selector")
      .style("left", `${x}px`)
      .style("top", `${Math.max(10, topY)}px`);

    const mkBtn = (parent, pathD, cb) => {
      const b = parent.append("button")
        .attr("class", "icon-btn")
        .style("width", "20px").style("height", "20px")
        .style("display", "flex").style("justify-content", "center").style("align-items", "center")
        .style("padding", "0");
      b.append("svg").attr("width", 12).attr("height", 12).attr("viewBox", "0 0 24 24")
        .append("path").attr("d", pathD).attr("fill", "currentColor");
      b.on("click", cb);
      return b;
    };

    const ARROW_L = "M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z";
    const ARROW_R = "M8.59 16.59L10 18l6-6-6-6-1.41 1.41L13.17 12z";

    const rowE = wrap.append("div")
      .style("display", "flex")
      .style("justify-content", "center")
      .style("align-items", "center")
      .style("gap", "4px");

    mkBtn(rowE, ARROW_L, () => {
      const e = (selected - 1 + E) % E;
      this._selectedExperts[layerIdx] = e;
      if (this.onChangeExpertSelection) this.onChangeExpertSelection(layerIdx, e);
    });
    rowE.append("span").text(`E${selected + 1}/${E}`).style("font-weight", "600");
    mkBtn(rowE, ARROW_R, () => {
      const e = (selected + 1) % E;
      this._selectedExperts[layerIdx] = e;
      if (this.onChangeExpertSelection) this.onChangeExpertSelection(layerIdx, e);
    });
  }

  _addLayerBadge(x, topY, hiddenIdx, state) {
    const spec = (state.layerSpecs || [])[hiddenIdx];
    if (!spec) return;
    const parts = [];
    if (spec.type === "moe") {
      parts.push("MoE");
      parts.push(spec.soft ? "soft" : `k=${spec.top_k}`);
      const D = spec.expert?.layers?.length || 1;
      parts.push(`D=${D}`);
      if (!spec.normalize) parts.push("un-norm");
      if (!spec.router_bias) parts.push("no-rbias");
    } else {
      parts.push(spec.act);
      if (spec.norm === "layernorm") parts.push(`LN·${spec.norm_pos}`);
      else if (spec.norm === "rmsnorm") parts.push(`RMS·${spec.norm_pos}`);
      if (spec.bias === false) parts.push("no-bias");
    }
    d3.select(this.container).append("div")
      .attr("class", "layer-badge")
      .style("left", `${x}px`)
      .style("top", `${Math.max(4, topY)}px`)
      .text(parts.join(" · "));
  }

  update(frame) {
    if (!frame || !this.node2coord) return;
    const self = this;

    this.svg.selectAll(".link:not(.moe-link):not(.moe-gate-link)").each(function (d) {
      if (!d) return;
      const layerW = frame.weights?.[d.layerIdx];
      if (!layerW || !layerW[d.dstIdx]) return;
      const w = layerW[d.dstIdx][d.srcIdx];
      if (w === null || w === undefined) return;
      d3.select(this)
        .attr("stroke", self._cssColor(w))
        .attr("stroke-width", self.linkWidthScale(Math.abs(w)));
    });

        // --- MoE router heatmap refresh ---
    const moeInfoList = frame.moeInfo || [];
    for (const info of moeInfoList) {
      const rw = info.routingWeights;
      if (!rw) continue;          // include_boundaries=False Frame: keep the old image
      const layer = info.layer;
      for (let e = 0; e < rw.length; e++) {
        const nodeId = `moe_router_${layer}_${e}`;
        const hash = this._hashFloat2D(rw[e]);
        if (this._routerImageCache.get(nodeId) === hash) continue;
        this._routerImageCache.set(nodeId, hash);
        const image = this.svg.select(
          `.router-node[data-node-id="${nodeId}"] .node-image`
        );
        if (image.empty()) continue;
        image.attr("href", this._makeRouterDataUrl(rw[e]));
      }
    }

    if (frame.nodeBoundaries) {
      this.svg.selectAll(".node.active").each(function () {
        const g = d3.select(this);
        const nodeId = this.getAttribute("data-node-id");
        if (this.classList.contains("router-node")) return;

        const boundary = frame.nodeBoundaries[nodeId];
        if (!boundary) return;

        const hash = self._hashBoundary(boundary);
        if (self._imageCache.get(nodeId) === hash) return;
        self._imageCache.set(nodeId, hash);

        const img = self.tempCtx.createImageData(10, 10);
        for (let j = 0; j < 10; j++) {
          for (let i = 0; i < 10; i++) {
            const [r, g2, b] = colorFor(boundary[i][j]);
            const idx = (j * 10 + i) * 4;
            img.data[idx] = r; img.data[idx + 1] = g2; img.data[idx + 2] = b; img.data[idx + 3] = 255;
          }
        }
        self.tempCtx.putImageData(img, 0, 0);
        self.offCtx.clearRect(0, 0, self.offCanvas.width, self.offCanvas.height);
        self.offCtx.imageSmoothingEnabled = true;
        self.offCtx.drawImage(self.tempCanvas, 0, 0, 10, 10, 0, 0, self.offCanvas.width, self.offCanvas.height);

        const image = g.select(".node-image");
        if (!image.empty()) {
          image.attr("href", self.offCanvas.toDataURL());
        }
      });
    }
  }

  _drawSkipLink(srcId, destId, srcL, srcI, dstL, dstI, container) {
    const src = this.node2coord[srcId], dest = this.node2coord[destId];
    if (!src || !dest) return;
    const sx = src.cx + this.rectSize / 2, sy = src.cy;
    const dx = dest.cx - this.rectSize / 2, dy = dest.cy;
    const path = `M${sx},${sy} C${sx + 30},${sy - 30} ${dx - 30},${dy - 30} ${dx},${dy}`;
    const self = this;
    container.append("path")
      .datum({ srcId, destId, srcL, srcI, dstL, dstI })
      .attr("d", path).attr("class", "skip-link").attr("fill", "none")
      .attr("stroke", "#888").attr("stroke-width", 1.5)
      .style("opacity", 0.8).attr("stroke-dasharray", "5 3")
      .attr("marker-end", "url(#markerArrow)")
      .style("cursor", "pointer")
      .on("click", function (d) {
        if (self.onRemoveResidual) self.onRemoveResidual(d.srcL, d.dstL);
      });
  }

  _cssColor(v) { const [r, g, b] = colorFor(v); return `rgb(${r},${g},${b})`; }

  _hashBoundary(b) {
    let h = 0;
    for (let i = 0; i < b.length; i++) {
      const row = b[i];
      for (let j = 0; j < row.length; j++) {
        h = (h * 31 + Math.round(row[j] * 100000)) | 0;
      }
    }
    return h;
  }
}