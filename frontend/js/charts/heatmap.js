import { classColor, colorFor } from "../utils.js";

export class HeatMap {
  constructor(container, width = 260, density = 100) {
    this.width = width;
    this.density = density;
    this.padding = 16;

    // Boundary dirty check cache
    this._lastBoundaryHash = null;
    this._lastDiscretize = null;
    this._lastMode = null;
    this._lastNumClasses = null;

    container.style.width = `${width + this.padding}px`;
    container.style.height = `${width + this.padding}px`;

    this.canvas = document.createElement("canvas");
    this.canvas.width = density;
    this.canvas.height = density;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${width}px`;
    this.canvas.style.imageRendering = "auto";
    this.canvas.style.position = "absolute";
    this.canvas.style.left = `0px`;
    this.canvas.style.top = `0px`;
    container.appendChild(this.canvas);

    this.ctx = this.canvas.getContext("2d");

    this.svg = d3.select(container).append("svg")
      .attr("width", width + this.padding)
      .attr("height", width + this.padding)
      .style("position", "absolute")
      .style("left", "0").style("top", "0")
      .style("pointer-events", "none");

    const xScale = d3.scaleLinear().domain([-6, 6]).range([0, width]);
    const yScale = d3.scaleLinear().domain([6, -6]).range([0, width]);

    const xAxis = d3.axisBottom(xScale).ticks(12).tickSizeOuter(0).tickPadding(4);
    const yAxis = d3.axisRight(yScale).ticks(12).tickSizeOuter(0).tickPadding(4);

    this.svg.append("g")
      .attr("transform", `translate(0, ${width})`)
      .call(xAxis)
      .call(g => g.select(".domain").style("stroke", "#ccd2da"))
      .call(g => g.selectAll(".tick text").style("font-size", "10px").style("fill", "#888"))
      .call(g => g.selectAll(".tick line").style("stroke", "#ccd2da"));

    this.svg.append("g")
      .attr("transform", `translate(${width}, 0)`)
      .call(yAxis)
      .call(g => g.select(".domain").style("stroke", "#ccd2da"))
      .call(g => g.selectAll(".tick text").style("font-size", "10px").style("fill", "#888"))
      .call(g => g.selectAll(".tick line").style("stroke", "#ccd2da"));

    this.trainLayer = this.svg.append("g");
    this.testLayer = this.svg.append("g");

    this.xScale = d3.scaleLinear().domain([-6, 6]).range([0, width]);
    this.yScale = d3.scaleLinear().domain([-6, 6]).range([width, 0]);
  }

  /**
   * @param boundary 100x100 grid.
   * @param discretize bool (ignored in class mode).
   * @param mode "value" | "class".
   * @param numClasses number of classes when mode === "class".
   */
  drawBoundary(boundary, discretize, mode = "value", numClasses = 1) {
    if (!boundary || boundary.length !== this.density) return;

    const isClass = mode === "class";
    const hash = this._hashBoundary(boundary);
    if (hash === this._lastBoundaryHash
        && discretize === this._lastDiscretize
        && isClass === (this._lastMode === "class")
        && numClasses === this._lastNumClasses) return;
    this._lastBoundaryHash = hash;
    this._lastDiscretize = discretize;
    this._lastMode = mode;
    this._lastNumClasses = numClasses;

    const img = this.ctx.createImageData(this.density, this.density);
    const data = img.data;
    for (let j = 0; j < this.density; j++) {
      for (let i = 0; i < this.density; i++) {
        const v = boundary[i][j];
        let rgb;
        if (isClass) {
          rgb = classColor(Math.round(v));
        } else {
          const vv = discretize ? (v >= 0 ? 1 : -1) : v;
          rgb = colorFor(vv);
        }
        const idx = (j * this.density + i) * 4;
        data[idx] = rgb[0]; data[idx + 1] = rgb[1]; data[idx + 2] = rgb[2]; data[idx + 3] = 220;
      }
    }
    this.ctx.putImageData(img, 0, 0);
  }

  setTrainPoints(points, colorFn) { this._drawPoints(this.trainLayer, points, false, colorFn); }
  setTestPoints(points, colorFn) { this._drawPoints(this.testLayer, points, true, colorFn); }

  _drawPoints(layer, points, isTest = false, colorFn = null) {
    const sel = layer.selectAll("circle").data(points, d => `${d[0]},${d[1]},${d[2]}`);
    sel.exit().remove();

    const enter = sel.enter().append("circle")
      .attr("r", 3)
      .attr("stroke-width", isTest ? 1.5 : 1)
      .attr("fill-opacity", 0.9);

    enter.merge(sel)
      .attr("cx", d => this.xScale(d[0]))
      .attr("cy", d => this.yScale(d[1]))
      .attr("fill", d => {
        const [r, g, b] = colorFn ? colorFn(d[2]) : colorFor(d[2]);
        return `rgb(${r},${g},${b})`;
      })
      .attr("stroke", isTest ? "#333" : "white")
      .attr("stroke-opacity", isTest ? 1 : 0.8);
  }

  _hashBoundary(b) {
    let h = 0;
    for (let i = 0; i < b.length; i++) {
      const row = b[i];
      for (let j = 0; j < row.length; j++) {
        h = (h * 31 + Math.round(row[j] * 10000)) | 0;
      }
    }
    return h;
  }
}