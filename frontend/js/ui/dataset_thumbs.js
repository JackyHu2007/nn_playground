import { classColor, colorFor } from "../utils.js";

export class DatasetThumbs {
  constructor(syntheticContainer, realContainer, onSelect) {
    this.syntheticContainer = syntheticContainer;
    this.realContainer = realContainer;
    this.onSelect = onSelect;
    this.datasets = [];
    this.selectedName = null;
    this.currentProblem = "classification";
  }

  async load() {
    try {
      const resp = await fetch("/api/datasets");
      const data = await resp.json();
      this.datasets = data.datasets;
      this.render();
    } catch (e) {
      console.error("Failed to load datasets", e);
    }
  }

  setSelected(name, problem) {
    const changed = name !== this.selectedName || problem !== this.currentProblem;
    this.selectedName = name;
    this.currentProblem = problem;
    if (changed) this.render();
  }

  render() {
    this.syntheticContainer.innerHTML = "";
    this.realContainer.innerHTML = "";

    const filtered = this.datasets.filter(ds => ds.category === this.currentProblem);

    filtered.forEach(ds => {
      const isReal = ds.name.includes("rocks") || ds.name.includes("poro")
        || ds.name.includes("reg-porosity") || ds.name.includes("reg-dts");

      const div = document.createElement("div");
      div.className = "dataset";
      if (ds.name === this.selectedName) div.classList.add("selected");
      div.title = ds.name;

      const canvas = document.createElement("canvas");
      canvas.width = 100; canvas.height = 100;
      div.appendChild(canvas);

      this._drawPreview(canvas, ds);

      div.addEventListener("click", () => {
        this.onSelect(ds.name, ds.category);
      });

      if (isReal) {
        this.realContainer.appendChild(div);
      } else {
        this.syntheticContainer.appendChild(div);
      }
    });
  }

  _drawPreview(canvas, ds) {
    const ctx = canvas.getContext("2d");
    const isMC = ds.category === "multiclass";
    ds.preview.forEach(p => {
      const [r, g, b] = isMC ? classColor(p[2]) : colorFor(p[2]);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      const x = 100 * (p[0] + 6) / 12;
      const y = 100 * (-p[1] + 6) / 12;
      ctx.fillRect(x, y, 4, 4);
    });
  }
}