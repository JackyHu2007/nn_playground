export class LineChart {
  constructor(container, colors = ["#777", "#000"]) {
    this.colors = colors;
    this.data = [[], []];   // [train, test]
    this.minY = Infinity;
    this.maxY = -Infinity;

    const rect = container.getBoundingClientRect();
    this.width = Math.max(rect.width || 150, 80);
    this.height = 55;

    this.canvas = document.createElement("canvas");
    this.canvas.width = this.width * 2;
    this.canvas.height = this.height * 2;
    this.canvas.style.width = `${this.width}px`;
    this.canvas.style.height = `${this.height}px`;
    this.canvas.style.display = "block";
    container.appendChild(this.canvas);

    this.ctx = this.canvas.getContext("2d");
    this.ctx.scale(2, 2);
  }

  reset() {
    this.data = [[], []];
    this.minY = Infinity;
    this.maxY = -Infinity;
    this._redraw();
  }

  append(train, test) {
    if (!Number.isFinite(train) || !Number.isFinite(test)) return;
    this.data[0].push(train);
    this.data[1].push(test);
    this.minY = Math.min(this.minY, train, test);
    this.maxY = Math.max(this.maxY, train, test);
    this._redraw();
  }

  _redraw() {
    const { ctx, width, height } = this;
    ctx.clearRect(0, 0, width, height);
    const n = this.data[0].length;
    if (n < 2) return;

    const pad = 4;
    const yMin = this.minY;
    const yMax = Math.max(this.maxY, yMin + 1e-6);
    const xs = k => pad + (k / (n - 1)) * (width - 2 * pad);
    const ys = v => height - pad - ((v - yMin) / (yMax - yMin)) * (height - 2 * pad);

    for (let s = 0; s < 2; s++) {
      const series = this.data[s];
      if (series.length < 2) continue;
      ctx.beginPath();
      ctx.strokeStyle = this.colors[s];
      ctx.lineWidth = 1.5;
      for (let k = 0; k < series.length; k++) {
        const x = xs(k);
        const y = ys(series[k]);
        if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }
}