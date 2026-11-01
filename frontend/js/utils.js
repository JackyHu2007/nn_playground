const NEG = [50, 83, 150];
const MID = [232, 234, 235];
const POS = [22, 175, 202];

function lerp(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

export function colorFor(v) {
  if (!Number.isFinite(v)) v = 0;
  const x = Math.max(-1, Math.min(1, v));
  return x < 0 ? lerp(NEG, MID, x + 1) : lerp(MID, POS, x);
}

// --- categorical palette for multiclass (K in [3, 10]) ---
export const CLASS_COLORS = [
  [50, 83, 150],    // C0 blue
  [22, 175, 202],   // C1 cyan
  [0, 208, 94],     // C2 green
  [230, 140, 0],    // C3 orange
  [180, 60, 180],   // C4 purple
  [200, 50, 50],    // C5 red
  [120, 90, 40],    // C6 brown
  [90, 90, 90],     // C7 gray
  [0, 140, 140],    // C8 teal
  [140, 120, 0],    // C9 olive
  [220, 60, 140],   // C10 pink
  [80, 60, 160],    // C11 indigo
];

export function classColor(k) {
  const n = CLASS_COLORS.length;
  const i = ((Math.round(k) % n) + n) % n;
  return CLASS_COLORS[i];
}

export function fmtLoss(v) {
  return Number.isFinite(v) ? v.toFixed(3) : "–";
}

export function fmtInt(n) {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}