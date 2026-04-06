export const CLUSTER_COLORS = {
  0: "#5b8cff",
  1: "#22c55e",
  2: "#f59e0b",
  3: "#ef4444",
};

export const INDICATOR_LABELS = {
  S1: "Service Volume",
  S2: "Resolution Speed",
  S3: "Service Diversity",
  S4_pos: "Positive Services",
  S4_neg: "Negative Services",
  N1: "Housing Density Ratio",
  N2: "Space Crowding",
  N3: "Property Age & Rent Control",
  N4: "Affordability Need",
  N5: "Tenure Instability",
};

export function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const n = parseInt(h, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function rgbToHex({ r, g, b }) {
  const toHex = (x) => x.toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

/**
 * Equity ramp: red -> amber -> green.
 */
export function equityColor(score01) {
  const s = clamp(score01, 0, 1);
  const red = "#ef4444";
  const amber = "#f59e0b";
  const green = "#22c55e";

  if (s <= 0.5) {
    const t = s / 0.5;
    const a = hexToRgb(red);
    const b = hexToRgb(amber);
    return rgbToHex({
      r: Math.round(lerp(a.r, b.r, t)),
      g: Math.round(lerp(a.g, b.g, t)),
      b: Math.round(lerp(a.b, b.b, t)),
    });
  }
  const t = (s - 0.5) / 0.5;
  const a = hexToRgb(amber);
  const b = hexToRgb(green);
  return rgbToHex({
    r: Math.round(lerp(a.r, b.r, t)),
    g: Math.round(lerp(a.g, b.g, t)),
    b: Math.round(lerp(a.b, b.b, t)),
  });
}

export function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}

export function getQueryParam(name, fallback = null) {
  const u = new URL(window.location.href);
  return u.searchParams.get(name) ?? fallback;
}

