export const HMM_K_CHOICES = [3, 6, 9, 12, 15];
export const HMM_ETA_CHOICES = [0.0005, 0.002, 0.005, 0.03, 0.1];

export const DEFAULT_HMM_SEARCH = {
  k: [3, 6, 9, 12, 15],
  alpha: [0.5],
  eta: [0.0005, 0.002, 0.005],
};

const RULES = {
  k: { label: "K", integer: true, min: 2 },
  eta: { label: "eta", integer: false, min: 0, max: 1, maxCount: 8 },
};

export function formatSearchNumber(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "";
  let text = String(n);
  if (/e/i.test(text)) text = n.toFixed(12).replace(/0+$/, "").replace(/\.$/, "");
  return text;
}

function cleanList(value, kind) {
  const rule = RULES[kind];
  const raw = Array.isArray(value) ? value : String(value ?? "").split(/[\s,]+/);
  const seen = new Set();
  const out = [];
  for (const item of raw) {
    if (item === "" || item == null) continue;
    const n = Number(item);
    if (!Number.isFinite(n)) throw new Error(`${rule.label} must be a number`);
    if (rule.integer && !Number.isInteger(n)) throw new Error(`${rule.label} must be a whole number`);
    if (n <= rule.min && rule.min === 0) throw new Error(`${rule.label} must be greater than 0`);
    if (n < rule.min) throw new Error(`${rule.label} must be at least ${rule.min}`);
    if (rule.max != null && n > rule.max) throw new Error(`${rule.label} must be at most ${rule.max}`);
    const text = formatSearchNumber(rule.integer ? n : n);
    if (!/^\d+(\.\d+)?$/.test(text) || text === "0") throw new Error(`${rule.label} is not a usable value`);
    if (seen.has(text)) continue;
    seen.add(text);
    out.push(rule.integer ? n : Number(text));
  }
  if (!out.length) throw new Error(`Choose at least one ${rule.label}`);
  if (rule.maxCount != null && out.length > rule.maxCount) throw new Error(`Choose at most ${rule.maxCount} values of ${rule.label}`);
  out.sort((a, b) => a - b);
  return out;
}

export function normalizeSearch(input) {
  const source = input || {};
  const missing = source.k == null && source.K == null && source.alpha == null && source.eta == null;
  if (missing) {
    return {
      k: [...DEFAULT_HMM_SEARCH.k],
      alpha: [...DEFAULT_HMM_SEARCH.alpha],
      eta: [...DEFAULT_HMM_SEARCH.eta],
    };
  }
  return {
    k: cleanList(source.k ?? source.K ?? DEFAULT_HMM_SEARCH.k, "k"),
    alpha: [...DEFAULT_HMM_SEARCH.alpha],
    eta: cleanList(source.eta ?? DEFAULT_HMM_SEARCH.eta, "eta"),
  };
}
