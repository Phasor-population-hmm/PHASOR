import path from "path";

export const APP_ROOT = path.resolve(process.cwd(), "..");
export const PIPELINE = path.join(APP_ROOT, "phasor_pipeline");
export const MODE_PROJECT_DEFAULT = path.join(APP_ROOT, "mode_project");
export const OUTPUT_ROOT = process.env.PHASOR_OUTPUT_ROOT
  ? path.resolve(process.env.PHASOR_OUTPUT_ROOT)
  : path.join(APP_ROOT, "phasor_output");
export const VITERBI_BANDS = path.join(PIPELINE, "05_viterbi", "viterbi_bands_wt22.py");
export const HMM_CONFIG_PATH = path.join(process.cwd(), "hmm.config.json");

export function displayPath(target) {
  if (!path.isAbsolute(target)) return target;
  const rel = path.relative(APP_ROOT, target);
  if (rel.startsWith("..")) return target;
  return rel === "" ? "." : rel;
}
