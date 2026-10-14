import fs from "fs";
import path from "path";
import { asRunRoot, runLayout } from "./layout";

function realFile(filePath) {
  try {
    const stat = fs.statSync(filePath);
    return stat.isFile() && stat.size > 0;
  } catch {
    return false;
  }
}

function namesIn(dir) {
  if (!dir || !fs.existsSync(dir)) return [];
  try {
    if (!fs.statSync(dir).isDirectory()) return [];
    return fs.readdirSync(dir);
  } catch {
    return [];
  }
}

function hasNamedFile(dir, pattern) {
  return namesIn(dir).some((name) => pattern.test(name) && realFile(path.join(dir, name)));
}

function hasFilteredArrays(dir) {
  for (const name of namesIn(dir)) {
    if (name.startsWith(".") || name.startsWith("_")) continue;
    const full = path.join(dir, name);
    let stat;
    try {
      stat = fs.statSync(full);
    } catch {
      continue;
    }
    if (stat.isFile() && name.endsWith(".npz") && stat.size > 0) return true;
    if (stat.isDirectory() && namesIn(full).some((inner) => inner.endsWith(".npz") && realFile(path.join(full, inner)))) return true;
  }
  return false;
}

function hasStage4(dir) {
  if (realFile(path.join(dir || "", "Stage4_Visualization", "cluster_labels.npy"))) return true;
  return namesIn(dir).some((name) => realFile(path.join(dir, name, "Stage4_Visualization", "cluster_labels.npy")));
}

function hasLabelCsv(dir) {
  const maps = path.join(dir || "", "label_index_maps");
  return namesIn(maps).some((name) => name.endsWith(".csv") && realFile(path.join(maps, name)));
}

const FIGURE = /\.(png|pdf|svg|tiff?|html|csv)$/i;

export function resultsPresent(stageId, values) {
  if (stageId === "locate_synctones") return realFile(path.join(values.out || "", "synctones.csv"));
  if (stageId === "processor") return realFile(path.join(values.out || "", "array_4d.npz"));
  if (stageId === "stats") return realFile(path.join(values.out || "", "comprehensive_stats_dashboard.png"));
  if (stageId === "raster") return hasNamedFile(values.out, /^raster_.*\.png$/i);
  if (stageId === "filter") return hasFilteredArrays(values.out);
  if (stageId === "cluster") return hasStage4(values.data);
  if (stageId === "label_map") return hasLabelCsv(values.run);
  if (stageId === "fourier") {
    return realFile(path.join(values.cond || "", "Stage5_Cell_Fourier_Analysis", "cell_fourier_results.csv"));
  }
  if (stageId === "sustained_transient") {
    return realFile(path.join(values.cond || "", "Stage6_Cell_SustainedTransient", "cell_level_sustained_transient.csv"));
  }
  return false;
}

function isFigure(name, full) {
  if (name.startsWith(".") || name.startsWith("_")) return false;
  return FIGURE.test(name) && realFile(full);
}

export function plotResultsPresent(dir) {
  for (const name of namesIn(dir)) {
    const full = path.join(dir, name);
    if (isFigure(name, full)) return true;
    let stat;
    try {
      stat = fs.statSync(full);
    } catch {
      continue;
    }
    if (!stat.isDirectory() || name.startsWith(".") || name.startsWith("_")) continue;
    if (namesIn(full).some((inner) => isFigure(inner, path.join(full, inner)))) return true;
  }
  return false;
}

export function ensureLayout(folder) {
  const root = asRunRoot(folder);
  if (!root) return "";
  const layout = runLayout(root);
  const fourierRoot = layout.fourier ? path.dirname(layout.fourier) : "";
  for (const dir of [layout.preprocessing, layout.stats, layout.raster, layout.filtered, layout.labels, fourierRoot, layout.hmm, layout.analysis]) {
    if (dir) fs.mkdirSync(dir, { recursive: true });
  }
  return root;
}

export function resultFolder(stageId, values) {
  if (stageId === "cluster") return values.data || "";
  return values.out || values.cond || values.run || values.data || "";
}
