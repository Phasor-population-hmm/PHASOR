import fs from "fs";
import path from "path";

function hasEntries(dir) {
  if (!dir || !fs.existsSync(dir)) return false;
  let names;
  try {
    const stat = fs.statSync(dir);
    if (!stat.isDirectory()) return stat.isFile();
    names = fs.readdirSync(dir);
  } catch {
    return false;
  }
  return names.some((name) => !name.startsWith("."));
}

function hasStage4(dir) {
  if (!dir || !fs.existsSync(dir)) return false;
  const direct = path.join(dir, "Stage4_Visualization", "cluster_labels.npy");
  if (fs.existsSync(direct)) return true;
  let names;
  try {
    names = fs.readdirSync(dir);
  } catch {
    return false;
  }
  return names.some((name) => fs.existsSync(path.join(dir, name, "Stage4_Visualization", "cluster_labels.npy")));
}

export function resultsPresent(stageId, values) {
  if (stageId === "processor") return Boolean(values.out && fs.existsSync(path.join(values.out, "array_4d.npz")));
  if (stageId === "stats" || stageId === "raster" || stageId === "filter") return hasEntries(values.out);
  if (stageId === "cluster") return hasStage4(values.data);
  if (stageId === "label_map") return hasEntries(path.join(values.run || "", "label_index_maps"));
  if (stageId === "fourier") {
    return fs.existsSync(path.join(values.cond || "", "Stage5_Cell_Fourier_Analysis", "cell_fourier_results.csv"));
  }
  if (stageId === "sustained_transient") {
    return fs.existsSync(path.join(values.cond || "", "Stage6_Cell_SustainedTransient", "cell_level_sustained_transient.csv"));
  }
  return false;
}

export function plotResultsPresent(dir) {
  return hasEntries(dir);
}

export function resultFolder(stageId, values) {
  if (stageId === "cluster") return values.data || "";
  return values.out || values.cond || values.run || values.data || "";
}
