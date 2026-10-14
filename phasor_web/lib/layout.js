const STAGE_SUFFIXES = [
  /\/01_preprocessing\/stats$/,
  /\/01_preprocessing\/rasters$/,
  /\/01_preprocessing$/,
  /\/02_processing\/filtered$/,
  /\/02_processing$/,
  /\/03_labels\/label_index_maps$/,
  /\/03_labels$/,
  /\/04_fourier\/c\d+$/,
  /\/04_fourier$/,
  /\/05_hmm(?:\/phasor_hmm_results)?$/,
  /\/06_analysis(?:\/[^/]+)?$/,
];

export function asRunRoot(folder) {
  if (!folder) return "";
  let value = String(folder).replace(/\/+$/, "");
  let changed = true;
  while (changed && value) {
    changed = false;
    for (const pattern of STAGE_SUFFIXES) {
      if (pattern.test(value)) {
        value = value.replace(pattern, "");
        changed = true;
        break;
      }
    }
  }
  return value;
}

export function runLayout(root, contrast = "c90") {
  const base = String(root || "").replace(/\/+$/, "");
  const contrastName = contrast || "c90";
  if (!base) {
    return {
      root: "",
      preprocessing: "",
      stats: "",
      raster: "",
      filtered: "",
      labels: "",
      fourier: "",
      hmm: "",
      analysis: "",
    };
  }
  return {
    root: base,
    preprocessing: `${base}/01_preprocessing`,
    stats: `${base}/01_preprocessing/stats`,
    raster: `${base}/01_preprocessing/rasters`,
    filtered: `${base}/02_processing/filtered`,
    labels: `${base}/03_labels`,
    fourier: `${base}/04_fourier/${contrastName}`,
    hmm: `${base}/05_hmm`,
    analysis: `${base}/06_analysis`,
  };
}

export function canonicalField(stageId, key, layout) {
  if (!layout || !layout.root) return "";
  if (key === "out" || key === "output") {
    if (stageId === "locate_synctones" || stageId === "processor") return layout.preprocessing;
    if (stageId === "stats") return layout.stats;
    if (stageId === "raster") return layout.raster;
    if (stageId === "filter") return layout.filtered;
    if (stageId === "hmm") return layout.hmm;
    if (stageId === "plots") return layout.analysis;
  }
  if (key === "data" && stageId === "cluster") return layout.filtered;
  if (key === "run" || key === "runDir") return layout.root;
  if (key === "cond") return layout.fourier;
  return "";
}

export function chooseOutputFolder(saved, stageId, key, value, contrast = "c90") {
  const trimmed = String(value || "").replace(/\/+$/, "");
  const next = { ...(saved || {}), manual: { ...((saved && saved.manual) || {}) } };
  const manualKey = `${stageId}:${key}`;
  if (!trimmed) {
    delete next.manual[manualKey];
    return next;
  }
  const root = asRunRoot(trimmed);
  const previous = next.workspaceRoot || "";
  if (root && root !== previous) {
    for (const [name, stored] of Object.entries(next.manual)) {
      if (name.startsWith("file:")) continue;
      if (previous && String(stored).startsWith(previous)) delete next.manual[name];
    }
    next.workspaceRoot = root;
  }
  const layout = runLayout(next.workspaceRoot || root, contrast);
  const canonical = canonicalField(stageId, key, layout);
  if (!canonical || trimmed === canonical || trimmed === root) delete next.manual[manualKey];
  else next.manual[manualKey] = trimmed;
  next.runDir = layout.root;
  next.analysisOut = layout.analysis;
  next.hmmOut = layout.hmm;
  if (!next.manual["file:npz"] && layout.preprocessing) next.npz = `${layout.preprocessing}/array_4d.npz`;
  if (!next.manual["file:synctones"] && layout.preprocessing) next.synctones = `${layout.preprocessing}/synctones.csv`;
  if (!next.manual["file:bundle"] && layout.hmm) next.bundle = `${layout.hmm}/phasor_hmm_results/bundle.npz`;
  return next;
}
