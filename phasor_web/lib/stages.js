export const STAGES = [
  {
    id: "locate_synctones",
    title: "Locate synctones",
    phase: "Pre-processing",
    script: "01_preprocessing/mcs-data-tools/locate_synctones.py",
    template: "{python} {script} {h5}",
    fields: [{ key: "h5", label: "MEA recording (.h5)", kind: "file", required: true, filter: ".h5" }],
  },
  {
    id: "processor",
    title: "Processor → 4D array",
    phase: "Pre-processing",
    script: "01_preprocessing/Processor.py",
    template: "{python} {script} {spikes} {synctones} {stims}",
    fields: [
      { key: "spikes", label: "Spike file (.txt)", kind: "file", required: true, filter: ".txt" },
      { key: "synctones", label: "Synctones (.csv)", kind: "file", required: true, filter: ".csv" },
      { key: "stims", label: "Stimulus JSON", kind: "file", required: true, filter: ".json" },
    ],
  },
  {
    id: "stats",
    title: "Population stats",
    phase: "Pre-processing",
    script: "01_preprocessing/stats.py",
    template: "{python} {script} {npz} {stims} {spikes} {out}",
    fields: [
      { key: "npz", label: "4D array (.npz)", kind: "file", required: true, filter: ".npz" },
      { key: "stims", label: "Stimulus JSON", kind: "file", required: true, filter: ".json" },
      { key: "spikes", label: "Spike file (.txt)", kind: "file", required: true, filter: ".txt" },
      { key: "out", label: "Output folder", kind: "dir", required: true },
    ],
  },
  {
    id: "raster",
    title: "Raster sequence plots",
    phase: "Pre-processing",
    script: "01_preprocessing/raster_sequence_visualization.py",
    template: "{python} {script} {npz} {stims} {out}",
    fields: [
      { key: "npz", label: "4D array (.npz)", kind: "file", required: true, filter: ".npz" },
      { key: "stims", label: "Stimulus JSON", kind: "file", required: true, filter: ".json" },
      { key: "out", label: "Output folder", kind: "dir", required: true },
    ],
  },
  {
    id: "filter",
    title: "Filter by contrast",
    phase: "Processing",
    script: "02_filtering/create_filtered_npz.py",
    template: "{python} {script} --data_dir {data} --stims_json {stims} --output_dir {out}",
    fields: [
      { key: "data", label: "Data folder", kind: "dir", required: true },
      { key: "stims", label: "Stimulus JSON", kind: "file", required: true, filter: ".json" },
      { key: "out", label: "Output folder", kind: "dir", required: true },
    ],
  },
  {
    id: "cluster",
    title: "UMAP + OPTICS clustering",
    phase: "Processing",
    script: "03_clustering/automated_pipeline_clean.py",
    template: "{python} {script} --data_dir {data}",
    fields: [{ key: "data", label: "Filtered dataset folder", kind: "dir", required: true }],
  },
  {
    id: "label_map",
    title: "Label index maps",
    phase: "Fourier classification",
    script: "04_stage56/generate_label_index_map_csv.py",
    template: "{python} {script} {run}",
    fields: [{ key: "run", label: "Run directory", kind: "dir", required: true }],
  },
  {
    id: "fourier",
    title: "Fourier cell classification",
    phase: "Fourier classification",
    script: "04_stage56/fourier_cell_analysis.py",
    template: "{python} {script} {cond}",
    fields: [{ key: "cond", label: "Condition folder (c50 … c90)", kind: "dir", required: true }],
  },
  {
    id: "sustained_transient",
    title: "Sustained / transient",
    phase: "Fourier classification",
    script: "04_stage56/sustained_transient_cell_level.py",
    template: "{python} {script} {cond}",
    fields: [{ key: "cond", label: "Condition folder (c50 … c90)", kind: "dir", required: true }],
  },
  {
    id: "hmm",
    title: "Hidden Markov Model",
    phase: "Hidden Markov Model",
    script: "",
    template: "",
    fields: [{ key: "data", label: "Dataset folder and HMM output location", kind: "dir", required: true }],
    note:
      "Covariance soft-threshold search (η = 0.0005, 0.002, 0.005; α = 0.5), then refit and Viterbi. " +
      "A valid bundle.npz skips fitting.",
  },
];

export const PLOTS = [
  ["firing_rate_vs_contrast", "Firing rate of cells", "/covers/firing_rate.jpg"],
  ["viterbi_cycle_subtype_phase_activity", "Subtypes MEA Spread", "/covers/subtype_traces.jpg"],
  ["viterbi_mea_subtype_heatmaps", "Subtype maps", "/covers/subtype_maps.jpg"],
  ["viterbi_polar_sustained_transient", "Sustained / transient polar", "/covers/polar_sustained.jpg"],
  ["viterbi_polar_rep_cycle", "Polar by repetition and cycle", "/covers/polar_cycle.jpg"],
  ["viterbi_polar_12states_pizza", "Polar activity", "/covers/polar_activity.jpg"],
  ["dominant_modes_analysis", "Dominant modes", "/covers/dominant_modes.jpg"],
  ["mode_summary_plots", "Mode summary plots", "/covers/mode_summary.jpg"],
  ["interactive_viterbi_html", "Interactive HMM Viterbi Decoding", "/covers/interactive_html.jpg"],
];

export function stageById(id) {
  return STAGES.find((s) => s.id === id) || null;
}
