"use client";

import { useEffect, useState } from "react";

export const RECORDING_KEY = "phasor-recording";

export function outputLabel(value) {
  if (!value) return "";
  const marker = "phasor_output";
  const index = value.indexOf(marker);
  if (index >= 0) return value.slice(index);
  const parts = value.split("/").filter(Boolean);
  return parts.slice(-2).join("/");
}

export function analysisFolderName(plotId, title) {
  if (plotId === "firing_rate_vs_contrast") return "firing_rate";
  if (plotId === "viterbi_polar_sustained_transient") return "sustained_transient_polar";
  if (plotId === "dominant_modes_analysis") return "dominant_modes";
  if (plotId === "mode_summary_plots") return "mode_summary_plots";
  if (plotId === "interactive_viterbi_html") return "interactive_viterbi";
  const source = title || plotId;
  return source.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");
}

export function withContrast(folder, contrast) {
  if (!folder) return "";
  return /\/c\d+$/.test(folder) ? folder.replace(/\/c\d+$/, `/${contrast}`) : `${folder}/${contrast}`;
}

export function valuesForStage(stage, dataset, contrast = "c90") {
  const workspace = dataset.workspace || {};
  const files = dataset.files || {};
  const dirs = dataset.dirs || {};
  const outputKey = { stats: "stats", raster: "raster", filter: "filtered" }[stage.id];
  const next = {};
  for (const field of stage.fields) {
    if (field.key === "out") next.out = workspace[outputKey] || "";
    else if (field.key === "data" && stage.id === "filter") next.data = dirs.data || "";
    else if (field.key === "data") next.data = workspace.filtered || "";
    else if (field.key === "run") next.run = workspace.labels || "";
    else if (field.key === "cond") {
      const base = stage.id === "sustained_transient" ? workspace.sustained : workspace.fourier;
      next.cond = withContrast(base || "", contrast);
    } else next[field.key] = files[field.key] || "";
  }
  return next;
}

export async function prepareRecording(id) {
  if (!id || id === "other") return null;
  const res = await fetch("/api/datasets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  return res.json();
}

export function useRecordingCatalog() {
  const [datasets, setDatasets] = useState([]);
  const [choice, setChoiceState] = useState("");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let alive = true;
    fetch("/api/datasets")
      .then((res) => {
        if (!res.ok) throw new Error("recordings unavailable");
        return res.json();
      })
      .then((data) => {
        if (!alive) return;
        const rows = data.datasets || [];
        setDatasets(rows);
        const saved = localStorage.getItem(RECORDING_KEY) || "";
        const known = rows.some((row) => row.id === saved);
        const next = known || saved === "other" ? saved : rows[0]?.id || "";
        if (next && next !== saved) localStorage.setItem(RECORDING_KEY, next);
        setChoiceState(next);
        setReady(true);
      })
      .catch(() => {
        if (!alive) return;
        setDatasets([]);
        setReady(true);
      });
    function onCustom(event) {
      if (typeof event.detail === "string") setChoiceState(event.detail);
    }
    window.addEventListener("phasor-recording", onCustom);
    return () => {
      alive = false;
      window.removeEventListener("phasor-recording", onCustom);
    };
  }, []);

  function setChoice(id) {
    setChoiceState(id);
    if (id) localStorage.setItem(RECORDING_KEY, id);
    else localStorage.removeItem(RECORDING_KEY);
    window.dispatchEvent(new CustomEvent("phasor-recording", { detail: id }));
    if (id && id !== "other") prepareRecording(id);
  }

  const dataset = datasets.find((row) => row.id === choice) || null;
  const animals = [...new Set(datasets.map((row) => row.animal))];
  return { datasets, animals, choice, setChoice, dataset, ready };
}

export default function RecordingSelect({ choice, datasets, animals, onChange, hint }) {
  return (
    <div className="mb-5">
      <label className="mb-1.5 block text-sm text-neutral-400">Recording</label>
      <select
        value={choice}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2.5 text-sm text-neutral-100"
      >
        <option value="">Choose a recording</option>
        {animals.map((animal) => (
          <optgroup key={animal} label={animal}>
            {datasets
              .filter((row) => row.animal === animal)
              .map((row) => (
                <option key={row.id} value={row.id}>
                  {row.label}
                </option>
              ))}
          </optgroup>
        ))}
        <option value="other">Other files</option>
      </select>
      <p className="mt-1.5 text-xs leading-relaxed text-neutral-500">
        {hint || "WT1, WT2, WT3, and RD1 load their own files. Other files lets you browse a different dataset."}
      </p>
    </div>
  );
}
