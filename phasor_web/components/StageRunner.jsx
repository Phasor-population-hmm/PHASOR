"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import PathField from "./PathField";
import Console from "./Console";
import Breadcrumb from "./Breadcrumb";
import RecordingSelect, { outputLabel, outputRootFromDatasets, prepareRecording, readOtherInputs, rememberOtherInputs, useRecordingCatalog, valuesForOther, valuesForStage, withContrast } from "./RecordingPicker";
import { ensureFree } from "./JobDock";
import { useJobPoll } from "@/hooks/useJobPoll";
import { STAGES } from "@/lib/stages";

const STAGE_HINT = {
  locate_synctones: "The .h5 for this recording is already filled in.",
  stats: "The dashboard is saved in 01_preprocessing/stats.",
  raster: "Raster plots are saved in 01_preprocessing/rasters.",
  filter: "Writes one folder per contrast (c0, c50, c60, c70, c80, c90) at 2 Hz. Contrasts are not mixed into one file.",
  cluster: "Clusters each contrast folder in this directory. Run Filter by contrast first.",
  label_map: "Reads the recording array and the clustered contrast folders, then writes the maps here.",
  processor: "array_4d.npz and array_metadata.npz are saved in 01_preprocessing.",
  fourier: "Pick a contrast, including C 0 %. That contrast must already be clustered in the filtered folder. Individual cell profiles are not saved.",
  sustained_transient: "Uses the same contrast as Fourier classification, including C 0 %. Individual cell profiles are not saved.",
};

export default function StageRunner({ stage }) {
  const memoryKey = `phasor-stage-${stage.id}`;
  const [values, setValues] = useState(() => Object.fromEntries(stage.fields.map((f) => [f.key, ""])));
  const [saved, setSaved] = useState(null);
  const [error, setError] = useState("");
  const [contrast, setContrast] = useState("c90");
  const [askOverwrite, setAskOverwrite] = useState(false);
  const { datasets, animals, choice, setChoice, dataset } = useRecordingCatalog();
  const { lines, status, watch, reset } = useJobPoll();

  useEffect(() => {
    const raw = localStorage.getItem(memoryKey);
    if (!raw) return;
    try {
      setSaved(JSON.parse(raw));
    } catch {
      localStorage.removeItem(memoryKey);
    }
  }, [memoryKey]);

  useEffect(() => {
    fetch("/api/session")
      .then((res) => res.json())
      .then((session) => {
        if (session.status === "running" && session.kind === "job" && session.stage === stage.id && session.id) {
          watch(session.id);
        }
      })
      .catch(() => {});
  }, [stage.id, watch]);

  useEffect(() => {
    if (!choice) return;
    if (choice === "other") {
      setValues(valuesForOther(stage, readOtherInputs(), contrast));
      return;
    }
    if (!dataset) return;
    setValues(valuesForStage(stage, dataset, contrast));
  }, [choice, dataset, stage, contrast]);

  const siblings = STAGES.filter((s) => s.phase === stage.phase);

  function setField(key, value) {
    setValues((prev) => ({ ...prev, [key]: value }));
    if (choice !== "other") return;
    const patch = {};
    if (key === "out") patch.outputs = { [stage.id]: value };
    else patch[key] = value;
    if (["spikes", "synctones", "stims", "npz", "h5"].includes(key) && value) {
      const parent = value.replace(/\/[^/]+$/, "");
      const name = parent.split("/").filter(Boolean).pop();
      const outputRoot = outputRootFromDatasets(datasets);
      if (outputRoot && name) patch.workspaceRoot = `${outputRoot}/runs/${name}`;
      if (key !== "h5") patch.data = parent;
    }
    if (stage.id === "processor" && key === "out" && value) patch.npz = `${value.replace(/\/$/, "")}/array_4d.npz`;
    if (key === "data" && stage.id === "filter") patch.data = value;
    rememberOtherInputs(patch);
  }

  async function run(overwrite = false) {
    setError("");
    setAskOverwrite(false);
    if (!(await ensureFree())) return;
    if (dataset) await prepareRecording(dataset.id);
    if (choice === "other") {
      const patch = { ...values };
      if (values.out) patch.outputs = { [stage.id]: values.out };
      if (stage.id === "processor" && values.out) patch.npz = `${values.out.replace(/\/$/, "")}/array_4d.npz`;
      rememberOtherInputs(patch);
    }
    localStorage.setItem(memoryKey, JSON.stringify(values));
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stage: stage.id, values, overwrite }),
    });
    const data = await res.json();
    if (data.needsOverwrite) {
      setAskOverwrite(true);
      return;
    }
    if (data.busy) {
      return;
    }
    if (data.error) {
      setError(data.error);
      return;
    }
    watch(data.job);
  }

  return (
    <div className="animate-rise">
      <Breadcrumb trail={[{ label: "Pipeline", href: "/pipeline" }, { label: stage.phase, href: "/pipeline" }, { label: stage.title }]} />

      {siblings.length > 1 && (
        <div className="mb-6 flex flex-wrap gap-1.5">
          {siblings.map((s) => (
            <Link
              key={s.id}
              href={`/stage/${s.id}`}
              className={`rounded-full border px-3 py-1 text-[12.5px] transition ${
                s.id === stage.id ? "border-teal-500/40 bg-teal-500/10 text-teal-300" : "border-neutral-800 text-neutral-500 hover:text-neutral-300"
              }`}
            >
              {s.title}
            </Link>
          ))}
        </div>
      )}

      {saved && (
        <div className="mb-6 flex flex-wrap items-center gap-3 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
          <span className="text-sm text-neutral-300">Use the previous files for this step?</span>
          <button type="button" onClick={() => { setValues(saved); setSaved(null); }} className="rounded-md bg-teal-500 px-3 py-1.5 text-sm font-medium text-neutral-950">
            Previous files
          </button>
          <button type="button" onClick={() => setSaved(null)} className="rounded-md border border-white/10 px-3 py-1.5 text-sm text-neutral-300">
            New data
          </button>
        </div>
      )}

      <h1 className="text-3xl font-semibold tracking-tight text-neutral-50">{stage.title}</h1>
      <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-neutral-400">{stage.note || "Choose the inputs, then run."}</p>

      <div className="mt-8 max-w-2xl">
        {datasets.length > 0 && (
          <RecordingSelect choice={choice} datasets={datasets} animals={animals} onChange={setChoice} />
        )}
        {dataset && (
          <div className="mb-5 rounded-xl border border-teal-500/20 bg-teal-500/[0.06] px-4 py-3 text-sm text-neutral-300">
            <div className="font-medium text-teal-200">
              {dataset.animal} · {dataset.label}
            </div>
            <p className="mt-1 break-all font-mono text-[12px] text-neutral-400">Outputs save in {outputLabel(dataset.workspace.root)}</p>
            {STAGE_HINT[stage.id] && <p className="mt-2 text-[13px] text-neutral-400">{STAGE_HINT[stage.id]}</p>}
          </div>
        )}
        {choice === "other" && (
          <div className="mb-5 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-neutral-400">
            Files you choose here are filled in on the next steps. Change any field when that step needs a different file.
          </div>
        )}
        {(stage.id === "fourier" || stage.id === "sustained_transient") && choice && (
          <div className="mb-5">
            <label className="mb-1.5 block text-sm text-neutral-400">Contrast</label>
            <div className="flex flex-wrap gap-1.5">
              {["c0", "c50", "c60", "c70", "c80", "c90"].map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => {
                    setContrast(item);
                    setField("cond", withContrast(values.cond, item));
                  }}
                  className={`rounded-full border px-3 py-1 text-[12.5px] ${
                    contrast === item ? "border-teal-500/40 bg-teal-500/10 text-teal-200" : "border-neutral-800 text-neutral-400"
                  }`}
                >
                  {item.replace("c", "C ")} %
                </button>
              ))}
            </div>
          </div>
        )}
        {stage.fields.map((f) => (
          <PathField key={f.key} label={f.label} kind={f.kind} filter={f.filter} value={values[f.key]} onChange={(v) => setField(f.key, v)} />
        ))}

        {askOverwrite && (
          <div className="mb-4 rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-3 text-sm text-amber-100">
            <p>Results are already in this folder. Overwrite them?</p>
            <div className="mt-3 flex gap-2">
              <button type="button" onClick={() => run(true)} className="rounded-md bg-amber-300 px-3 py-1.5 text-sm font-medium text-neutral-950">
                Overwrite
              </button>
              <button type="button" onClick={() => setAskOverwrite(false)} className="rounded-md border border-white/15 px-3 py-1.5 text-sm text-neutral-200">
                Keep them
              </button>
            </div>
          </div>
        )}

        {error && <div className="mb-4 rounded-md border border-rose-500/25 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">{error}</div>}

        <button
          onClick={() => run(false)}
          disabled={status === "running"}
          className="rounded-md bg-teal-500 px-6 py-3 text-[15px] font-medium text-neutral-950 transition hover:bg-teal-400 disabled:opacity-50"
        >
          {status === "running" ? "Running…" : `Run ${stage.title}`}
        </button>
      </div>

      <Console lines={lines} status={status} onClear={reset} />
    </div>
  );
}
