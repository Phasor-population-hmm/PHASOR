"use client";

import { useEffect, useState } from "react";
import PathField from "./PathField";
import Console from "./Console";
import RecordingSelect, { analysisFolderName, outputLabel, prepareRecording, readOtherInputs, readOutputRoot, readRecordingManual, rememberOtherInputs, rememberRecordingFolder, useRecordingCatalog, writeOtherInputs } from "./RecordingPicker";
import { chooseOutputFolder, runLayout } from "@/lib/layout";
import { ensureFree } from "./JobDock";
import { PLOT_NOTES, PLOTS } from "@/lib/stages";
import { useJobPoll } from "@/hooks/useJobPoll";

const MEMORY = "phasor-analysis-files";

export default function PlotsRunner() {
  const [bundle, setBundle] = useState("");
  const [output, setOutput] = useState("");
  const [runDir, setRunDir] = useState("");
  const [rep, setRep] = useState(1);
  const [cycle, setCycle] = useState(3);
  const [error, setError] = useState("");
  const [askOverwrite, setAskOverwrite] = useState(null);
  const [activeId, setActiveId] = useState("");
  const { lines, status, watch, reset } = useJobPoll();
  const [jobId, setJobId] = useState("");
  const { datasets, animals, choice, setChoice, dataset } = useRecordingCatalog();

  useEffect(() => {
    if (!choice) return;
    if (choice === "other") {
      const saved = readOtherInputs();
      const layout = runLayout(saved.workspaceRoot || "");
      setBundle(saved.bundle || "");
      setRunDir(saved.manual?.["plots:runDir"] || layout.root || "");
      setOutput(saved.manual?.["plots:out"] || layout.analysis || "");
      return;
    }
    if (!dataset) return;
    const custom = readOutputRoot(dataset.id);
    const layout = custom ? runLayout(custom) : null;
    const manual = readRecordingManual(dataset.id);
    setBundle(dataset.files.bundle || "");
    setOutput(manual["plots:out"] || layout?.analysis || dataset.workspace.analysis || "");
    setRunDir(manual["plots:runDir"] || layout?.root || dataset.workspace.root || dataset.dirs.data || "");
  }, [choice, dataset]);

  function remember(nextBundle, nextOutput) {
    localStorage.setItem(MEMORY, JSON.stringify({ bundle: nextBundle, output: nextOutput }));
  }

  useEffect(() => {
    fetch("/api/session")
      .then((res) => res.json())
      .then((session) => {
        if (session.status === "running" && session.kind === "job" && session.href === "/plots" && session.id) {
          setJobId(session.id);
          watch(session.id);
        }
      })
      .catch(() => {});
  }, [watch]);

  async function run(chosenPlot, overwrite = false) {
    setError("");
    setAskOverwrite(null);
    if (!bundle) {
      setError("Choose a bundle.npz file.");
      return;
    }
    if (!output) {
      setError("Choose a folder to save the results.");
      return;
    }
    if (!(await ensureFree())) return;
    if (dataset) await prepareRecording(dataset.id);
    remember(bundle, output);
    if (choice === "other") rememberOtherInputs({ bundle, runDir, analysisOut: output });
    const title = PLOTS.find((row) => row[0] === chosenPlot)?.[1];
    const folder = chosenPlot === "viterbi_polar_rep_cycle" ? `polar_rep${rep}_cycle${cycle}` : analysisFolderName(chosenPlot, title);
    const destination = `${output}/${folder}`;
    const res = await fetch("/api/plots", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bundles: [{ bundle, run_dir: runDir || undefined, name: dataset?.id || readOtherInputs().workspaceRoot?.split("/").pop() }],
        plot: chosenPlot,
        output: destination,
        rep,
        cycle,
        overwrite,
      }),
    });
    const data = await res.json();
    if (data.needsOverwrite) {
      setAskOverwrite(chosenPlot);
      return;
    }
    if (data.busy) return;
    if (data.error) {
      setError(data.error);
      return;
    }
    setActiveId(chosenPlot);
    setJobId(data.job);
    watch(data.job);
  }

  async function terminate() {
    if (!jobId) return;
    await fetch(`/api/job/${jobId}`, { method: "POST" });
  }

  return (
    <div className="animate-rise">
      <h1 className="text-3xl font-semibold tracking-tight text-neutral-50">Post-HMM plots & analysis</h1>
      <p className="mt-3 max-w-2xl text-[15px] text-neutral-400">
        Each analysis reads the decoded model and writes its figures in a separate folder.
      </p>

      <div className="mt-8 max-w-5xl">
        <RecordingSelect
          choice={choice}
          datasets={datasets}
          animals={animals}
          onChange={setChoice}
          hint="A listed recording fills the bundle and the results folder."
        />
        {(dataset || output) && (
          <div className="mb-5 rounded-xl border border-teal-500/20 bg-teal-500/[0.06] px-4 py-3 text-sm text-neutral-300">
            Each analysis is saved under {outputLabel(output || dataset?.workspace.analysis)}
          </div>
        )}
        <PathField label="Bundle file" kind="file" filter=".npz" value={bundle} onChange={(value) => { setBundle(value); if (choice === "other") rememberOtherInputs({ bundle: value, manual: { "file:bundle": value } }); }} placeholder="Choose bundle.npz" />
        {choice === "other" && (
          <PathField label="Recording folder" kind="dir" value={runDir} onChange={(value) => {
            const saved = writeOtherInputs(chooseOutputFolder(readOtherInputs(), "plots", "runDir", value, "c90"));
            const layout = runLayout(saved.workspaceRoot);
            setRunDir(saved.manual?.["plots:runDir"] || layout.root);
            setOutput(saved.manual?.["plots:out"] || layout.analysis);
          }} placeholder="Folder that holds 01_preprocessing, 02_processing, and the later stages" />
        )}
        <PathField label="Results folder" kind="dir" value={output} onChange={(value) => {
          if (choice === "other") {
            const saved = writeOtherInputs(chooseOutputFolder(readOtherInputs(), "plots", "out", value, "c90"));
            const layout = runLayout(saved.workspaceRoot);
            setOutput(saved.manual?.["plots:out"] || layout.analysis);
            setRunDir(saved.manual?.["plots:runDir"] || layout.root);
            if (!saved.manual?.["file:bundle"] && saved.bundle) setBundle(saved.bundle);
            return;
          }
          if (dataset) {
            const saved = rememberRecordingFolder(dataset.id, "plots", "out", value, "c90");
            setOutput(saved.output || saved.analysis);
            setRunDir(saved.runDir || runDir);
            return;
          }
          setOutput(value);
        }} placeholder="Choose where to save results" />

        <div className="mb-3 rounded-xl border border-white/10 bg-white/[0.03] p-4">
          <div className="font-mono text-[11px] uppercase tracking-wide text-neutral-500">Repetition and cycle</div>
          <p className="mt-1 text-sm text-neutral-400">Polar by repetition and cycle uses these. The other analyses do not.</p>
          <div className="mt-3 flex flex-wrap gap-4">
            <label className="text-sm text-neutral-300">
              Rep
              <select value={rep} onChange={(event) => setRep(Number(event.target.value))} className="ml-2 rounded-md border border-neutral-800 bg-neutral-900 px-2 py-1">
                {Array.from({ length: 15 }, (_, index) => index + 1).map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>
            <label className="text-sm text-neutral-300">
              Cycle
              <select value={cycle} onChange={(event) => setCycle(Number(event.target.value))} className="ml-2 rounded-md border border-neutral-800 bg-neutral-900 px-2 py-1">
                {[1, 2, 3, 4, 5, 6].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </label>
          </div>
        </div>

        {status === "running" && (
          <div className="mb-3 flex flex-wrap items-center gap-2 rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-2 text-sm text-amber-100">
            <span className="h-2 w-2 animate-pulse rounded-full bg-amber-300" />
            {PLOTS.find((row) => row[0] === activeId)?.[1] || "This analysis"} is running.
            <button type="button" onClick={terminate} className="ml-auto rounded-md bg-rose-500 px-3 py-1.5 text-sm font-medium text-white">
              Terminate
            </button>
          </div>
        )}
        <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
          {PLOTS.map(([id, title, cover]) => (
            <button
              key={id}
              type="button"
              onClick={() => run(id, false)}
              disabled={status === "running"}
              className="group overflow-hidden rounded-2xl border border-white/10 bg-neutral-950 text-left transition hover:-translate-y-0.5 hover:border-teal-400/50 disabled:opacity-50"
            >
              <span className="relative block aspect-square overflow-hidden bg-[#101418]">
                <img src={`${cover}?v=2`} alt="" className="h-full w-full object-cover transition duration-300 group-hover:scale-105" />
                {status === "running" && activeId === id && (
                  <span className="absolute left-3 top-3 rounded-full bg-black/70 px-2.5 py-1 font-mono text-[11px] uppercase tracking-wide text-amber-200">
                    Running
                  </span>
                )}
                <span className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/90 via-black/70 to-transparent p-3 text-white">
                  <span className="block text-[14px] font-medium leading-snug">{title}</span>
                  <span className="mt-1 block text-[11px] leading-snug text-neutral-200">{PLOT_NOTES[id]}</span>
                </span>
              </span>
            </button>
          ))}
        </div>
        <p className="text-xs text-neutral-500">Select a square to run that analysis.</p>
        {askOverwrite && (
          <div className="mt-4 rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-3 text-sm text-amber-100">
            <p>Results are already in this folder. Overwrite them?</p>
            <div className="mt-3 flex gap-2">
              <button type="button" onClick={() => run(askOverwrite, true)} className="rounded-md bg-amber-300 px-3 py-1.5 text-sm font-medium text-neutral-950">
                Overwrite
              </button>
              <button type="button" onClick={() => setAskOverwrite(null)} className="rounded-md border border-white/15 px-3 py-1.5 text-sm text-neutral-200">
                Keep them
              </button>
            </div>
          </div>
        )}
        {error && <div className="mt-4 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">{error}</div>}
      </div>

      <Console lines={lines} status={status} onClear={reset} />
    </div>
  );
}
