"use client";

import { useEffect, useRef, useState } from "react";
import PathField from "./PathField";
import Console from "./Console";
import HmmStepper from "./HmmStepper";
import Breadcrumb from "./Breadcrumb";
import StateGraph from "./visuals/StateGraph";
import RecordingSelect, { outputLabel, prepareRecording, useRecordingCatalog } from "./RecordingPicker";
import { ensureFree } from "./JobDock";

const INPUT_FIELDS = [
  { key: "spikes", label: "Spike times (.txt)", kind: "file" },
  { key: "synctones", label: "Synctones (.csv)", kind: "file" },
  { key: "stimulus", label: "Stimulus definition (.json)", kind: "file" },
];

export default function HmmRunner() {
  const [dataDir, setDataDir] = useState("");
  const [inputs, setInputs] = useState({ spikes: "", synctones: "", stimulus: "" });
  const [scanInfo, setScanInfo] = useState(null);
  const [lines, setLines] = useState([]);
  const [status, setStatus] = useState("idle");
  const [step, setStep] = useState("");
  const [error, setError] = useState("");
  const pollRef = useRef(null);
  const { datasets, animals, choice, setChoice, dataset } = useRecordingCatalog();

  useEffect(() => {
    fetch("/api/session")
      .then((res) => res.json())
      .then((session) => {
        if (session.status === "running" && session.kind === "hmm" && session.data) {
          setDataDir(session.data);
          setStatus("running");
          setStep(session.stage === "hmm" ? "search" : "");
          watchLog(session.data);
        }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetch("/api/hmm/scan")
      .then((r) => r.json())
      .then((data) => setScanInfo(data));
    return () => pollRef.current && clearInterval(pollRef.current);
  }, []);

  useEffect(() => {
    if (!choice) return;
    if (choice === "other") {
      setDataDir("");
      setInputs({ spikes: "", synctones: "", stimulus: "" });
      return;
    }
    if (!dataset) return;
    setDataDir(dataset.workspace.hmm || "");
    setInputs({
      spikes: dataset.files.spikes || "",
      synctones: dataset.files.synctones || "",
      stimulus: dataset.files.stims || "",
    });
  }, [choice, dataset]);

  function setInput(key, value) {
    setInputs((prev) => ({ ...prev, [key]: value }));
  }

  async function detect() {
    setError("");
    if (!dataDir) {
      setError("Select a dataset folder first.");
      return;
    }
    const res = await fetch(`/api/hmm/discover?data=${encodeURIComponent(dataDir)}`);
    const found = await res.json();
    if (found.error) {
      setError(found.error);
      return;
    }
    setInputs({ spikes: found.spikes || "", synctones: found.synctones || "", stimulus: found.stimulus || "" });
    setLines([]);
    if (found.bundle) {
      setLines(["Valid bundle found. HMM fitting will be skipped.", found.bundle]);
      setStatus("done");
    } else {
      setLines(["HMM inputs detected.", ...INPUT_FIELDS.map((f) => `  ${f.key.padEnd(10)} ${found[f.key] || "not found"}`)]);
    }
  }

  function watchLog(folder = dataDir) {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      const res = await fetch(`/api/hmm/log?data=${encodeURIComponent(folder)}`);
      const j = await res.json();
      if (!j.exists) return;
      const out = [];
      if (j.status?.step) {
        setStep(j.status.step);
        out.push(`HMM ${j.status.status || "running"} · ${j.status.step}`);
      }
      out.push(...j.lines);
      setLines(out);

      if (j.status?.status === "cancelled") {
        out.push("", "Terminated. The Hidden Markov Model run has stopped.");
        setLines(out);
        setStatus("failed");
        clearInterval(pollRef.current);
        pollRef.current = null;
        return;
      }
      if (j.status?.status === "failed") {
        out.push("", "HMM workflow failed. Check the search, refit, and analysis logs in the workspace.");
        setLines(out);
        setStatus("failed");
        clearInterval(pollRef.current);
        pollRef.current = null;
        return;
      }
      if (j.bundle) {
        out.push("", "bundle.npz is ready.", j.bundle);
        setLines(out);
        setStatus("done");
        setStep("done");
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }, 3000);
  }

  function useExisting() {
    if (!dataset?.files?.bundle) {
      setError("This recording has no HMM results yet.");
      return;
    }
    setError("");
    setStatus("done");
    setStep("done");
    setLines([
      "Existing HMM results loaded. Nothing was refit.",
      "  bundle  " + dataset.files.bundle,
      "",
      "Open Analysis to plot this recording. A new fit is only started if you click Start HMM.",
    ]);
  }

  async function terminateHmm() {
    await fetch("/api/hmm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "stop", data: dataDir }),
    });
    setStatus("failed");
    setLines((prev) => [...prev, "", "Terminated."]);
  }

  async function startHmm() {
    setError("");
    if (!(await ensureFree())) return;
    if (dataset) await prepareRecording(dataset.id);
    const res = await fetch("/api/hmm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: dataDir, chain: {}, inputs }),
    });
    const r = await res.json();
    setLines([]);
    if (r.error) {
      setStatus("failed");
      setError(r.error);
      return;
    }
    if (r.skipped) {
      setLines(["bundle.npz already present, skipping the HMM fit.", "  " + r.bundle, "", "Ready. Open the Viterbi plots page to render results."]);
      setStatus("done");
      setStep("done");
      return;
    }
    setLines([
      "HMM started on this computer, in a terminal process.",
      "It uses " + (r.cores || 3) + " cores.",
      "  pid       " + r.pid,
      "  log       " + r.log,
      "  results   " + dataDir,
      "",
      "You can leave this page. The run continues in the terminal.",
    ]);
    setStatus("running");
    setStep("search");
    watchLog();
  }

  return (
    <div className="animate-rise">
      <Breadcrumb trail={[{ label: "Pipeline", href: "/pipeline" }, { label: "HMM fit (Viterbi decode)" }]} />

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[1fr_260px] lg:items-center">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-neutral-50">Hidden Markov Model</h1>
          <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-neutral-400">
            Chow–Liu tree emissions with a covariance soft-threshold. Search uses η ∈ {"{"}0.0005, 0.002, 0.005{"}"} at
            α = 0.5, then refits K ∈ {"{"}3, 6, 9, 12, 15{"}"} and decodes with Viterbi. A valid bundle.npz skips fitting.
          </p>
        </div>
        <StateGraph className="hidden h-[150px] w-full text-neutral-600 lg:block" />
      </div>

      <div className="mt-8 max-w-2xl">
        <RecordingSelect
          choice={choice}
          datasets={datasets}
          animals={animals}
          onChange={setChoice}
          hint="Preloaded recordings fill the spike, synctone, and stimulus files. Other files means you browse them yourself."
        />
        {dataset?.files?.bundle && (
          <div className="mb-5 rounded-xl border border-teal-500/20 bg-teal-500/[0.06] px-4 py-3">
            <div className="text-sm font-medium text-teal-200">HMM results are already available for this recording</div>
            <p className="mt-1 text-sm text-neutral-400">Existing results are loaded for this recording. A new fit is saved in {outputLabel(dataset.workspace.hmm)}.</p>
            <button type="button" onClick={useExisting} className="mt-3 rounded-md border border-teal-500/30 px-3 py-1.5 text-sm text-teal-100">
              Use existing results
            </button>
          </div>
        )}
        {choice === "other" && (
          <div className="mb-5 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-neutral-400">
            Browse the dataset folder and the three input files, then start the fit.
          </div>
        )}
        <PathField label="HMM output folder" kind="dir" value={dataDir} onChange={setDataDir} />
        <button onClick={detect} className="-mt-2 mb-6 rounded-md border border-neutral-800 px-4 py-2 text-sm text-neutral-300 hover:border-neutral-700">
          Detect HMM inputs
        </button>

        {INPUT_FIELDS.map((f) => (
          <PathField key={f.key} label={f.label} kind={f.kind} value={inputs[f.key]} onChange={(v) => setInput(f.key, v)} />
        ))}

        <div className="mt-2 rounded-xl border border-neutral-900 bg-neutral-950 p-5">
          {!scanInfo ? (
            <div className="text-sm text-neutral-500">Scanning HMM project…</div>
          ) : !scanInfo.exists ? (
            <div className="text-sm leading-relaxed">
              <span className="text-rose-400">HMM project not found.</span>
              <br />
              Configure <code className="font-mono text-teal-300">project_dir</code> in{" "}
              <code className="font-mono">phasor_web/hmm.config.json</code>.
            </div>
          ) : (
            <>
              <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-100">
                This runs on this computer, in a terminal process. It needs 3 cores.
              </div>
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div>
                  <div className="font-mono text-[11px] uppercase tracking-wide text-neutral-500">Project</div>
                  <div className="mt-1 truncate font-mono text-[12.5px] text-neutral-300" title={scanInfo.project_dir}>
                    {scanInfo.project_dir}
                  </div>
                </div>
                <div>
                  <div className="font-mono text-[11px] uppercase tracking-wide text-neutral-500">Python</div>
                  <div className="mt-1 truncate font-mono text-[12.5px] text-neutral-300" title={scanInfo.python}>
                    {scanInfo.python}
                  </div>
                </div>
              </div>
              <p className="mt-4 text-[13px] text-neutral-500">
                Three η searches run in parallel, then the selected model is refit and decoded. Finished search tables
                are reused if the run is interrupted.
              </p>
            </>
          )}
        </div>

        {error && <div className="mt-4 rounded-md border border-rose-500/25 bg-rose-500/10 px-3 py-2 text-sm text-rose-300">{error}</div>}

        <div className="mt-6 flex flex-wrap gap-2">
          <button
            onClick={startHmm}
            disabled={status === "running"}
            className="rounded-md bg-teal-500 px-6 py-3 text-[15px] font-medium text-neutral-950 transition hover:bg-teal-400 disabled:opacity-50"
          >
            {status === "running" ? "Running…" : "Start HMM"}
          </button>
          {status === "running" && (
            <button type="button" onClick={terminateHmm} className="rounded-md bg-rose-500 px-6 py-3 text-[15px] font-medium text-white">
              Terminate
            </button>
          )}
        </div>
      </div>

      {status !== "idle" && (
        <div className="mt-8 max-w-2xl rounded-xl border border-neutral-900 bg-neutral-950 p-6">
          <HmmStepper current={step} failed={status === "failed"} />
        </div>
      )}

      <Console lines={lines} status={status} onClear={() => setLines([])} />
    </div>
  );
}
