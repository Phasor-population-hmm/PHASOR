"use client";

import { useEffect, useRef, useState } from "react";
import PathField from "./PathField";
import Console from "./Console";
import HmmStepper from "./HmmStepper";
import Breadcrumb from "./Breadcrumb";
import StateGraph from "./visuals/StateGraph";
import RecordingSelect, { outputLabel, prepareRecording, readOtherInputs, readOutputRoot, readRecordingManual, rememberOtherInputs, rememberRecordingFolder, useRecordingCatalog, writeOtherInputs } from "./RecordingPicker";
import { chooseOutputFolder, runLayout } from "@/lib/layout";
import { ensureFree } from "./JobDock";
import { DEFAULT_HMM_SEARCH, HMM_ETA_CHOICES, HMM_K_CHOICES, formatSearchNumber, normalizeSearch } from "@/lib/hmmSearch";

const SEARCH_MEMORY = "phasor-hmm-search";

function loadSearch() {
  if (typeof window === "undefined") return { ...DEFAULT_HMM_SEARCH, k: [...DEFAULT_HMM_SEARCH.k], alpha: [...DEFAULT_HMM_SEARCH.alpha], eta: [...DEFAULT_HMM_SEARCH.eta] };
  try {
    return normalizeSearch(JSON.parse(localStorage.getItem(SEARCH_MEMORY) || "null") || {});
  } catch {
    return { k: [...DEFAULT_HMM_SEARCH.k], alpha: [...DEFAULT_HMM_SEARCH.alpha], eta: [...DEFAULT_HMM_SEARCH.eta] };
  }
}

function SearchSettings({ value, onChange }) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState({ k: "", eta: "" });
  const [note, setNote] = useState("");

  function toggle(kind, item) {
    const current = value[kind];
    const has = current.some((entry) => entry === item);
    const next = has ? current.filter((entry) => entry !== item) : [...current, item];
    try {
      const parsed = normalizeSearch({ ...value, [kind]: next });
      onChange(parsed);
      setNote("");
    } catch (err) {
      setNote(err.message);
    }
  }

  function addCustom(kind) {
    if (!String(draft[kind]).trim()) {
      setNote("Enter a value to add");
      return;
    }
    try {
      const parsed = normalizeSearch({ ...value, [kind]: [...value[kind], draft[kind]] });
      onChange(parsed);
      setDraft((prev) => ({ ...prev, [kind]: "" }));
      setNote("");
    } catch (err) {
      setNote(err.message);
    }
  }

  const groups = [
    { kind: "k", label: "K", choices: HMM_K_CHOICES, hint: "Number of hidden states. Any whole number of 2 or more." },
    { kind: "eta", label: "eta", choices: HMM_ETA_CHOICES, hint: "Covariance soft-threshold. Greater than 0 and at most 1." },
  ];

  return (
    <div className="mb-5 rounded-xl border border-neutral-800 bg-neutral-950">
      <button type="button" onClick={() => setOpen((prev) => !prev)} className="flex w-full items-center justify-between px-4 py-3 text-left">
        <span>
          <span className="block text-sm text-neutral-200">Search settings</span>
          <span className="mt-1 block font-mono text-[12px] text-neutral-500">
            K {value.k.join(", ")}   alpha 0.5   eta {value.eta.map(formatSearchNumber).join(", ")}
          </span>
        </span>
        <span className="text-sm text-neutral-400">{open ? "Hide" : "Change"}</span>
      </button>
      {open && (
        <div className="border-t border-neutral-800 px-4 py-4">
          <p className="text-sm leading-relaxed text-neutral-400">
            Highlighted values are used for the next fit. Alpha stays at 0.5. The standard search is K 3, 6, 9, 12, and 15, and eta 0.0005, 0.002, and 0.005. Each selected eta is searched at alpha 0.5, and every selected K is tried inside that search. Refit keeps the combination the search prefers.
          </p>
          {groups.map((group) => {
            const extras = value[group.kind].filter((item) => !group.choices.includes(item));
            const shown = [...group.choices, ...extras];
            return (
              <div key={group.kind} className="mt-4">
                <div className="text-sm text-neutral-200">{group.label}</div>
                <p className="mt-1 text-xs text-neutral-500">{group.hint}</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {shown.map((item) => {
                    const on = value[group.kind].includes(item);
                    return (
                      <button
                        key={formatSearchNumber(item)}
                        type="button"
                        onClick={() => toggle(group.kind, item)}
                        className={on ? "rounded-md border border-teal-500/40 bg-teal-500/15 px-3 py-1.5 font-mono text-[13px] text-teal-100" : "rounded-md border border-neutral-800 px-3 py-1.5 font-mono text-[13px] text-neutral-400"}
                      >
                        {formatSearchNumber(item)}
                      </button>
                    );
                  })}
                </div>
                <div className="mt-2 flex gap-2">
                  <input
                    value={draft[group.kind]}
                    onChange={(event) => setDraft((prev) => ({ ...prev, [group.kind]: event.target.value }))}
                    placeholder="Add a value"
                    className="w-36 rounded-md border border-neutral-800 bg-neutral-900 px-3 py-1.5 font-mono text-[13px] text-neutral-100"
                  />
                  <button type="button" onClick={() => addCustom(group.kind)} className="rounded-md border border-neutral-800 px-3 py-1.5 text-sm text-neutral-300">
                    Add
                  </button>
                </div>
              </div>
            );
          })}
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm text-neutral-400">
              {value.eta.length} searches at alpha 0.5, with {value.k.length} values of K in each. Up to 3 searches run at the same time.
            </p>
            <button
              type="button"
              onClick={() => {
                onChange({ k: [...DEFAULT_HMM_SEARCH.k], alpha: [...DEFAULT_HMM_SEARCH.alpha], eta: [...DEFAULT_HMM_SEARCH.eta] });
                setNote("");
              }}
              className="text-sm text-neutral-400 underline decoration-neutral-700"
            >
              Use the default values
            </button>
          </div>
          {note && <p className="mt-3 text-sm text-rose-300">{note}</p>}
        </div>
      )}
    </div>
  );
}

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
  const [search, setSearch] = useState({
    k: [...DEFAULT_HMM_SEARCH.k],
    alpha: [...DEFAULT_HMM_SEARCH.alpha],
    eta: [...DEFAULT_HMM_SEARCH.eta],
  });
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
    setSearch(loadSearch());
  }, []);

  function updateSearch(next) {
    setSearch(next);
    localStorage.setItem(SEARCH_MEMORY, JSON.stringify(next));
  }

  useEffect(() => {
    fetch("/api/hmm/scan")
      .then((r) => r.json())
      .then((data) => setScanInfo(data));
    return () => pollRef.current && clearInterval(pollRef.current);
  }, []);

  useEffect(() => {
    if (!choice) return;
    if (choice === "other") {
      const saved = readOtherInputs();
      const layout = runLayout(saved.workspaceRoot || "");
      setDataDir(saved.manual?.["hmm:out"] || layout.hmm || "");
      setInputs({
        spikes: saved.spikes || "",
        synctones: saved.synctones || "",
        stimulus: saved.stimulus || saved.stims || "",
      });
      return;
    }
    if (!dataset) return;
    const custom = readOutputRoot(dataset.id);
    const layout = custom ? runLayout(custom) : null;
    const manual = readRecordingManual(dataset.id);
    setDataDir(manual["hmm:out"] || layout?.hmm || dataset.workspace.hmm || "");
    setInputs({
      spikes: dataset.files.spikes || "",
      synctones: dataset.files.synctones || "",
      stimulus: dataset.files.stims || "",
    });
  }, [choice, dataset]);

  function setDataFolder(value) {
    if (choice === "other") {
      const saved = writeOtherInputs(chooseOutputFolder(readOtherInputs(), "hmm", "out", value, "c90"));
      setDataDir(saved.manual?.["hmm:out"] || runLayout(saved.workspaceRoot).hmm);
      return;
    }
    if (dataset) {
      const saved = rememberRecordingFolder(dataset.id, "hmm", "out", value, "c90");
      setDataDir(saved.output || saved.hmm);
      return;
    }
    setDataDir(value);
  }

  function setInput(key, value) {
    setInputs((prev) => ({ ...prev, [key]: value }));
    if (choice !== "other") return;
    const patch = { [key]: value, manual: { [`file:${key}`]: value } };
    if (key === "stimulus") {
      patch.stims = value;
      patch.manual["file:stims"] = value;
    }
    rememberOtherInputs(patch);
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
        out.push("", "The fit stopped. The console above has the error.");
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
      body: JSON.stringify({ data: dataDir, chain: {}, inputs, search }),
    });
    const r = await res.json();
    setLines([]);
    if (r.error) {
      setStatus("failed");
      setError(r.error);
      return;
    }
    if (r.already) {
      setLines(["Search, refit, and analysis are already saved.", "  " + (r.saved || dataDir + "/results_run1")]);
      setStatus("done");
      setStep("done");
      return;
    }
    if (r.skipped) {
      setLines(["bundle.npz already present, skipping the HMM fit.", "  " + r.bundle, "", "Ready. Open the Viterbi plots page to render results."]);
      setStatus("done");
      setStep("done");
      return;
    }
    const used = r.search || search;
    const resumeNote = {
      analysis: "Search and refit are already saved. Analysis continues from those results.",
      refit: "Search results are already saved. Refit continues from those results.",
      search: "HMM started on the selected recording.",
    }[r.resume] || "HMM started on the selected recording.";
    setLines([
      resumeNote,
      "It uses " + (r.cores || 3) + " cores.",
      "K " + used.k.join(", "),
      "alpha " + used.alpha.map(formatSearchNumber).join(", "),
      "eta " + used.eta.map(formatSearchNumber).join(", "),
      "  results   " + dataDir + "/results_run1",
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
            Fits a population hidden Markov model to the selected spike file, synctone times, and stimulus file.
            Alpha stays at 0.5. The default search uses K of 3, 6, 9, 12, and 15, and eta 0.0005, 0.002, and 0.005.
            Search settings can use other K and eta values. The fit then refits the preferred model and decodes the latent state.
            Search tables, the refit model, and the figures are saved in 05_hmm/results_run1.
            An existing bundle.npz is reused.
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
          hint="A listed recording fills the spike, synctone, and stimulus files."
        />
        {dataset?.files?.bundle && (
          <div className="mb-5 rounded-xl border border-teal-500/20 bg-teal-500/[0.06] px-4 py-3">
            <div className="text-sm font-medium text-teal-200">HMM results are already available for this recording</div>
            <p className="mt-1 text-sm text-neutral-400">A new fit is saved in {outputLabel(dataset.workspace.hmm)}.</p>
            <button type="button" onClick={useExisting} className="mt-3 rounded-md border border-teal-500/30 px-3 py-1.5 text-sm text-teal-100">
              Use existing results
            </button>
          </div>
        )}
        {choice === "other" && (
          <div className="mb-5 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-neutral-400">
            The folder chosen on an earlier step is kept. The fit is saved in 05_hmm.
          </div>
        )}
        <PathField label="HMM output folder" kind="dir" value={dataDir} onChange={setDataFolder} />
        <button onClick={detect} className="-mt-2 mb-6 rounded-md border border-neutral-800 px-4 py-2 text-sm text-neutral-300 hover:border-neutral-700">
          Detect HMM inputs
        </button>

        {INPUT_FIELDS.map((f) => (
          <PathField key={f.key} label={f.label} kind={f.kind} value={inputs[f.key]} onChange={(v) => setInput(f.key, v)} />
        ))}

        <SearchSettings value={search} onChange={updateSearch} />

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
                The fit runs on this computer and uses 3 cores. Progress is shown in the console below.
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
                The selected eta values run at alpha 0.5, up to 3 at a time, then the preferred model is refit and decoded.
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
