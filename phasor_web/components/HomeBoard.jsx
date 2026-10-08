"use client";

import Link from "next/link";
import { outputLabel, useRecordingCatalog } from "./RecordingPicker";

const STEPS = [
  { href: "/stage/locate_synctones", label: "Recording" },
  { href: "/stage/filter", label: "Processing" },
  { href: "/stage/fourier", label: "Fourier" },
  { href: "/hmm", label: "Hidden Markov Model" },
  { href: "/plots", label: "Post-HMM plots" },
];

export default function HomeBoard() {
  const { datasets, animals, choice, setChoice, dataset } = useRecordingCatalog();
  if (!datasets.length) return null;

  return (
    <div className="mt-12">
      <div className="mb-4 flex items-end justify-between gap-4">
        <div>
          <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-teal-400">Choose a recording</div>
          <p className="mt-1 max-w-xl text-sm text-neutral-500">
            Pick WT1, WT2, WT3, or RD1 and every page loads that recording. Other files clears the preloaded paths so you can browse a different dataset.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {animals.map((animal) => {
          const rows = datasets.filter((row) => row.animal === animal);
          const active = rows.some((row) => row.id === choice);
          return (
            <div
              key={animal}
              className={`rounded-2xl border p-4 transition ${
                active ? "border-teal-500/40 bg-teal-500/[0.07]" : "border-white/[0.07] bg-white/[0.02] hover:border-white/15"
              }`}
            >
              <div className="font-mono text-xs uppercase tracking-[0.16em] text-neutral-400">{animal}</div>
              <div className="mt-3 flex flex-col gap-1.5">
                {rows.map((row) => {
                  const on = choice === row.id;
                  return (
                    <button
                      key={row.id}
                      type="button"
                      onClick={() => setChoice(row.id)}
                      className={`rounded-lg border px-3 py-2 text-left text-sm transition ${
                        on
                          ? "border-teal-400/50 bg-teal-400/15 text-teal-50"
                          : "border-transparent text-neutral-300 hover:border-white/10 hover:bg-white/[0.04]"
                      }`}
                    >
                      {row.label}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      <button
        type="button"
        onClick={() => setChoice("other")}
        className={`mt-3 w-full rounded-2xl border px-4 py-3 text-left text-sm transition ${
          choice === "other" ? "border-teal-500/40 bg-teal-500/10 text-teal-100" : "border-white/[0.07] text-neutral-400 hover:border-white/15"
        }`}
      >
        Other files. Browse a dataset that is not preloaded.
      </button>

      {dataset && (
        <div className="mt-4 rounded-2xl border border-white/[0.08] bg-neutral-950 p-5">
          <div className="text-lg font-medium text-neutral-100">
            {dataset.animal} · {dataset.label}
          </div>
          <p className="mt-1 text-sm text-neutral-500">The .h5, spike file, stimulus, and HMM bundle are loaded. New results are saved in this recording’s folder.</p>
          <div className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-2">
            <FileLine label="Recording" value={dataset.files.h5} />
            <FileLine label="Output folder" value={outputLabel(dataset.workspace.root)} />
          </div>
          <div className="mt-5 flex flex-wrap gap-2">
            {STEPS.map((step, index) => (
              <Link
                key={step.href}
                href={step.href}
                className="group flex items-center gap-2 rounded-full border border-white/10 px-3 py-1.5 text-sm text-neutral-200 transition hover:border-teal-500/40 hover:text-teal-100"
              >
                <span className="grid h-5 w-5 place-items-center rounded-full bg-teal-500/15 font-mono text-[11px] text-teal-300">{index + 1}</span>
                {step.label}
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function FileLine({ label, value }) {
  const name = value && value.includes("phasor_output") ? value : value ? value.split("/").pop() : "";
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
      <div className="font-mono text-[10px] uppercase tracking-wide text-neutral-500">{label}</div>
      <div className="mt-1 truncate text-sm text-neutral-200" title={value}>
        {name || value}
      </div>
    </div>
  );
}
