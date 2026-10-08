"use client";

import { useRecordingCatalog } from "./RecordingPicker";

export default function RecordingStrip() {
  const { animals, datasets, choice, setChoice, dataset } = useRecordingCatalog();
  if (!datasets.length) return null;
  return (
    <div className="mt-8 rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
      <div className="mb-3 font-mono text-[11px] uppercase tracking-[0.16em] text-neutral-500">Active recording</div>
      <div className="flex flex-wrap gap-2">
        {animals.map((animal) =>
          datasets
            .filter((row) => row.animal === animal)
            .map((row) => (
              <button
                key={row.id}
                type="button"
                onClick={() => setChoice(row.id)}
                className={`rounded-full border px-3 py-1.5 text-sm transition ${
                  choice === row.id
                    ? "border-teal-400/50 bg-teal-400/15 text-teal-50"
                    : "border-white/10 text-neutral-400 hover:border-white/20 hover:text-neutral-200"
                }`}
              >
                {row.animal} {row.label}
              </button>
            ))
        )}
        <button
          type="button"
          onClick={() => setChoice("other")}
          className={`rounded-full border px-3 py-1.5 text-sm transition ${
            choice === "other" ? "border-teal-400/50 bg-teal-400/15 text-teal-50" : "border-white/10 text-neutral-400 hover:text-neutral-200"
          }`}
        >
          Other files
        </button>
      </div>
      {dataset && (
        <p className="mt-3 break-all font-mono text-[12px] text-neutral-500">Saves to {dataset.workspace.root}</p>
      )}
    </div>
  );
}
