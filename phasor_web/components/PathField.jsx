"use client";

import { useEffect, useState } from "react";

function shortFolder(value) {
  if (!value) return "";
  const marker = "phasor_output";
  const index = value.indexOf(marker);
  if (index >= 0) return value.slice(index);
  const parts = value.split("/").filter(Boolean);
  return parts.slice(-2).join("/");
}

function matchesFilter(entry, filter) {
  if (!filter || entry.is_dir) return true;
  const name = entry.name.toLowerCase();
  return filter
    .toLowerCase()
    .split(",")
    .some((ext) => name.endsWith(ext.trim()));
}

export default function PathField({ label, kind = "file", value, onChange, filter, placeholder }) {
  const [open, setOpen] = useState(false);
  const [listing, setListing] = useState(null);
  const [showHidden, setShowHidden] = useState(false);

  useEffect(() => {
    if (open) load(value && kind === "dir" ? value : "");
  }, [open, showHidden]);

  async function load(path) {
    const params = new URLSearchParams({ path: path || "", want: "any", hidden: showHidden ? "1" : "0" });
    const res = await fetch(`/api/browse?${params}`);
    setListing(await res.json());
  }

  function pick(entry) {
    if (entry.is_dir) {
      if (kind === "dir") onChange(entry.path);
      load(entry.path);
      return;
    }
    if (kind === "dir") return;
    onChange(entry.path);
    setOpen(false);
  }

  const entries = (listing?.entries || []).filter((e) => (kind === "dir" ? e.is_dir : matchesFilter(e, filter)));
  const marker = value ? value.indexOf("phasor_output") : -1;
  const shown = marker >= 0 ? value.slice(marker) : value ? value.split("/").filter(Boolean).pop() : "";

  return (
    <div className="mb-4">
      <label className="mb-1.5 block text-sm text-neutral-400">{label}</label>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="flex w-full items-center justify-between rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2.5 text-left hover:border-teal-500/40"
      >
        <span className={`truncate font-mono text-[13.5px] ${shown ? "text-neutral-100" : "text-neutral-600"}`}>
          {shown || placeholder || (kind === "dir" ? "Choose a folder" : "Choose a file")}
        </span>
        <span className="ml-3 shrink-0 text-xs text-teal-300">Browse</span>
      </button>

      {open && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-6" onClick={() => setOpen(false)}>
          <div
            className="flex max-h-[80vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-white/10 bg-neutral-950 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 border-b border-white/10 px-4 py-3">
              <button
                type="button"
                disabled={!listing?.parent}
                onClick={() => listing?.parent && load(listing.parent)}
                className="rounded px-2 py-1 text-neutral-400 hover:bg-white/5 disabled:opacity-30"
              >
                ↑
              </button>
              {(listing?.shortcuts || []).slice(0, 5).map((s) => (
                <button
                  key={s.path}
                  type="button"
                  onClick={() => load(s.path)}
                  className="rounded-full border border-white/10 px-2.5 py-0.5 text-xs text-neutral-300 hover:border-teal-500/40"
                >
                  {s.label}
                </button>
              ))}
              <span className="flex-1" />
              <label className="flex items-center gap-1.5 text-xs text-neutral-500">
                <input type="checkbox" checked={showHidden} onChange={(e) => setShowHidden(e.target.checked)} />
                hidden
              </label>
            </div>
            <div className="min-h-0 flex-1 overflow-auto py-1">
              {entries.map((entry) => (
                <button
                  key={entry.path}
                  type="button"
                  onClick={() => pick(entry)}
                  className="flex w-full items-center gap-2 px-4 py-2 text-left font-mono text-[13px] text-neutral-200 hover:bg-white/5"
                >
                  <span className={entry.is_dir ? "text-teal-400" : "text-neutral-600"}>{entry.is_dir ? "▸" : "·"}</span>
                  <span className="truncate">{entry.name}</span>
                </button>
              ))}
              {entries.length === 0 && <div className="px-4 py-8 text-center text-sm text-neutral-500">Nothing to choose here</div>}
            </div>
            <div className="flex items-center justify-between gap-3 border-t border-white/10 px-4 py-3">
              <span className="truncate font-mono text-[11px] text-neutral-500">{shortFolder(listing?.cwd)}</span>
              <div className="flex gap-2">
                {kind === "dir" && listing?.cwd && (
                  <button
                    type="button"
                    onClick={() => {
                      onChange(listing.cwd);
                      setOpen(false);
                    }}
                    className="rounded-md bg-teal-500 px-3 py-1.5 text-sm font-medium text-neutral-950"
                  >
                    Use this folder
                  </button>
                )}
                <button type="button" onClick={() => setOpen(false)} className="rounded-md border border-white/10 px-3 py-1.5 text-sm text-neutral-300">
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
