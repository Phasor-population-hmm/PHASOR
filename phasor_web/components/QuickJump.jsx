"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { STAGES } from "@/lib/stages";

const ENTRIES = [
  { label: "Overview", href: "/", group: "Console" },
  { label: "Pipeline", href: "/pipeline", group: "Console" },
  ...STAGES.map((s) => ({ label: s.title, href: s.id === "hmm" ? "/hmm" : `/stage/${s.id}`, group: s.phase })),
  { label: "Viterbi plots", href: "/plots", group: "Reporting" },
];

export default function QuickJump() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState(0);
  const inputRef = useRef(null);
  const rootRef = useRef(null);
  const router = useRouter();

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return ENTRIES.slice(0, 8);
    return ENTRIES.filter((e) => e.label.toLowerCase().includes(q) || e.group.toLowerCase().includes(q));
  }, [query]);

  useEffect(() => {
    function onKey(e) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
        setOpen(true);
      }
      if (e.key === "Escape") setOpen(false);
    }
    function onClickAway(e) {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onClickAway);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onClickAway);
    };
  }, []);

  useEffect(() => setIndex(0), [query]);

  function go(entry) {
    if (!entry) return;
    router.push(entry.href);
    setOpen(false);
    setQuery("");
    inputRef.current?.blur();
  }

  function onKeyDown(e) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setIndex((i) => Math.min(i + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      go(results[index]);
    }
  }

  return (
    <div ref={rootRef} className="relative w-64">
      <div className="flex items-center gap-2 rounded-md border border-neutral-800 bg-neutral-900/60 px-3 py-1.5 focus-within:border-teal-500/50">
        <svg width="14" height="14" viewBox="0 0 20 20" fill="none" className="shrink-0 text-neutral-500">
          <circle cx="9" cy="9" r="6" stroke="currentColor" strokeWidth="1.6" />
          <path d="M14 14l4 4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        <input
          ref={inputRef}
          value={query}
          onFocus={() => setOpen(true)}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Jump to…"
          className="w-full bg-transparent text-[13.5px] text-neutral-200 outline-none placeholder:text-neutral-600"
        />
        <kbd className="shrink-0 rounded border border-neutral-800 px-1 py-0.5 font-mono text-[10px] text-neutral-600">⌘K</kbd>
      </div>

      {open && (
        <div className="absolute right-0 top-[calc(100%+8px)] z-40 w-80 overflow-hidden rounded-lg border border-neutral-800 bg-neutral-900 py-1.5 shadow-[0_20px_50px_-12px_rgba(0,0,0,0.6)]">
          {results.length === 0 && <div className="px-4 py-4 text-center text-sm text-neutral-600">no matches</div>}
          {results.map((entry, i) => (
            <button
              key={entry.href}
              onClick={() => go(entry)}
              onMouseEnter={() => setIndex(i)}
              className={`flex w-full items-center justify-between px-4 py-2 text-left text-[13.5px] ${
                i === index ? "bg-neutral-800 text-neutral-100" : "text-neutral-300"
              }`}
            >
              <span>{entry.label}</span>
              <span className="font-mono text-[10px] uppercase tracking-wide text-neutral-600">{entry.group}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
