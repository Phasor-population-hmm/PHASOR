"use client";

import { useEffect, useRef } from "react";

function classify(line) {
  if (line.startsWith("$ ")) return "text-teal-400";
  if (/traceback|error|failed|exception/i.test(line)) return "text-rose-400";
  if (/complete|ready|bundle\.npz is ready/i.test(line)) return "text-emerald-400";
  if (/^HMM |detached|profile |workspace /.test(line)) return "text-amber-400";
  return "text-neutral-300";
}

export default function Console({ lines, status = "idle", onClear }) {
  const boxRef = useRef(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    const el = boxRef.current;
    if (!el) return;
    if (stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [lines]);

  function handleScroll() {
    const el = boxRef.current;
    if (!el) return;
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }

  function text() {
    return lines.join("\n");
  }

  async function copy() {
    await navigator.clipboard.writeText(text());
  }

  function save() {
    const blob = new Blob([text() + "\n"], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "phasor-console.log";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <div className="mt-6 overflow-hidden rounded-lg border border-neutral-900 bg-black">
      <div className="flex items-center gap-2 border-b border-neutral-900 bg-neutral-950 px-3 py-2">
        <span className="flex-1 font-mono text-[11px] uppercase tracking-widest text-neutral-500">console</span>
        <StatusPill status={status} />
        <button className="rounded border border-neutral-800 px-2 py-1 font-mono text-[11px] text-neutral-500 hover:text-neutral-200" onClick={copy}>
          copy
        </button>
        <button className="rounded border border-neutral-800 px-2 py-1 font-mono text-[11px] text-neutral-500 hover:text-neutral-200" onClick={save}>
          save
        </button>
        <button className="rounded border border-neutral-800 px-2 py-1 font-mono text-[11px] text-neutral-500 hover:text-neutral-200" onClick={onClear}>
          clear
        </button>
      </div>
      <div
        ref={boxRef}
        onScroll={handleScroll}
        className="max-h-[480px] min-h-[220px] overflow-auto px-5 py-4 font-mono text-[13.5px] leading-relaxed"
      >
        {lines.length === 0 ? (
          <div className="text-neutral-600">console idle.</div>
        ) : (
          lines.map((line, i) => (
            <div key={i} className={`whitespace-pre-wrap break-words ${classify(line)}`}>
              {line}
            </div>
          ))
        )}
        {status === "running" && <span className="inline-block h-4 w-2 translate-y-0.5 animate-blink bg-teal-400/80" />}
      </div>
    </div>
  );
}

function StatusPill({ status }) {
  if (status === "idle") return null;
  const styles = { running: "text-amber-400", done: "text-emerald-400", failed: "text-rose-400" };
  const dot = { running: "bg-amber-400 animate-pulse", done: "bg-emerald-400", failed: "bg-rose-400" };
  return (
    <span className={`flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider ${styles[status] || "text-neutral-500"}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${dot[status] || "bg-neutral-600"}`} />
      {status}
    </span>
  );
}
