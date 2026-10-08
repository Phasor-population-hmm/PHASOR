"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

export function announceBusy(session) {
  window.dispatchEvent(new CustomEvent("phasor-busy", { detail: session }));
}

export async function ensureFree() {
  const session = await fetch("/api/session").then((res) => res.json());
  if (session.status === "running") {
    announceBusy(session);
    return false;
  }
  return true;
}

export default function JobDock() {
  const [session, setSession] = useState(null);
  const [busy, setBusy] = useState(null);
  const [done, setDone] = useState(null);
  const previous = useRef(null);

  useEffect(() => {
    let stop = false;
    async function tick() {
      const next = await fetch("/api/session").then((res) => res.json()).catch(() => null);
      if (stop || !next) return;
      const prior = previous.current;
      const finished = next.status && next.status !== "running" && next.status !== "idle" && !next.acked && next.label;
      if (finished && (!prior || prior.status === "running")) {
        setDone(next);
      }
      previous.current = next;
      setSession(next.status === "running" ? next : null);
    }
    tick();
    const timer = setInterval(tick, 1000);
    function onBusy(event) {
      setBusy(event.detail || { label: "A job" });
    }
    window.addEventListener("phasor-busy", onBusy);
    return () => {
      stop = true;
      clearInterval(timer);
      window.removeEventListener("phasor-busy", onBusy);
    };
  }, []);

  async function acknowledge() {
    await fetch("/api/session", { method: "POST" });
    setDone(null);
  }

  async function terminate() {
    if (!session || session.kind !== "hmm") return;
    await fetch("/api/hmm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "stop", data: session.data }),
    });
  }

  return (
    <>
      {session && (
        <div className="fixed inset-x-0 bottom-0 z-40 border-t border-teal-500/30 bg-neutral-950/95 px-6 py-3 backdrop-blur">
          <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-3">
            <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400" />
            <span className="text-sm text-neutral-100">{session.label} is running on this computer.</span>
            {session.output && <span className="truncate font-mono text-[12px] text-neutral-500">{session.output}</span>}
            <span className="flex-1" />
            <Link href={session.href || "/"} className="rounded-md border border-white/15 px-3 py-1.5 text-sm text-neutral-100">
              Open
            </Link>
            {session.kind === "hmm" && (
              <button type="button" onClick={terminate} className="rounded-md bg-rose-500 px-3 py-1.5 text-sm font-medium text-white">
                Terminate
              </button>
            )}
          </div>
        </div>
      )}

      {busy && (
        <Modal
          title="Already one job or operation in process"
          body={`${busy.label || "A job"} is still running. Wait for it to finish, or terminate the Hidden Markov Model run if that is the one in progress.`}
          onClose={() => setBusy(null)}
        />
      )}

      {done && (
        <Modal
          title={done.status === "done" ? `${done.label} is done` : `${done.label} stopped`}
          body={done.output ? `The files are in ${done.output}` : "The run has stopped."}
          onClose={acknowledge}
        />
      )}
    </>
  );
}

function Modal({ title, body, action, onClose }) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-6">
      <div className="relative w-full max-w-2xl overflow-hidden rounded-2xl border border-white/10 bg-neutral-950 p-6 pr-14 shadow-2xl">
        <button
          type="button"
          onClick={onClose}
          aria-label="Close"
          className="absolute right-3 top-3 grid h-8 w-8 place-items-center rounded-full text-lg text-neutral-400 hover:bg-white/10 hover:text-white"
        >
          ×
        </button>
        <h2 className="break-words text-lg font-medium leading-snug text-neutral-50">{title}</h2>
        <p className="mt-3 break-all text-sm leading-relaxed text-neutral-300">{body}</p>
        {action && <div className="mt-5 flex justify-end">{action}</div>}
      </div>
    </div>
  );
}
