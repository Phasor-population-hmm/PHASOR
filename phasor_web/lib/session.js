import fs from "fs";
import path from "path";
import { OUTPUT_ROOT, displayPath } from "./paths";

const FILE = path.join(OUTPUT_ROOT, ".active_job.json");

function pidAlive(pid) {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export function readSession() {
  try {
    return JSON.parse(fs.readFileSync(FILE, "utf8"));
  } catch {
    return null;
  }
}

function writeSession(session) {
  fs.mkdirSync(OUTPUT_ROOT, { recursive: true });
  fs.writeFileSync(FILE, JSON.stringify(session, null, 2));
}

export function claim(session) {
  writeSession({
    ...session,
    status: "running",
    acked: false,
    started: Date.now(),
    finished: null,
  });
}

export function finish(patch) {
  const current = readSession() || {};
  if (patch.id && current.id && patch.id !== current.id) return;
  writeSession({
    ...current,
    ...patch,
    finished: Date.now(),
    acked: false,
  });
}

export function ackSession() {
  const current = readSession();
  if (!current) return null;
  current.acked = true;
  writeSession(current);
  return current;
}

function hmmStatus(dataDir) {
  if (!dataDir) return {};
  const statusPath = path.join(dataDir, "phasor_hmm_results", "phasor_hmm_status.json");
  try {
    return JSON.parse(fs.readFileSync(statusPath, "utf8"));
  } catch {
    return {};
  }
}

export function runningSession() {
  const current = readSession();
  if (!current || current.status !== "running") return null;
  if (current.pid && !pidAlive(current.pid)) return null;
  return current;
}

export function publicSession() {
  let current = readSession();
  if (current && current.status === "running" && current.kind === "hmm" && current.pid && !pidAlive(current.pid)) {
    const status = hmmStatus(current.data);
    const ok = status.status === "complete";
    finish({ status: ok ? "done" : "failed", message: status.message || (ok ? "" : "The Hidden Markov Model run stopped.") });
    current = readSession();
  }
  if (!current) return { status: "idle" };
  return {
    id: current.id || "",
    label: current.label || "Job",
    kind: current.kind || "job",
    status: current.status || "idle",
    stage: current.stage || "",
    href: current.href || "/",
    output: current.output ? displayPath(current.output) : "",
    data: current.data || "",
    pid: current.pid || null,
    message: current.message || "",
    acked: Boolean(current.acked),
  };
}
