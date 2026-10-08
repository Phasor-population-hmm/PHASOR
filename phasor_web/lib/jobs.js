import { spawn } from "child_process";
import { randomBytes } from "crypto";
import fs from "fs";
import path from "path";
import { APP_ROOT, OUTPUT_ROOT, VITERBI_BANDS, PIPELINE } from "./paths";
import { claim, finish, runningSession } from "./session";

const jobs = globalThis.__phasorJobs || (globalThis.__phasorJobs = new Map());

class Job {
  constructor(label) {
    this.id = randomBytes(6).toString("hex");
    this.label = label;
    this.status = "queued";
    this.log = [];
    this.returncode = null;
    this.started = Date.now();
    this.finished = null;
  }

  line(text) {
    for (const part of text.split(/\r?\n/)) {
      if (part.length) this.log.push(part);
    }
  }

  snapshot() {
    return {
      id: this.id,
      label: this.label,
      status: this.status,
      returncode: this.returncode,
      log: this.log,
      elapsed: Math.round(((this.finished || Date.now()) - this.started) / 100) / 10,
    };
  }
}

function baseEnv() {
  return {
    ...process.env,
    PHASOR_OUTPUT_ROOT: OUTPUT_ROOT,
    PHASOR_VITERBI_BANDS: VITERBI_BANDS,
    PYTHONUNBUFFERED: "1",
    MPLBACKEND: "Agg",
  };
}

export function buildArgv(stage, values, pythonPath) {
  const tokens = stage.template.split(" ").filter(Boolean);
  return tokens
    .map((tok) => {
      if (tok === "{python}") return pythonPath;
      if (tok === "{script}") return path.join(PIPELINE, stage.script);
      const m = tok.match(/^\{(\w+)\}$/);
      if (m) return values[m[1]] || "";
      return tok;
    })
    .filter((t) => t !== "");
}

export function launch(label, argv, opts = {}) {
  if (runningSession()) {
    const error = new Error("busy");
    error.busy = true;
    throw error;
  }
  const job = new Job(label);
  jobs.set(job.id, job);
  fs.mkdirSync(OUTPUT_ROOT, { recursive: true });

  job.status = "running";
  job.output = opts.output || "";
  job.line("$ " + argv.map((a) => (a.includes(" ") ? `"${a}"` : a)).join(" "));

  const child = spawn(argv[0], argv.slice(1), {
    cwd: opts.cwd || APP_ROOT,
    env: { ...baseEnv(), ...(opts.extraEnv || {}) },
    detached: true,
  });
  job.pid = child.pid;
  claim({
    id: job.id,
    label,
    kind: "job",
    pid: child.pid,
    output: opts.output || "",
    href: opts.href || "/",
    stage: opts.stage || "",
  });

  child.stdout.on("data", (chunk) => job.line(chunk.toString()));
  child.stderr.on("data", (chunk) => job.line(chunk.toString()));
  child.on("error", (err) => {
    job.line(`[runner error] ${err.message}`);
    job.returncode = -1;
    job.status = "failed";
    job.finished = Date.now();
    finish({ id: job.id, status: "failed", message: err.message });
  });
  child.on("close", (code) => {
    job.returncode = code;
    job.status = code === 0 ? "done" : "failed";
    job.finished = Date.now();
    finish({ id: job.id, status: job.status, message: code === 0 ? "" : `Exited with code ${code}` });
  });
  child.unref();

  return job;
}

export function stopJob(id) {
  const job = jobs.get(id);
  if (!job?.pid) return false;
  try {
    process.kill(-job.pid, "SIGTERM");
  } catch {
    try {
      process.kill(job.pid, "SIGTERM");
    } catch {
      return false;
    }
  }
  job.line("Terminated.");
  job.status = "failed";
  job.finished = Date.now();
  finish({ id: job.id, status: "failed", message: "Terminated." });
  return true;
}

export function getJob(id) {
  return jobs.get(id) || null;
}
