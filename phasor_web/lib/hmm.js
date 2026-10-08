import { execFileSync, spawn } from "child_process";
import fs from "fs";
import path from "path";
import { randomBytes } from "crypto";
import { APP_ROOT, MODE_PROJECT_DEFAULT, HMM_CONFIG_PATH } from "./paths";
import { pipelinePython } from "./pythonPath";
import { displayPath } from "./paths";
import { claim, finish, readSession, runningSession } from "./session";

const SEARCH_ETA = [0.0005, 0.002, 0.005];
const SEARCH_ALPHA = [0.5];

export function loadConfig() {
  if (fs.existsSync(HMM_CONFIG_PATH)) {
    try {
      return JSON.parse(fs.readFileSync(HMM_CONFIG_PATH, "utf8"));
    } catch {
      return {};
    }
  }
  return {};
}

function saveConfig(cfg) {
  fs.writeFileSync(HMM_CONFIG_PATH, JSON.stringify(cfg, null, 2));
}

export function projectDir() {
  const raw = loadConfig().project_dir || MODE_PROJECT_DEFAULT;
  return path.isAbsolute(raw) ? raw : path.join(APP_ROOT, raw);
}

export function pythonExecutable() {
  const configured = loadConfig().python || "";
  if (configured && fs.existsSync(configured)) return configured;
  return pipelinePython();
}

function validBundle(bundlePath) {
  const script =
    "import numpy as np, sys\n" +
    "try:\n" +
    "    with np.load(sys.argv[1], allow_pickle=False) as d:\n" +
    "        keys = set(d.files)\n" +
    '        if {"modes_train","modes_test"}.issubset(keys):\n' +
    '            ok = d["modes_train"].size > 0 and d["modes_test"].size > 0\n' +
    "        else:\n" +
    '            ok = "modes" in keys and d["modes"].size > 0\n' +
    '        print("1" if ok else "0")\n' +
    "except Exception:\n" +
    '    print("0")\n';
  try {
    const out = execFileSync(pythonExecutable(), ["-c", script, bundlePath], { encoding: "utf8" });
    return out.trim() === "1";
  } catch {
    return false;
  }
}

function walkGlob(root, pattern) {
  // supports the small set of globs used here: "*bundle*.npz", "dir/*bundle*.npz", "results*/final_analysis/*bundle*.npz"
  const parts = pattern.split("/");
  let dirs = [root];
  for (let i = 0; i < parts.length - 1; i++) {
    const seg = parts[i];
    const next = [];
    for (const d of dirs) {
      if (!fs.existsSync(d)) continue;
      for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
        if (entry.isDirectory() && matchSegment(entry.name, seg)) next.push(path.join(d, entry.name));
      }
    }
    dirs = next;
  }
  const filePattern = parts[parts.length - 1];
  const results = [];
  for (const d of dirs) {
    if (!fs.existsSync(d)) continue;
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      if (entry.isFile() && matchSegment(entry.name, filePattern)) results.push(path.join(d, entry.name));
    }
  }
  return results.sort();
}

function matchSegment(name, seg) {
  const re = new RegExp("^" + seg.split("*").map(escapeRe).join(".*") + "$");
  return re.test(name);
}
function escapeRe(s) {
  return s.replace(/[.+?^${}()|[\]\\]/g, "\\$&");
}

export function findBundle(dataDir) {
  const root = dataDir;
  if (!fs.existsSync(root)) return null;
  const candidates = [
    ...walkGlob(root, "*bundle*.npz"),
    ...walkGlob(root, "phasor_hmm_results/*bundle*.npz"),
    ...walkGlob(root, "final_analysis/*bundle*.npz"),
    ...walkGlob(root, "results*/final_analysis/*bundle*.npz"),
  ];
  return candidates.find(validBundle) || null;
}

function datasetKeyFromScript(scriptPath) {
  const text = fs.readFileSync(scriptPath, "utf8");
  const patterns = [/PROJECT_ROOT\s*\/\s*"data"\s*\/\s*"([^"]+)"/, /PROJECT_ROOT\s*\/\s*'data'\s*\/\s*'([^']+)'/];
  for (const re of patterns) {
    const m = text.match(re);
    if (m) return m[1];
  }
  return null;
}

function profileFor(folder) {
  const scripts = fs
    .readdirSync(folder, { withFileTypes: true })
    .filter((e) => e.isFile() && e.name.endsWith(".py") && !e.name.startsWith("test_") && !e.name.startsWith("__"))
    .map((e) => path.join(folder, e.name));
  const models = scripts.filter((s) => path.basename(s).toLowerCase().includes("model")).sort();
  const refits = scripts.filter((s) => path.basename(s).toLowerCase().includes("refit")).sort();
  const analyses = scripts
    .filter((s) => {
      const n = path.basename(s).toLowerCase();
      return n.includes("analysis") || n.includes("analyze");
    })
    .sort();
  if (!models.length || !refits.length || !analyses.length) return null;
  const datasetKey = datasetKeyFromScript(models[0]);
  if (!datasetKey) return null;
  const launcher = path.join(folder, "launch_9_jobs.sh");
  return {
    id: path.relative(projectDir(), folder),
    name: path.basename(folder),
    directory: folder,
    models: models[0],
    refit: refits[0],
    analysis: analyses[0],
    launcher: fs.existsSync(launcher) ? launcher : "",
    dataset_key: datasetKey,
  };
}

export function scan() {
  const root = projectDir();
  const cfg = loadConfig();
  const python = pythonExecutable();
  if (!fs.existsSync(root)) {
    return {
      project_dir: displayPath(root),
      python: displayPath(python),
      exists: false,
      profiles: [],
      last_profile: cfg.last_profile || "",
    };
  }
  const profiles = [];
  const testRoot = path.join(root, "test");
  if (fs.existsSync(testRoot)) {
    const folders = fs
      .readdirSync(testRoot, { withFileTypes: true })
      .filter((e) => e.isDirectory() && !e.name.startsWith(".") && !e.name.startsWith("_"))
      .map((e) => path.join(testRoot, e.name))
      .sort();
    for (const folder of folders) {
      const profile = profileFor(folder);
      if (profile) profiles.push(profile);
    }
  }
  return {
    project_dir: displayPath(root),
    python: displayPath(python),
    exists: true,
    profiles,
    last_profile: cfg.last_profile || "",
  };
}

export function saveChain(chain) {
  const cfg = loadConfig();
  cfg.last_profile = chain.profile || "";
  saveConfig(cfg);
}

function firstByDepth(paths) {
  if (!paths.length) return null;
  return [...paths].sort((a, b) => {
    const da = a.split(path.sep).length;
    const db = b.split(path.sep).length;
    return da - db || a.localeCompare(b);
  })[0];
}

function walkFiles(root) {
  const out = [];
  const stack = [root];
  while (stack.length) {
    const dir = stack.pop();
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of entries) {
      const full = path.join(dir, e.name);
      if (e.isDirectory()) stack.push(full);
      else if (e.isFile()) out.push(full);
    }
  }
  return out;
}

export function discoverInputs(dataDir) {
  const root = path.resolve(dataDir);
  if (!fs.existsSync(root)) return { error: `Dataset folder not found: ${root}` };

  const files = walkFiles(root);
  const stim = firstByDepth(files.filter((f) => f.toLowerCase().endsWith(".json") && path.basename(f).toLowerCase().includes("ffsine")));
  const sync = firstByDepth(files.filter((f) => f.toLowerCase().endsWith(".csv") && path.basename(f).toLowerCase().includes("synctone")));
  const spikeCandidates = files.filter((f) => {
    const base = path.basename(f).toLowerCase();
    return f.toLowerCase().endsWith(".txt") && !base.includes("log") && !base.includes("summary");
  });
  const spikes = firstByDepth(spikeCandidates.filter((f) => path.basename(f).toLowerCase().includes("spike"))) || firstByDepth(spikeCandidates);

  return {
    data_dir: root,
    stimulus: stim || "",
    synctones: sync || "",
    spikes: spikes || "",
    bundle: findBundle(root) || "",
  };
}

function copyInput(source, destination) {
  fs.mkdirSync(path.dirname(destination), { recursive: true });
  if (fs.existsSync(destination) && fs.statSync(destination).size === fs.statSync(source).size) return;
  fs.copyFileSync(source, destination);
}

function patchScript(source, destination, oldKey, newKey) {
  let text = fs.readFileSync(source, "utf8");
  if (oldKey) {
    text = text.split(`"${oldKey}"`).join(`"${newKey}"`).split(`'${oldKey}'`).join(`'${newKey}'`);
  }
  fs.writeFileSync(destination, text);
}

function profileById(profileId) {
  const info = scan();
  return info.profiles.find((p) => p.id === profileId || p.directory === profileId) || null;
}

function shellScript(workspace, staged, outputDir, profile, maxParallel) {
  const python = pythonExecutable();
  const models = path.join(workspace, path.basename(profile.models));
  const refit = path.join(workspace, path.basename(profile.refit));
  const analysis = path.join(workspace, path.basename(profile.analysis));
  const results = path.join(staged, "results_run1");
  const bundle = path.join(results, "final_analysis", "bundle.npz");
  const logDir = path.join(workspace, "logs");
  const copiedBundle = path.join(outputDir, "bundle.npz");
  const status = path.join(outputDir, "phasor_hmm_status.json");
  const eta = SEARCH_ETA.join(" ");
  const alpha = SEARCH_ALPHA.join(" ");

  return `#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${projectDir()}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p "${logDir}" "${results}" "${outputDir}"
on_error() {
  code=$?
  printf '{"status":"failed","step":"pipeline","exit_code":%s,"updated":"%s"}\\n' "$code" "$(date -Iseconds)" > "${status}"
  exit "$code"
}
trap on_error ERR
printf '{"status":"running","step":"search","updated":"%s"}\\n' "$(date -Iseconds)" > "${status}"

search_pids=()
for eta in ${eta}; do
  for alpha in ${alpha}; do
    csv="${results}/chow_eta\${eta}_alpha\${alpha}.csv"
    if [ -s "$csv" ]; then
      continue
    fi
    "${python}" -u "${models}" "$eta" "$alpha" > "${logDir}/search_eta\${eta}_alpha\${alpha}.log" 2>&1 &
    search_pids+=("$!")
    while [ "$(jobs -pr | wc -l)" -ge "${maxParallel}" ]; do
      sleep 2
    done
  done
done
if [ "\${#search_pids[@]}" -gt 0 ]; then
  for pid in "\${search_pids[@]}"; do
    wait "$pid"
  done
fi

printf '{"status":"running","step":"refit","updated":"%s"}\\n' "$(date -Iseconds)" > "${status}"
if ! compgen -G "${results}/chowliu_K*.npz" > /dev/null; then
  "${python}" -u "${refit}" > "${logDir}/refit.log" 2>&1
fi

printf '{"status":"running","step":"analysis","updated":"%s"}\\n' "$(date -Iseconds)" > "${status}"
if [ ! -s "${bundle}" ]; then
  "${python}" -u "${analysis}" > "${logDir}/analysis.log" 2>&1
fi

if [ ! -s "${bundle}" ]; then
  printf '{"status":"failed","step":"analysis","message":"bundle.npz was not created","updated":"%s"}\\n' "$(date -Iseconds)" > "${status}"
  exit 1
fi
cp -f "${bundle}" "${copiedBundle}"
printf '{"status":"complete","step":"done","bundle":"${copiedBundle}","updated":"%s"}\\n' "$(date -Iseconds)" > "${status}"
`;
}

export function start(dataDir, chain, inputs) {
  if (runningSession()) return { error: "Already one job or operation in process.", busy: true };
  const resolvedData = path.resolve(dataDir);
  if (!fs.existsSync(resolvedData)) return { error: `Dataset folder not found: ${resolvedData}` };

  const bundle = findBundle(resolvedData);
  if (bundle) return { skipped: true, bundle };

  const profiles = scan().profiles;
  const profile = profileById((chain || {}).profile || "") || profiles[0] || null;
  if (!profile) return { error: "HMM scripts are not installed on this computer." };

  const discovered = discoverInputs(resolvedData);
  const supplied = inputs || {};
  const selected = {};
  for (const key of ["spikes", "synctones", "stimulus"]) {
    const v = supplied[key] || discovered[key] || "";
    selected[key] = v;
  }
  const missing = Object.entries(selected).filter(([, v]) => !v || !fs.existsSync(v) || !fs.statSync(v).isFile());
  if (missing.length) {
    return { error: "Missing HMM input: " + missing.map(([k]) => k).join(", "), inputs: discovered };
  }

  const root = projectDir();
  const runId = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14) + "_" + randomBytes(3).toString("hex");
  const datasetKey = `phasor_${runId}`;
  const staged = path.join(root, "data", datasetKey);
  const workspace = path.join(root, "test", `_phasor_${runId}`);
  const outputDir = path.join(resolvedData, "phasor_hmm_results");
  fs.mkdirSync(workspace, { recursive: true });
  fs.mkdirSync(path.join(staged, "run1"), { recursive: true });
  fs.mkdirSync(outputDir, { recursive: true });

  copyInput(selected.spikes, path.join(staged, "run1", "spikes.txt"));
  copyInput(selected.synctones, path.join(staged, "run1", "synctones.csv"));
  copyInput(selected.stimulus, path.join(staged, "FFsine_2Hz_stims.json"));

  const oldKey = datasetKeyFromScript(profile.models);
  for (const role of ["models", "refit", "analysis"]) {
    const source = profile[role];
    patchScript(source, path.join(workspace, path.basename(source)), oldKey, datasetKey);
  }

  const cfg = loadConfig();
  const maxParallel = Math.max(1, parseInt(cfg.max_parallel_searches || 3, 10));
  const runScript = path.join(workspace, "run_pipeline.sh");
  fs.writeFileSync(runScript, shellScript(workspace, staged, outputDir, profile, maxParallel));
  fs.chmodSync(runScript, 0o755);

  const logPath = path.join(outputDir, "phasor_hmm.log");
  fs.writeFileSync(
    logPath,
    `PHASOR HMM run\nstarted: ${new Date().toISOString()}\nprofile: ${profile.id}\ndata: ${resolvedData}\nworkspace: ${workspace}\n\n`
  );
  const logFd = fs.openSync(logPath, "a");
  const child = spawn("/bin/bash", [runScript], {
    cwd: workspace,
    stdio: ["ignore", logFd, logFd],
    detached: true,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  child.unref();
  fs.closeSync(logFd);

  claim({
    id: `hmm-${child.pid}`,
    label: "Hidden Markov Model",
    kind: "hmm",
    pid: child.pid,
    output: outputDir,
    data: resolvedData,
    href: "/hmm",
    stage: "hmm",
  });
  saveChain({ profile: profile.id });
  return {
    started: true,
    pid: child.pid,
    log: logPath,
    cores: maxParallel,
    local: true,
    workspace,
    inputs: selected,
    started_at: new Date().toISOString(),
  };
}

export function stop(dataDir) {
  const session = readSession();
  const pid = session?.kind === "hmm" ? session.pid : null;
  if (pid) {
    try {
      process.kill(-pid, "SIGTERM");
    } catch {
      try {
        process.kill(pid, "SIGTERM");
      } catch {
        /* already gone */
      }
    }
  }
  const root = dataDir ? path.resolve(dataDir) : session?.data;
  if (root) {
    const statusPath = path.join(root, "phasor_hmm_results", "phasor_hmm_status.json");
    try {
      fs.mkdirSync(path.dirname(statusPath), { recursive: true });
      fs.writeFileSync(
        statusPath,
        JSON.stringify({ status: "cancelled", step: "cancelled", message: "Terminated.", updated: new Date().toISOString() })
      );
    } catch {
      /* status file is optional */
    }
  }
  finish({ status: "failed", message: "Terminated." });
  return { stopped: true };
}

export function tail(dataDir, n = 120) {
  const root = path.resolve(dataDir);
  const output = path.join(root, "phasor_hmm_results");
  const logPath = path.join(output, "phasor_hmm.log");
  const statusPath = path.join(output, "phasor_hmm_status.json");
  let status = {};
  if (fs.existsSync(statusPath)) {
    try {
      status = JSON.parse(fs.readFileSync(statusPath, "utf8"));
    } catch {
      status = {};
    }
  }
  const exists = fs.existsSync(logPath);
  const lines = exists ? fs.readFileSync(logPath, "utf8").split("\n").slice(-n) : [];
  const bundle = findBundle(root);
  return { lines, exists, bundle: bundle || "", status };
}
