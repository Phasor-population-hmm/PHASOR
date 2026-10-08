import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { PLOTS } from "@/lib/stages";
import { PIPELINE, OUTPUT_ROOT } from "@/lib/paths";
import { launch } from "@/lib/jobs";
import { pipelinePython } from "@/lib/pythonPath";

const PLOT_IDS = new Set(PLOTS.map((row) => row[0]));

export async function POST(request) {
  const body = await request.json();
  const bundles = body.bundles || [];
  const plotId = body.plot;
  const outputDir = body.output ? path.resolve(body.output) : OUTPUT_ROOT;
  if (!bundles.length) return NextResponse.json({ error: "choose a bundle.npz file first" }, { status: 400 });
  if (!PLOT_IDS.has(plotId)) return NextResponse.json({ error: "unknown plot" }, { status: 400 });

  const datasets = [];
  for (let item of bundles) {
    if (typeof item === "string") item = { bundle: item };
    const bundlePath = path.resolve(item.bundle || "");
    if (!fs.existsSync(bundlePath) || !fs.statSync(bundlePath).isFile()) {
      return NextResponse.json({ error: `bundle not found: ${bundlePath}` }, { status: 400 });
    }
    const parent = path.dirname(bundlePath);
    let name;
    if (item.name) {
      name = item.name;
    } else if (path.basename(parent) === "final_analysis" && path.basename(path.dirname(parent)).startsWith("results")) {
      name = path.basename(path.dirname(path.dirname(parent)));
    } else {
      name = path.basename(parent) || path.basename(bundlePath, path.extname(bundlePath));
    }
    let runDir;
    if (item.run_dir) runDir = path.resolve(item.run_dir);
    else if (path.basename(parent) === "phasor_hmm_results") runDir = path.dirname(parent);
    else runDir = parent;
    if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) {
      return NextResponse.json({ error: `run directory not found: ${runDir}` }, { status: 400 });
    }
    datasets.push({ name, bundle: bundlePath, run_dir: runDir });
  }

  fs.mkdirSync(outputDir, { recursive: true });
  const dsJson = path.join(outputDir, "_selected_datasets.json");
  fs.writeFileSync(dsJson, JSON.stringify(datasets, null, 2));
  const python = pipelinePython();
  const env = { PHASOR_DATASETS_JSON: dsJson, PHASOR_OUTPUT_ROOT: outputDir };

  const scriptDir = path.join(PIPELINE, "06_plots");
  const script = path.join(scriptDir, `${plotId}.py`);
  if (!fs.existsSync(script)) return NextResponse.json({ error: `plot script not found: ${script}` }, { status: 400 });

  const title = PLOTS.find((row) => row[0] === plotId)?.[1] || plotId;
  const argv = [python, script];
  if (plotId === "viterbi_polar_rep_cycle") {
    argv.push("--rep", String(body.rep || 1), "--cycle", String(body.cycle || 3));
  }
  try {
    const job = launch(title, argv, {
      extraEnv: env,
      output: outputDir,
      href: "/plots",
      stage: plotId,
    });
    return NextResponse.json({ job: job.id, output: outputDir });
  } catch (err) {
    if (err.busy) return NextResponse.json({ error: "Already one job or operation in process.", busy: true }, { status: 409 });
    throw err;
  }
}
