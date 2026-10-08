import fs from "fs";
import path from "path";
import Link from "next/link";
import { STAGES, PLOTS } from "@/lib/stages";
import { APP_ROOT, OUTPUT_ROOT, displayPath } from "@/lib/paths";
import { projectDir, pythonExecutable } from "@/lib/hmm";
import { IconArrowRight } from "@/components/icons";
import PhaseClock from "@/components/visuals/PhaseClock";
import HomeBoard from "@/components/HomeBoard";
import SpikeTrain from "@/components/visuals/SpikeTrain";
import StatCounter from "@/components/StatCounter";

function exists(p) {
  try {
    return fs.existsSync(p);
  } catch {
    return false;
  }
}

export default function OverviewPage() {
  const venvPython = path.join(APP_ROOT, ".venv", "bin", "python3");
  const status = [
    { label: "Pipeline scripts", ok: exists(path.join(APP_ROOT, "phasor_pipeline")), detail: "phasor_pipeline/" },
    { label: "HMM engine", ok: exists(projectDir()), detail: displayPath(projectDir()) },
    { label: "Python interpreter", ok: true, detail: exists(venvPython) ? displayPath(venvPython) : `${pythonExecutable()} (system)` },
    { label: "Output directory", ok: true, detail: displayPath(OUTPUT_ROOT) },
  ];

  return (
    <div className="animate-rise">
      <div className="grid grid-cols-1 gap-12 lg:grid-cols-[1fr_340px] lg:items-center">
        <div>
          <div className="mb-2 font-mono text-xs uppercase tracking-[0.24em] text-teal-400">Console</div>
          <h1 className="text-gradient text-5xl font-semibold tracking-tight sm:text-6xl">PHASOR</h1>
          <p className="mt-4 max-w-xl text-[15px] leading-relaxed text-neutral-400">
            Population HMM Analysis of Stimulus-locked Output in the Retina. Stimulus-locked hidden Markov models of
            retinal ganglion cell population activity, from raw recording to decoded latent states.
          </p>

          <div className="mt-8 flex gap-10">
            <StatCounter value={STAGES.length} label="pipeline stages" />
            <StatCounter value={PLOTS.length} label="analyses" />
            <StatCounter value={15} label="candidate states" />
          </div>

          <Link
            href="/pipeline"
            className="group mt-9 flex max-w-xl items-center justify-between rounded-2xl border border-white/[0.07] bg-gradient-to-br from-white/[0.04] to-transparent p-6 transition hover:border-teal-500/30"
          >
            <div>
              <div className="text-[16px] font-medium text-neutral-100">Open the pipeline</div>
              <p className="mt-1.5 text-sm text-neutral-500">
                Pre-processing → Processing → Fourier classification → HMM, laid out as one flow.
              </p>
            </div>
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-white/10 text-neutral-400 transition group-hover:border-teal-500/40 group-hover:text-teal-300">
              <IconArrowRight />
            </span>
          </Link>
        </div>

        <div className="mx-auto w-full max-w-[300px]">
          <PhaseClock />
        </div>
      </div>

      <HomeBoard />

      <div className="mt-10 overflow-hidden rounded-xl border border-white/[0.06] bg-white/[0.015] px-5 py-4">
        <div className="mb-3 font-mono text-[11px] uppercase tracking-wide text-neutral-600">binary population activity</div>
        <SpikeTrain className="opacity-70" />
      </div>

      <div className="mt-12 grid grid-cols-1 gap-3 md:grid-cols-3">
        {[
          {
            href: "/pipeline",
            kicker: "Fig. 1",
            title: "Stimulus lock and classification",
            body: "Synctones, 4-D population array, contrast filter, UMAP + OPTICS, Fourier ON / OFF / biphasic and sustained / transient labels.",
          },
          {
            href: "/hmm",
            kicker: "Model",
            title: "Hidden Markov Model",
            body: "Covariance threshold η in {0.0005, 0.002, 0.005}, pseudocount α = 0.5, forward-chaining CV, elbow model order, and decode.",
          },
          {
            href: "/plots",
            kicker: "Plots",
            title: "Post-HMM plots & analysis",
            body: "Firing rate of cells, ON / OFF subtype traces, sustained and transient polars, and a polar for the repetition and cycle you choose.",
          },
        ].map((card) => (
          <Link
            key={card.href}
            href={card.href}
            className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-5 transition hover:border-teal-500/30"
          >
            <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-teal-400">{card.kicker}</div>
            <div className="mt-2 text-[16px] font-medium text-neutral-100">{card.title}</div>
            <p className="mt-2 text-sm leading-relaxed text-neutral-500">{card.body}</p>
          </Link>
        ))}
      </div>

      <div className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        {status.map((s) => (
          <div key={s.label} className="rounded-xl border border-white/[0.06] bg-white/[0.015] p-4">
            <div className="flex items-center gap-2">
              <span className={`h-1.5 w-1.5 rounded-full ${s.ok ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]" : "bg-rose-400"}`} />
              <span className="font-mono text-[11px] uppercase tracking-wide text-neutral-500">{s.label}</span>
            </div>
            <div className="mt-2 truncate font-mono text-[12.5px] text-neutral-300" title={s.detail}>
              {s.detail}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
