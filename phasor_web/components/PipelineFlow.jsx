"use client";

import Link from "next/link";
import { STAGES } from "@/lib/stages";
import { IconWave, IconFilter, IconSine, IconNetwork, IconChart } from "./icons";
import StateGraph from "./visuals/StateGraph";
import Tilt from "./Tilt";

const PHASE_META = {
  "Pre-processing": {
    icon: IconWave,
    blurb: "Recover stimulus timing and assemble the raw recording into a synchronized 4-D spike array.",
  },
  Processing: {
    icon: IconFilter,
    blurb: "Split by contrast condition and run unsupervised UMAP + OPTICS clustering.",
  },
  "Fourier classification": {
    icon: IconSine,
    blurb: "Harmonic decomposition assigns each unit a polarity and kinetic phenotype.",
  },
};

const PHASES = ["Pre-processing", "Processing", "Fourier classification"];

export default function PipelineFlow() {
  return (
    <div>
      <div className="flex flex-col gap-6 lg:flex-row lg:items-stretch lg:gap-0">
        {PHASES.map((phase) => (
          <div key={phase} className="flex flex-1 items-stretch">
            <PhaseCard phase={phase} />
            <Connector />
          </div>
        ))}
        <HmmNode />
      </div>

      <div className="mt-10 grid grid-cols-1 gap-3 rounded-2xl border border-white/[0.07] bg-white/[0.015] p-6 lg:grid-cols-[280px_1fr]">
        <StateGraph className="h-[190px] w-full text-neutral-500" />
        <div className="flex flex-col justify-center">
          <div className="font-mono text-xs uppercase tracking-[0.16em] text-neutral-500">latent state transitions</div>
          <p className="mt-2 max-w-md text-sm leading-relaxed text-neutral-400">
            Each fitted model is a K-state Chow-Liu hidden Markov chain. Dots trace the transition matrix A, moving
            state to state the way the decoded population actually switches between modes across a stimulus cycle.
          </p>
        </div>
      </div>

      <div className="mt-14 flex items-center gap-3">
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-white/10 to-white/10" />
        <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-neutral-600">reporting</span>
        <div className="h-px flex-1 bg-gradient-to-l from-transparent via-white/10 to-white/10" />
      </div>

      <Link
        href="/plots"
        className="group mt-6 flex items-center gap-4 rounded-2xl border border-white/[0.07] bg-gradient-to-br from-white/[0.04] to-transparent p-6 transition hover:border-teal-500/30"
      >
        <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl border border-teal-500/20 bg-teal-500/[0.06] text-teal-300">
          <IconChart />
        </span>
        <div>
          <div className="font-mono text-xs uppercase tracking-[0.16em] text-teal-400">Post-HMM</div>
          <div className="mt-0.5 text-[15px] text-neutral-100">Post-HMM plots & analysis</div>
          <p className="mt-1 text-sm text-neutral-500">Firing rate of cells, subtype traces, and polar figures from a finished model.</p>
        </div>
      </Link>
    </div>
  );
}

function PhaseCard({ phase }) {
  const stages = STAGES.filter((s) => s.phase === phase);
  const { icon: Icon, blurb } = PHASE_META[phase];
  return (
    <Tilt className="flex-1 [transform-style:preserve-3d]">
      <div className="h-full rounded-2xl border border-white/[0.07] bg-white/[0.015] p-5 transition hover:border-white/[0.14]">
        <div className="flex items-center gap-2.5">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-white/10 text-neutral-400">
            <Icon />
          </span>
          <div className="font-mono text-xs uppercase tracking-[0.14em] text-neutral-400">{phase}</div>
        </div>
        <p className="mt-3 text-sm leading-relaxed text-neutral-500">{blurb}</p>
        <div className="mt-4 flex flex-col gap-1">
          {stages.map((stage) => (
            <Link
              key={stage.id}
              href={`/stage/${stage.id}`}
              className="group flex items-center gap-2.5 rounded-md border border-transparent px-2.5 py-2 text-[13.5px] text-neutral-300 transition hover:border-white/10 hover:bg-white/[0.03]"
            >
              <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-neutral-700 transition group-hover:bg-teal-400" />
              {stage.title}
            </Link>
          ))}
        </div>
      </div>
    </Tilt>
  );
}

function HmmNode() {
  const stage = STAGES.find((s) => s.id === "hmm");
  return (
    <Tilt className="flex-1 lg:ml-8 [transform-style:preserve-3d]">
      <Link
        href="/hmm"
        className="group relative flex h-full flex-col justify-between overflow-hidden rounded-2xl border border-teal-500/25 bg-gradient-to-b from-teal-500/[0.09] via-teal-500/[0.02] to-transparent p-5 transition hover:border-teal-500/50"
      >
        <div className="pointer-events-none absolute -right-10 -top-10 h-40 w-40 rounded-full bg-teal-400/10 blur-3xl transition group-hover:bg-teal-400/20" />
        <div className="relative">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-teal-500/30 bg-teal-500/10 text-teal-300">
            <IconNetwork />
          </span>
          <div className="mt-2.5 font-mono text-xs uppercase tracking-[0.14em] text-teal-300">Hidden Markov Model</div>
          <div className="mt-1 text-[15px] font-medium text-neutral-100">{stage.title}</div>
          <p className="mt-2 text-sm leading-relaxed text-neutral-500">
            Fits the Chow-Liu hidden Markov model and decodes latent population states with Viterbi.
          </p>
        </div>
        <div className="relative mt-5 font-mono text-[11px] text-teal-400/80 opacity-0 transition group-hover:opacity-100">open workflow →</div>
      </Link>
    </Tilt>
  );
}

function Connector() {
  return (
    <div className="hidden w-10 shrink-0 items-center justify-center lg:flex">
      <svg width="28" height="16" viewBox="0 0 28 16" fill="none">
        <defs>
          <linearGradient id="connGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="#525252" />
            <stop offset="1" stopColor="#2dd4bf" stopOpacity=".7" />
          </linearGradient>
        </defs>
        <path d="M0 8h20M14 2l6 6-6 6" stroke="url(#connGrad)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  );
}
