import PipelineFlow from "@/components/PipelineFlow";
import RecordingStrip from "@/components/RecordingStrip";

export default function PipelinePage() {
  return (
    <div className="animate-rise">
      <div className="mb-1 font-mono text-xs uppercase tracking-[0.2em] text-teal-400">Pipeline</div>
      <h1 className="text-3xl font-semibold tracking-tight text-neutral-50">Stimulus-locked HMM analysis</h1>
      <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-neutral-400">
        Pre-processing, Processing, Fourier classification, Hidden Markov Model, then Post-HMM plots and analysis.
      </p>

      <RecordingStrip />

      <div className="mt-10">
        <PipelineFlow />
      </div>
    </div>
  );
}
