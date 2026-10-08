const STEPS = [
  { key: "search", label: "Search", detail: "η = 0.0005, 0.002, 0.005 · α = 0.5" },
  { key: "refit", label: "Refit", detail: "5 seeds · elbow K* · K in {3, 6, 9, 12, 15}" },
  { key: "analysis", label: "Analysis", detail: "Viterbi decode, bundle.npz" },
  { key: "done", label: "Done", detail: "ready for reporting" },
];

export default function HmmStepper({ current, failed }) {
  const currentIndex = STEPS.findIndex((s) => s.key === current);

  return (
    <div className="flex items-stretch gap-0">
      {STEPS.map((step, i) => {
        const reached = currentIndex >= 0 && i <= currentIndex;
        const active = i === currentIndex && !failed;
        const isFailed = failed && i === currentIndex;
        return (
          <div key={step.key} className="flex flex-1 items-center">
            <div className="flex flex-col items-center gap-2 text-center">
              <div
                className={`flex h-9 w-9 items-center justify-center rounded-full border-2 font-mono text-xs transition ${
                  isFailed
                    ? "border-rose-500 bg-rose-500/10 text-rose-400"
                    : active
                    ? "border-amber-400 bg-amber-400/10 text-amber-300 animate-pulse"
                    : reached
                    ? "border-emerald-400 bg-emerald-400/10 text-emerald-300"
                    : "border-neutral-800 text-neutral-600"
                }`}
              >
                {i + 1}
              </div>
              <div>
                <div className={`text-[13px] font-medium ${reached ? "text-neutral-200" : "text-neutral-600"}`}>{step.label}</div>
                <div className="text-[11px] text-neutral-600">{step.detail}</div>
              </div>
            </div>
            {i < STEPS.length - 1 && <div className={`mx-2 h-px flex-1 ${reached && i < currentIndex ? "bg-emerald-400/50" : "bg-neutral-800"}`} />}
          </div>
        );
      })}
    </div>
  );
}
