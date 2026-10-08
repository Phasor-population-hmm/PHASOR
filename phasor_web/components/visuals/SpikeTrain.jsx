function randomTicks(seed, count) {
  let s = seed;
  const rand = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  return Array.from({ length: count }, () => rand());
}

const ROWS = [randomTicks(11, 90), randomTicks(37, 90), randomTicks(58, 90)];
const COLORS = ["#2dd4bf", "#f59e0b", "#a78bfa"];

export default function SpikeTrain({ className }) {
  return (
    <div className={`overflow-hidden ${className || ""}`}>
      <div className="flex w-[200%] animate-[spikeScroll_18s_linear_infinite]">
        {[0, 1].map((rep) => (
          <div key={rep} className="flex w-1/2 flex-col gap-2.5">
            {ROWS.map((row, r) => (
              <div key={r} className="flex h-4 items-end gap-[3px]">
                {row.map((v, i) => (
                  <span
                    key={i}
                    style={{ height: `${8 + v * 100}%`, background: COLORS[r], opacity: 0.25 + v * 0.55 }}
                    className="w-[2px] shrink-0 rounded-sm"
                  />
                ))}
              </div>
            ))}
          </div>
        ))}
      </div>
      <style>{`@keyframes spikeScroll { to { transform: translateX(-50%); } }`}</style>
    </div>
  );
}
