const NODES = [
  { id: "a", x: 40, y: 60, color: "#2dd4bf" },
  { id: "b", x: 150, y: 30, color: "#f59e0b" },
  { id: "c", x: 260, y: 60, color: "#2dd4bf" },
  { id: "d", x: 70, y: 150, color: "#a78bfa" },
  { id: "e", x: 230, y: 150, color: "#f87171" },
  { id: "f", x: 150, y: 180, color: "#34d399" },
];

const EDGES = [
  ["a", "b"],
  ["b", "c"],
  ["a", "d"],
  ["b", "f"],
  ["c", "e"],
  ["d", "f"],
  ["f", "e"],
  ["d", "a"],
];

const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));

export default function StateGraph({ className }) {
  return (
    <svg viewBox="0 0 300 210" className={className}>
      {EDGES.map(([from, to], i) => {
        const a = byId[from];
        const b = byId[to];
        return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#ffffff" strokeOpacity="0.08" strokeWidth="1.2" />;
      })}

      {EDGES.map(([from, to], i) => {
        const a = byId[from];
        const b = byId[to];
        const pathId = `edge-${i}`;
        return (
          <g key={pathId}>
            <path id={pathId} d={`M${a.x},${a.y} L${b.x},${b.y}`} fill="none" stroke="none" />
            <circle r="2" fill={a.color} opacity="0.9">
              <animateMotion dur={`${3 + (i % 4)}s`} repeatCount="indefinite" begin={`${i * 0.35}s`}>
                <mpath href={`#${pathId}`} />
              </animateMotion>
            </circle>
          </g>
        );
      })}

      {NODES.map((n) => (
        <g key={n.id}>
          <circle cx={n.x} cy={n.y} r="10" fill={n.color} opacity="0.12" />
          <circle cx={n.x} cy={n.y} r="4.5" fill={n.color} />
        </g>
      ))}
    </svg>
  );
}
