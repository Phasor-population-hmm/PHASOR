"use client";

import { useEffect, useState } from "react";

const STATES = [
  { angle: 18, radius: 108, color: "#2dd4bf", size: 6 },
  { angle: 61, radius: 74, color: "#f59e0b", size: 8 },
  { angle: 97, radius: 96, color: "#2dd4bf", size: 5 },
  { angle: 140, radius: 60, color: "#a78bfa", size: 7 },
  { angle: 172, radius: 100, color: "#f87171", size: 5 },
  { angle: 214, radius: 82, color: "#f59e0b", size: 6 },
  { angle: 253, radius: 106, color: "#34d399", size: 5 },
  { angle: 289, radius: 68, color: "#2dd4bf", size: 8 },
  { angle: 327, radius: 92, color: "#a78bfa", size: 5 },
];

function polar(angleDeg, radius, cx = 150, cy = 150) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  const round = (value) => Math.round(value * 1000) / 1000;
  return [round(cx + radius * Math.cos(rad)), round(cy + radius * Math.sin(rad))];
}

export default function PhaseClock() {
  const [r, setR] = useState(0.74);

  useEffect(() => {
    const id = setInterval(() => setR((prev) => Math.min(0.97, Math.max(0.55, prev + (Math.random() - 0.5) * 0.05))), 2200);
    return () => clearInterval(id);
  }, []);

  const ringRadii = [40, 74, 108];

  return (
    <div className="flex flex-col items-center gap-3">
      <svg viewBox="0 0 300 300" className="h-full w-full">
        <defs>
          <radialGradient id="clockGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#2dd4bf" stopOpacity="0.16" />
            <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="sweepGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#2dd4bf" stopOpacity="0" />
            <stop offset="100%" stopColor="#2dd4bf" stopOpacity="0.35" />
          </linearGradient>
        </defs>

        <circle cx="150" cy="150" r="140" fill="url(#clockGlow)" />

        {ringRadii.map((rad) => (
          <circle key={rad} cx="150" cy="150" r={rad} fill="none" stroke="#ffffff" strokeOpacity="0.07" strokeWidth="1" />
        ))}

        {Array.from({ length: 12 }).map((_, i) => {
          const [x1, y1] = polar(i * 30, 108);
          const [x2, y2] = polar(i * 30, 118);
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#ffffff" strokeOpacity="0.15" strokeWidth="1.5" />;
        })}

        {[
          [0, "peak"],
          [90, "falling"],
          [180, "trough"],
          [270, "rising"],
        ].map(([angle, label]) => {
          const [x, y] = polar(Number(angle), 126);
          return (
            <text key={label} x={x} y={y} textAnchor="middle" dominantBaseline="middle" fontSize="8" fill="#525252" fontFamily="var(--font-mono)" letterSpacing="1">
              {label}
            </text>
          );
        })}

        <g style={{ transformOrigin: "150px 150px", animation: "phaseSweep 9s linear infinite" }}>
          <path d={`M150,150 L150,42 A108,108 0 0,1 ${polar(35, 108).join(",")} Z`} fill="url(#sweepGrad)" />
          <line x1="150" y1="150" x2="150" y2="42" stroke="#5eead4" strokeWidth="1.5" strokeLinecap="round" />
        </g>

        {STATES.map((s, i) => {
          const [x, y] = polar(s.angle, s.radius);
          return (
            <g key={i}>
              <circle cx={x} cy={y} r={s.size + 5} fill={s.color} opacity="0.14" style={{ animation: `clockPulse 2.6s ease-in-out ${i * 0.22}s infinite` }} />
              <circle cx={x} cy={y} r={s.size} fill={s.color} opacity="0.9" />
            </g>
          );
        })}

        <circle cx="150" cy="150" r="3" fill="#e5e5e5" />
      </svg>

      <div className="flex items-center justify-center gap-2 font-mono text-[11px] text-neutral-500">
        <span className="h-1.5 w-1.5 rounded-full bg-teal-400 animate-pulse" />
        phase lock R = {r.toFixed(2)}
      </div>

      <style>{`
        @keyframes phaseSweep { to { transform: rotate(360deg); } }
        @keyframes clockPulse { 0%,100% { transform: scale(1); opacity: .14; } 50% { transform: scale(1.6); opacity: .35; } }
      `}</style>
    </div>
  );
}
