"use client";

import { useCountUp } from "@/hooks/useCountUp";

export default function StatCounter({ value, label }) {
  const n = useCountUp(value);
  return (
    <div>
      <div className="font-mono text-2xl font-semibold text-neutral-100">{n}</div>
      <div className="mt-0.5 text-xs text-neutral-500">{label}</div>
    </div>
  );
}
