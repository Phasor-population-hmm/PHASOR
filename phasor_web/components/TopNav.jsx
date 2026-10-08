"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import QuickJump from "./QuickJump";
import { useRecordingCatalog } from "./RecordingPicker";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/pipeline", label: "Pipeline" },
  { href: "/hmm", label: "Hidden Markov Model" },
  { href: "/plots", label: "Post-HMM plots" },
];

function isActive(pathname, href) {
  if (href === "/") return pathname === "/";
  if (href === "/pipeline") return pathname === "/pipeline" || pathname.startsWith("/stage/");
  return pathname.startsWith(href);
}

export default function TopNav() {
  const pathname = usePathname();
  const { choice, dataset, ready } = useRecordingCatalog();
  const chip = dataset
    ? `${dataset.animal} · ${dataset.label}`
    : choice === "other"
      ? "Other files"
      : ready
        ? "No recording"
        : "Loading recording";

  return (
    <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-[#08090a]/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-[1400px] items-center gap-8 px-8">
        <Link href="/" className="flex items-center gap-2.5">
          <Dial />
          <span className="font-mono text-[15px] font-semibold tracking-[0.22em] text-neutral-100">PHASOR</span>
        </Link>

        <nav className="flex items-center gap-0.5">
          {LINKS.map((link) => {
            const active = isActive(pathname, link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`relative rounded-md px-3.5 py-2 text-[13.5px] transition ${
                  active ? "text-neutral-50" : "text-neutral-500 hover:text-neutral-200"
                }`}
              >
                {link.label}
                {active && <span className="absolute inset-x-3.5 -bottom-[1px] h-[2px] rounded-full bg-gradient-to-r from-teal-400 to-teal-300" />}
              </Link>
            );
          })}
        </nav>

        <div className="flex-1" />
        <div
          title={dataset?.workspace?.root || "Choose a recording on the overview"}
          className="hidden max-w-[220px] truncate rounded-full border border-teal-500/30 bg-teal-500/10 px-3 py-1.5 text-[12.5px] text-teal-100 sm:block"
        >
          {chip}
        </div>
        <QuickJump />
      </div>
    </header>
  );
}

function Dial() {
  return (
    <svg viewBox="0 0 40 40" className="h-8 w-8 shrink-0">
      <defs>
        <linearGradient id="dialRing" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#2dd4bf" stopOpacity=".9" />
          <stop offset="1" stopColor="#2dd4bf" stopOpacity="0" />
        </linearGradient>
      </defs>
      <circle cx="20" cy="20" r="18" fill="none" stroke="url(#dialRing)" strokeWidth="1.5" />
      <circle cx="20" cy="20" r="1.6" fill="#2dd4bf" />
      <g style={{ transformOrigin: "20px 20px", animation: "spin 3.2s linear infinite" }}>
        <line x1="20" y1="20" x2="34" y2="20" stroke="#2dd4bf" strokeWidth="1.8" />
        <circle cx="34" cy="20" r="2.4" fill="#f59e0b" />
      </g>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </svg>
  );
}
