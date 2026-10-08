import Link from "next/link";

export default function Breadcrumb({ trail }) {
  return (
    <div className="mb-5 flex items-center gap-2 font-mono text-[12px] text-neutral-500">
      {trail.map((step, i) => (
        <span key={i} className="flex items-center gap-2">
          {i > 0 && <span className="text-neutral-700">/</span>}
          {step.href ? (
            <Link href={step.href} className="hover:text-teal-400">
              {step.label}
            </Link>
          ) : (
            <span className="text-neutral-300">{step.label}</span>
          )}
        </span>
      ))}
    </div>
  );
}
