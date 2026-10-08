import Link from "next/link";

export default function NotFound() {
  return (
    <div className="animate-rise">
      <div className="mb-1 font-mono text-xs uppercase tracking-[0.2em] text-teal-400">404</div>
      <h1 className="text-3xl font-semibold tracking-tight text-neutral-50">Not found</h1>
      <p className="mt-3 text-neutral-400">That page doesn&apos;t exist.</p>
      <Link href="/" className="mt-6 inline-flex rounded-md bg-teal-500 px-5 py-2.5 text-sm font-medium text-neutral-950 hover:bg-teal-400">
        Back to overview
      </Link>
    </div>
  );
}
