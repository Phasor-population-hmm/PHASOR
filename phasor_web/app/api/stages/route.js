import { NextResponse } from "next/server";
import { STAGES, PLOTS } from "@/lib/stages";

export async function GET() {
  const stages = STAGES.map((s) => ({
    id: s.id,
    title: s.title,
    phase: s.phase,
    note: s.note || "",
    fields: s.fields.map((f) => ({ key: f.key, label: f.label, kind: f.kind, required: f.required, filter: f.filter || "" })),
  }));
  const plots = PLOTS.map(([id, title]) => ({ id, title }));
  return NextResponse.json({ stages, plots });
}
