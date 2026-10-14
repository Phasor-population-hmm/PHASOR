import { NextResponse } from "next/server";
import { getJob, stopJob } from "@/lib/jobs";

export async function GET(request, { params }) {
  const { id } = await params;
  const job = getJob(id);
  if (!job) return NextResponse.json({ error: "no such job" }, { status: 404 });
  return NextResponse.json(job.snapshot());
}

export async function POST(request, { params }) {
  const { id } = await params;
  if (!stopJob(id)) return NextResponse.json({ error: "This run is not active." }, { status: 404 });
  return NextResponse.json({ stopped: true });
}
