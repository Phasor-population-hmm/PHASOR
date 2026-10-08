import { NextResponse } from "next/server";
import { getJob } from "@/lib/jobs";

export async function GET(request, { params }) {
  const { id } = await params;
  const job = getJob(id);
  if (!job) return NextResponse.json({ error: "no such job" }, { status: 404 });
  return NextResponse.json(job.snapshot());
}
