import { NextResponse } from "next/server";
import { discoverInputs } from "@/lib/hmm";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const dataDir = searchParams.get("data") || "";
  if (!dataDir) return NextResponse.json({ error: "select a dataset folder" }, { status: 400 });
  return NextResponse.json(discoverInputs(dataDir));
}
