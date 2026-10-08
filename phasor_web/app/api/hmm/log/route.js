import { NextResponse } from "next/server";
import { tail } from "@/lib/hmm";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const dataDir = searchParams.get("data") || "";
  if (!dataDir) return NextResponse.json({ error: "no folder" }, { status: 400 });
  return NextResponse.json(tail(dataDir));
}
