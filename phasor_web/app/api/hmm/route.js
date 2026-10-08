import { NextResponse } from "next/server";
import { start, stop, saveChain } from "@/lib/hmm";

export async function POST(request) {
  const body = await request.json();
  if (body.action === "stop") {
    return NextResponse.json(stop(body.data || ""));
  }
  const dataDir = body.data || "";
  if (!dataDir) return NextResponse.json({ error: "select a dataset folder" }, { status: 400 });
  const chain = body.chain || null;
  if (chain) saveChain(chain);
  const result = start(dataDir, chain, body.inputs);
  if (result.busy) return NextResponse.json(result, { status: 409 });
  return NextResponse.json(result);
}
