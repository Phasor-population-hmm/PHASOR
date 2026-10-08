import { NextResponse } from "next/server";
import { scan } from "@/lib/hmm";

export async function GET() {
  return NextResponse.json(scan());
}
