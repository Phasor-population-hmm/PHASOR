import { NextResponse } from "next/server";
import { ackSession, publicSession } from "@/lib/session";

export async function GET() {
  return NextResponse.json(publicSession());
}

export async function POST() {
  ackSession();
  return NextResponse.json(publicSession());
}
