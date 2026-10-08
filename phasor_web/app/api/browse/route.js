import { NextResponse } from "next/server";
import { listing } from "@/lib/browseFs";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const data = listing(searchParams.get("path") || "", searchParams.get("want") || "any", searchParams.get("hidden") === "1");
  return NextResponse.json(data);
}
