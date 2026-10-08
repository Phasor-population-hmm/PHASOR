import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

const TYPES = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".html": "text/html; charset=utf-8",
  ".pdf": "application/pdf",
};

export async function GET(request) {
  const raw = new URL(request.url).searchParams.get("path") || "";
  const file = path.resolve(raw);
  const ext = path.extname(file).toLowerCase();
  if (!TYPES[ext] || !file.startsWith("/") || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    return NextResponse.json({ error: "file not available" }, { status: 404 });
  }
  const data = fs.readFileSync(file);
  return new NextResponse(data, { headers: { "Content-Type": TYPES[ext], "Cache-Control": "no-store" } });
}
