import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { APP_ROOT, OUTPUT_ROOT } from "@/lib/paths";

function resolvePath(value, base) {
  if (!value) return "";
  return path.isAbsolute(value) ? value : path.join(base, value);
}

export async function GET() {
  const catalogPath = path.join(APP_ROOT, "example_data", "catalog.json");
  if (!fs.existsSync(catalogPath)) {
    return NextResponse.json({ datasets: [] });
  }
  const catalog = JSON.parse(fs.readFileSync(catalogPath, "utf8"));
  const datasets = catalog.map((row) => {
    const files = {};
    for (const [key, rel] of Object.entries(row.files || {})) {
      const abs = resolvePath(rel, APP_ROOT);
      files[key] = fs.existsSync(abs) ? abs : "";
    }
    const dirs = {};
    for (const [key, rel] of Object.entries(row.dirs || {})) {
      const abs = resolvePath(rel, APP_ROOT);
      dirs[key] = fs.existsSync(abs) ? abs : "";
    }
    const workspace = {};
    for (const [key, rel] of Object.entries(row.workspace || {})) {
      workspace[key] = resolvePath(rel, OUTPUT_ROOT);
    }
    return {
      id: row.id,
      animal: row.animal,
      label: row.label,
      files,
      dirs,
      analysis_base: (() => {
        const analysisBase = resolvePath(row.analysis_base, APP_ROOT);
        return analysisBase && fs.existsSync(analysisBase) ? analysisBase : "";
      })(),
      workspace,
    };
  });
  return NextResponse.json({ datasets });
}

export async function POST(request) {
  const body = await request.json();
  const catalogPath = path.join(APP_ROOT, "example_data", "catalog.json");
  if (!fs.existsSync(catalogPath)) return NextResponse.json({ error: "No recordings are installed." }, { status: 404 });
  const catalog = JSON.parse(fs.readFileSync(catalogPath, "utf8"));
  const row = catalog.find((item) => item.id === body.id);
  if (!row) return NextResponse.json({ error: "Unknown recording." }, { status: 404 });
  const workspace = {};
  for (const [key, rel] of Object.entries(row.workspace || {})) {
    const abs = resolvePath(rel, OUTPUT_ROOT);
    fs.mkdirSync(abs, { recursive: true });
    workspace[key] = abs;
  }
  return NextResponse.json({ id: row.id, workspace });
}
