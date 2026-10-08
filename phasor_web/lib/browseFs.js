import fs from "fs";
import os from "os";
import path from "path";

const MEDIA_ROOTS = ["/media", "/mnt", "/run/media"];

function safeReaddir(dir) {
  try {
    return fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return [];
  }
}

export function shortcuts() {
  const out = [
    { label: "Home", path: os.homedir() },
    { label: "Filesystem", path: "/" },
  ];
  for (const base of MEDIA_ROOTS) {
    if (!fs.existsSync(base)) continue;
    out.push({ label: path.basename(base), path: base });
    for (const child of safeReaddir(base)) {
      if (!child.isDirectory() || child.name.startsWith(".")) continue;
      const childPath = path.join(base, child.name);
      for (const drive of safeReaddir(childPath)) {
        if (drive.isDirectory()) {
          out.push({ label: drive.name, path: path.join(childPath, drive.name) });
        }
      }
      break;
    }
  }
  const seen = new Set();
  const uniq = [];
  for (const s of out) {
    if (!seen.has(s.path)) {
      seen.add(s.path);
      uniq.push(s);
    }
  }
  return uniq.slice(0, 10);
}

function formatSize(bytes) {
  let n = bytes;
  for (const unit of ["B", "K", "M", "G"]) {
    if (n < 1024 || unit === "G") {
      return unit === "B" ? `${n.toFixed(0)}${unit}` : `${n.toFixed(1)}${unit}`;
    }
    n /= 1024;
  }
  return `${bytes}B`;
}

export function listing(rawPath, want = "any", showHidden = false) {
  const raw = (rawPath || "").trim();
  let p = raw ? raw.replace(/^~/, os.homedir()) : os.homedir();
  let error = null;

  if (!fs.existsSync(p)) {
    error = `Not found: ${p}`;
    p = os.homedir();
  } else if (fs.statSync(p).isFile()) {
    p = path.dirname(p);
  }

  const entries = [];
  try {
    const children = safeReaddir(p)
      .filter((c) => showHidden || !c.name.startsWith("."))
      .sort((a, b) => {
        const aIsFile = !a.isDirectory();
        const bIsFile = !b.isDirectory();
        if (aIsFile !== bIsFile) return aIsFile ? 1 : -1;
        return a.name.toLowerCase().localeCompare(b.name.toLowerCase());
      });

    for (const child of children) {
      const full = path.join(p, child.name);
      let isDir;
      try {
        isDir = fs.statSync(full).isDirectory();
      } catch {
        continue;
      }
      if (want === "dir" && !isDir) continue;
      let size = "";
      if (!isDir) {
        try {
          size = formatSize(fs.statSync(full).size);
        } catch {}
      }
      entries.push({ name: child.name, path: full, is_dir: isDir, size });
    }
  } catch (err) {
    error = err.code === "EACCES" ? `Permission denied: ${p}` : String(err.message || err);
  }

  const parent = path.dirname(p);
  return {
    cwd: p,
    parent: parent !== p ? parent : null,
    entries,
    shortcuts: shortcuts(),
    error,
  };
}
