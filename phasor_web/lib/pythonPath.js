import fs from "fs";
import path from "path";
import { APP_ROOT } from "./paths";

export function pipelinePython() {
  if (process.env.PHASOR_PYTHON) return process.env.PHASOR_PYTHON;
  const venv = path.join(APP_ROOT, ".venv", "bin", "python3");
  if (fs.existsSync(venv)) return venv;
  return "python3";
}
