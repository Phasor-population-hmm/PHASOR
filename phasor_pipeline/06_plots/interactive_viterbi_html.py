#!/usr/bin/env python3
"""Write the interactive all-repetition Viterbi HTML into PHASOR_OUTPUT_ROOT."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

OUT = Path(os.environ.get("PHASOR_OUTPUT_ROOT", "phasor_output"))
_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = _ROOT / "05_viterbi" / "viterbi_bands_wt22.py"

_gs = importlib.util.spec_from_file_location(
    "gcp", _ROOT / "05_viterbi" / "generate_contrast_progression_png.py"
)
gcp = importlib.util.module_from_spec(_gs)
_gs.loader.exec_module(gcp)


def main() -> None:
    if not gcp.DATASETS:
        raise SystemExit("No dataset was selected.")
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in gcp.DATASETS:
        bundle = Path(ds["bundle"])
        run_dir = gcp.ds_run_dir(ds)
        if not bundle.exists():
            raise SystemExit(f"Missing bundle: {bundle}")
        if not run_dir.exists():
            raise SystemExit(f"Missing stimulus folder: {run_dir}")
        print(f"\n===== {ds['name']} =====", flush=True)
        print(f"Out {OUT}", flush=True)
        command = [
            sys.executable,
            str(SCRIPT),
            "--bundle", str(bundle),
            "--out_dir", str(OUT),
            "--base_dir", str(run_dir),
            "--all_reps_only",
        ]
        code = subprocess.call(command)
        if code != 0:
            raise SystemExit(code)
        print("saved all_reps_interactive.html", flush=True)


if __name__ == "__main__":
    main()
