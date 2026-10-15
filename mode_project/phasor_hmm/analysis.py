"""Decode the fitted model for the recording staged in PHASOR_HMM_DATA."""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


def main():
    project_root = Path(__file__).resolve().parents[1]
    script_dir = str(Path(__file__).resolve().parent)
    # This file is named analysis.py. If its folder stays at the front of
    # sys.path, "import analysis" loads this file instead of mode_project/analysis.
    sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry or ".") != script_dir]
    sys.path.insert(0, str(project_root))
    data_root = Path(os.environ["PHASOR_HMM_DATA"])
    run_dir = data_root / "run1"
    results = data_root / "results_run1"
    models = sorted(results.glob("chowliu_K*.npz"), key=lambda p: p.stat().st_mtime)
    if not models:
        raise SystemExit(f"No fitted model in {results}")
    model_path = models[-1]
    k_text = model_path.name.split("_K", 1)[-1].split("_", 1)[0]
    n_modes = int(k_text)

    print("HMM analysis", flush=True)
    print(f"Recording folder: {os.environ.get('PHASOR_HMM_SOURCE', data_root)}", flush=True)
    print(f"Model: {model_path.name}", flush=True)

    src = project_root / "test" / "wt13" / "wt13_run1_analysis.py"
    code = src.read_text(encoding="utf-8")
    replacements = {
        'PROJECT_ROOT = Path("/Users/sichenchen/Desktop/mode_project")':
        f'PROJECT_ROOT = Path("{project_root}")',
        'DATA_DIR = PROJECT_ROOT / "data" / "wt13" / "run1"':
        f'DATA_DIR = Path("{run_dir}")',
        'STIM_JSON_PATH = DATA_DIR / "FFsine_2Hz_stims.json"':
        f'STIM_JSON_PATH = Path("{data_root / "FFsine_2Hz_stims.json"}")',
        'SYNCTONES_CSV_PATH = DATA_DIR / "2026-03-11T12-17-28wt13_Green=255_OD=4_2Hz_run1_B-00068_synctones.csv"':
        f'SYNCTONES_CSV_PATH = Path("{run_dir / "synctones.csv"}")',
        'SPIKE_TXT_PATH = DATA_DIR / "2026-03-11T12-17-28wt13_Green=255_OD=4_2Hz_run1_B-00068.txt"':
        f'SPIKE_TXT_PATH = Path("{run_dir / "spikes.txt"}")',
        'MODEL_PATH = DATA_DIR / "chowliu_K12_eta0.005_alpha0.5.npz"':
        f'MODEL_PATH = Path("{model_path}")',
        'OUTPUT_DIR = DATA_DIR / "final_analysis"':
        f'OUTPUT_DIR = Path("{results / "final_analysis"}")',
        "N_MODES = 12":
        f"N_MODES = {n_modes}",
    }
    for old, new in replacements.items():
        if old not in code:
            raise RuntimeError("The analysis script no longer matches the expected settings block.")
        code = code.replace(old, new, 1)

    ns = {"__name__": "__main__", "__file__": str(src), "__package__": None}
    exec(compile(code, str(src), "exec"), ns, ns)


if __name__ == "__main__":
    main()
