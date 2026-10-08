#!/usr/bin/env python3
"""WT2 Run2 final analysis (WT2 Run2 100 ms) — auto-picks refit chowliu_K*.npz."""

from pathlib import Path


def _pick_model(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("chowliu_K*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No chowliu_K*.npz in {results_dir}")
    if len(candidates) > 1:
        print(f"Multiple models found; using {candidates[-1].name}")
    return candidates[-1]


def main():
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "test" / "wt13" / "wt13_run1_analysis.py"
    code = src.read_text(encoding="utf-8")

    results_dir = project_root / "data" / "wt2_run2_100ms" / "results_run1"
    model_path = _pick_model(results_dir)

    replacements = {
        'PROJECT_ROOT = Path("/Users/sichenchen/Desktop/mode_project")':
        f'PROJECT_ROOT = Path("{project_root}")',
        'DATA_DIR = PROJECT_ROOT / "data" / "wt13" / "run1"':
        'DATA_DIR = PROJECT_ROOT / "data" / "wt2_run2_100ms" / "run1"',
        'STIM_JSON_PATH = DATA_DIR / "FFsine_2Hz_stims.json"':
        'STIM_JSON_PATH = PROJECT_ROOT / "data" / "wt2_run2_100ms" / "FFsine_2Hz_stims.json"',
        'SYNCTONES_CSV_PATH = DATA_DIR / "2026-03-11T12-17-28wt13_Green=255_OD=4_2Hz_run1_B-00068_synctones.csv"':
        'SYNCTONES_CSV_PATH = DATA_DIR / "synctones.csv"',
        'SPIKE_TXT_PATH = DATA_DIR / "2026-03-11T12-17-28wt13_Green=255_OD=4_2Hz_run1_B-00068.txt"':
        'SPIKE_TXT_PATH = DATA_DIR / "spikes.txt"',
        'MODEL_PATH = DATA_DIR / "chowliu_K12_eta0.005_alpha0.5.npz"':
        f'MODEL_PATH = Path("{model_path}")',
        'OUTPUT_DIR = DATA_DIR / "final_analysis"':
        'OUTPUT_DIR = PROJECT_ROOT / "data" / "wt2_run2_100ms" / "results_run1" / "final_analysis"',
        'BIN_SIZE_S = 0.01':
        'BIN_SIZE_S = 0.100',
        "    z_reps_train = z_vit_train.reshape(10, 3000)\n    z_reps_test = z_vit_test.reshape(5, 3000)":
        "    n_train_rep = 10\n    n_test_rep = 5\n    bins_per_rep_train = len(z_vit_train) // n_train_rep\n    bins_per_rep_test = len(z_vit_test) // n_test_rep\n    z_reps_train = z_vit_train.reshape(n_train_rep, bins_per_rep_train)\n    z_reps_test = z_vit_test.reshape(n_test_rep, bins_per_rep_test)",
    }

    for old, new in replacements.items():
        if old not in code:
            raise RuntimeError(f"Expected string not found in source: {old}")
        code = code.replace(old, new)

    ns = {"__name__": "__main__", "__file__": str(src), "Path": Path}
    exec(compile(code, str(src), "exec"), ns, ns)


if __name__ == "__main__":
    main()
