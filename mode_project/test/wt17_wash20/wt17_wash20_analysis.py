#!/usr/bin/env python3
"""
Run the original wt13_run1_analysis.py logic for wt17_wash20.

Only paths/settings are changed; analysis code path remains identical.
"""

from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "test" / "wt13" / "wt13_run1_analysis.py"
    code = src.read_text(encoding="utf-8")

    # Keep original logic; only update user settings block.
    replacements = {
        'PROJECT_ROOT = Path("/Users/sichenchen/Desktop/mode_project")':
        f'PROJECT_ROOT = Path("{project_root}")',
        'DATA_DIR = PROJECT_ROOT / "data" / "wt13" / "run1"':
        'DATA_DIR = PROJECT_ROOT / "data" / "wt17_wash20" / "run1"',
        'STIM_JSON_PATH = DATA_DIR / "FFsine_2Hz_stims.json"':
        'STIM_JSON_PATH = PROJECT_ROOT / "data" / "wt17_wash20" / "FFsine_2Hz_stims.json"',
        'SYNCTONES_CSV_PATH = DATA_DIR / "2026-03-11T12-17-28wt13_Green=255_OD=4_2Hz_run1_B-00068_synctones.csv"':
        'SYNCTONES_CSV_PATH = DATA_DIR / "synctones.csv"',
        'SPIKE_TXT_PATH = DATA_DIR / "2026-03-11T12-17-28wt13_Green=255_OD=4_2Hz_run1_B-00068.txt"':
        'SPIKE_TXT_PATH = DATA_DIR / "spikes.txt"',
        'MODEL_PATH = DATA_DIR / "chowliu_K12_eta0.005_alpha0.5.npz"':
        'MODEL_PATH = PROJECT_ROOT / "data" / "wt17_wash20" / "results_run1" / "chowliu_K12_eta0.1_alpha0.5.npz"',
        'OUTPUT_DIR = DATA_DIR / "final_analysis"':
        'OUTPUT_DIR = PROJECT_ROOT / "data" / "wt17_wash20" / "results_run1" / "final_analysis"',
    }

    for old, new in replacements.items():
        if old not in code:
            raise RuntimeError(f"Expected string not found in source: {old}")
        code = code.replace(old, new)

    ns = {"__name__": "__main__", "__file__": str(src)}
    exec(compile(code, str(src), "exec"), ns, ns)


if __name__ == "__main__":
    main()
