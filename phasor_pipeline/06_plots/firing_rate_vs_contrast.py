#!/usr/bin/env python3
"""Mean firing rate versus contrast. No Hill fit.

Writes, into PHASOR_OUTPUT_ROOT:
  all_cells_firing_rate.png
  on_off_biphasic_firing_rate.png
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BIN_MS = 10.0
CONTRASTS = (0, 50, 60, 70, 80, 90)
SUBTYPES = ("ON", "OFF", "Biphasic")
COLORS = {"All cells": "#222222", "ON": "#E69F00", "OFF": "#0072B2", "Biphasic": "#009E73"}
OUT = Path(os.environ.get("PHASOR_OUTPUT_ROOT", "phasor_output"))

_ROOT = Path(__file__).resolve().parent.parent
_gs = importlib.util.spec_from_file_location("gcp", _ROOT / "05_viterbi" / "generate_contrast_progression_png.py")
gcp = importlib.util.module_from_spec(_gs)
_gs.loader.exec_module(gcp)


def cond_dir(root: Path, contrast: int) -> Path:
    for candidate in (
        root / "Filtered_Data" / f"c{contrast}",
        root / "Filtered Data" / f"c{contrast}",
        root / "clustering" / "filtered" / f"c{contrast}_f2Hz",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No c{contrast} folder under {root}")


def load_array(root: Path, contrast: int) -> np.ndarray:
    folder = cond_dir(root, contrast)
    files = list(folder.glob("*filtered_4d.npz")) or list(folder.glob("*_4d.npz"))
    if not files:
        raise FileNotFoundError(f"No 4D array in {folder}")
    with np.load(files[0]) as data:
        key = "array" if "array" in data.files else data.files[0]
        return data[key].astype(np.float64)


def labels_for(root: Path) -> dict[tuple[int, int], str]:
    for contrast in (90, 80, 70, 60, 50):
        try:
            folder = cond_dir(root, contrast)
        except FileNotFoundError:
            continue
        csv = folder / "Stage5_Cell_Fourier_Analysis" / "cell_fourier_results.csv"
        if not csv.exists():
            continue
        frame = pd.read_csv(csv)
        array = load_array(root, contrast)
        cells = np.argwhere(np.any(array > 0, axis=(0, 1)))
        mapping = {}
        if "cell_idx" in frame.columns:
            for _, row in frame.iterrows():
                label = str(row["label"])
                index = int(row["cell_idx"])
                if label in SUBTYPES and 0 <= index < len(cells):
                    mapping[(int(cells[index, 0]), int(cells[index, 1]))] = label
        else:
            count = min(len(frame), len(cells))
            for index in range(count):
                label = str(frame["label"].iloc[index])
                if label in SUBTYPES:
                    mapping[(int(cells[index, 0]), int(cells[index, 1]))] = label
        if mapping:
            print(f"  subtype identities from C{contrast} ({len(mapping)} cells)", flush=True)
            return mapping
    return {}


def population(root: Path) -> list[tuple[int, int]]:
    """One unit list for every contrast, so a silent cell stays in the mean as 0 Hz."""
    seen: dict[tuple[int, int], None] = {}
    for contrast in CONTRASTS:
        array = load_array(root, contrast)
        for ch, unit in np.argwhere(np.any(array > 0, axis=(0, 1))):
            seen[(int(ch), int(unit))] = None
    return list(seen)


def table_for(root: Path) -> pd.DataFrame:
    identities = labels_for(root)
    units = population(root)
    rows = []
    for contrast in CONTRASTS:
        array = load_array(root, contrast)
        rates = np.array([
            array[:, :, ch, unit].mean() * (1000.0 / BIN_MS)
            if ch < array.shape[2] and unit < array.shape[3] else 0.0
            for ch, unit in units
        ]) if units else np.array([])
        mean = float(rates.mean()) if len(rates) else np.nan
        sem = float(rates.std(ddof=1) / np.sqrt(len(rates))) if len(rates) > 1 else 0.0
        rows.append({"contrast": contrast, "group": "All cells", "mean": mean, "sem": sem, "n": int(len(rates))})
        for subtype in SUBTYPES:
            chosen = [
                float(array[:, :, ch, unit].mean() * (1000.0 / BIN_MS))
                for (ch, unit), label in identities.items()
                if label == subtype and ch < array.shape[2] and unit < array.shape[3]
            ]
            values = np.asarray(chosen, dtype=np.float64)
            rows.append({
                "contrast": contrast,
                "group": subtype,
                "mean": float(values.mean()) if len(values) else np.nan,
                "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            })
    return pd.DataFrame(rows)


def draw(frame: pd.DataFrame, groups: tuple[str, ...], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for group in groups:
        part = frame[frame["group"] == group].sort_values("contrast").dropna(subset=["mean"])
        if part.empty:
            continue
        color = COLORS[group]
        ax.plot(part["contrast"], part["mean"], color=color, lw=2.6, marker="o", ms=7, label=group)
        ax.fill_between(part["contrast"], part["mean"] - part["sem"], part["mean"] + part["sem"], color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel("Stimulus contrast (%)", fontweight="bold")
    ax.set_ylabel("Mean firing rate (Hz)", fontweight="bold")
    ax.set_xticks(list(CONTRASTS))
    ax.set_xlim(-4, 98)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_title(title, fontweight="bold")
    if len(groups) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.name}", flush=True)


def main() -> None:
    if not gcp.DATASETS:
        raise SystemExit("No dataset was selected.")
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in gcp.DATASETS:
        root = gcp.ds_run_dir(ds)
        print(f"\n===== {ds['name']} =====", flush=True)
        print(f"  data {root}", flush=True)
        frame = table_for(root)
        frame.to_csv(OUT / "firing_rate_vs_contrast.csv", index=False)
        draw(frame, ("All cells",), f"{ds['name']}   All cells", OUT / "all_cells_firing_rate.png")
        draw(frame, SUBTYPES, f"{ds['name']}   ON / OFF / Biphasic", OUT / "on_off_biphasic_firing_rate.png")
    print(f"Done → {OUT}", flush=True)


if __name__ == "__main__":
    main()
