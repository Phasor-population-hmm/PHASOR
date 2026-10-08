#!/usr/bin/env python3
"""Three mode-summary figures, written directly into PHASOR_OUTPUT_ROOT.

  mode_summary_dwell_time_vs_contrast.png
  mode_summary_mode_count_vs_contrast.png
  <recording>_occupancy_dwell.png
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

plt.rcParams["axes.grid"] = False

BIN_MS = 10
REP_LEN_BINS = 3000
CONDITIONS = ["c0", "c50", "c60", "c70", "c80", "c90"]
CONTRAST_OF = {"c0": 0, "c50": 50, "c60": 60, "c70": 70, "c80": 80, "c90": 90}
OUT = Path(os.environ.get("PHASOR_OUTPUT_ROOT", "phasor_output"))

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("vb", _ROOT / "05_viterbi" / "viterbi_bands_wt22.py")
vb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)
_gs = importlib.util.spec_from_file_location("gcp", _ROOT / "05_viterbi" / "generate_contrast_progression_png.py")
gcp = importlib.util.module_from_spec(_gs)
_gs.loader.exec_module(gcp)


def load_reps(bundle: Path):
    data = np.load(bundle, allow_pickle=True)
    modes_train = np.asarray(data["modes_train"]).astype(int)
    modes_test = np.asarray(data["modes_test"]).astype(int)
    if modes_train.shape[0] % REP_LEN_BINS or modes_test.shape[0] % REP_LEN_BINS:
        raise ValueError(f"{bundle.name}: mode length is not divisible by {REP_LEN_BINS}")
    z_reps = np.concatenate(
        [
            modes_train.reshape(-1, REP_LEN_BINS),
            modes_test.reshape(-1, REP_LEN_BINS),
        ],
        axis=0,
    )
    contrast = np.asarray(data["contrast"]).astype(float)
    segments = vb.infer_segments_from_contrast(contrast, REP_LEN_BINS, BIN_MS)
    return z_reps, segments


def summarize(z_reps: np.ndarray, segments) -> pd.DataFrame:
    buckets = {name: [] for name in CONDITIONS}
    for rep in z_reps:
        for start_s, end_s, contrast_value in segments:
            name = vb.condition_name_from_contrast(contrast_value)
            if name not in buckets:
                continue
            start = int(round(start_s * 1000.0 / BIN_MS))
            end = int(round(end_s * 1000.0 / BIN_MS))
            runs = vb.segments_from_states(rep[start:end])
            dwells = [(stop - begin) * BIN_MS for begin, stop, _mode in runs]
            buckets[name].append((len(runs), float(np.mean(dwells)) if dwells else 0.0))
    rows = []
    for name in CONDITIONS:
        items = buckets[name]
        if not items:
            rows.append({
                "contrast_pct": CONTRAST_OF[name],
                "mean_mode_count": np.nan,
                "std_mode_count": np.nan,
                "mean_dwell_ms": np.nan,
                "std_dwell_ms": np.nan,
            })
            continue
        counts = np.array([item[0] for item in items], dtype=float)
        dwells = np.array([item[1] for item in items], dtype=float)
        spread = float(counts.std(ddof=1)) if len(items) > 1 else 0.0
        dwell_spread = float(dwells.std(ddof=1)) if len(items) > 1 else 0.0
        rows.append({
            "contrast_pct": CONTRAST_OF[name],
            "mean_mode_count": float(counts.mean()),
            "std_mode_count": spread,
            "mean_dwell_ms": float(dwells.mean()),
            "std_dwell_ms": dwell_spread,
        })
    return pd.DataFrame(rows)


def line_plot(x, y, yerr, xlabel, ylabel, title, path: Path, color: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.errorbar(x, y, yerr=yerr, color=color, lw=2.2, marker="o", ms=8, capsize=4)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.name}", flush=True)


def occupancy_plot(bundle: Path, display: str, path: Path) -> None:
    fingerprints = bundle.parent / "state_fingerprints.npz"
    if not fingerprints.exists():
        raise FileNotFoundError(f"Missing {fingerprints.name} next to the bundle")
    with np.load(fingerprints, allow_pickle=True) as data:
        records = list(data["state_fps"])
    states, occupancy, dwells = [], [], []
    for record in records:
        states.append(int(record["state_id"]))
        occupancy.append(float(record.get("occupancy_emp", record.get("occupancy", np.nan))))
        times = record.get("dwell_times_emp")
        if times is not None and len(times) > 0:
            values = np.asarray(times, dtype=float)
            dwells.append(values * 1000.0 if np.nanmedian(values) < 50 else values)
        else:
            dwells.append(np.array([]))
    order = np.argsort(states)
    states = [states[i] for i in order]
    occupancy = [occupancy[i] for i in order]
    dwells = [dwells[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    axes[0].bar([str(state) for state in states], occupancy, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("State ID", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Occupancy (fraction)", fontsize=12, fontweight="bold")
    axes[0].set_title("Occupancy per state", fontsize=13, fontweight="bold")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    drawn = [values for values in dwells if len(values) > 0]
    labels = [str(state) for state, values in zip(states, dwells) if len(values) > 0]
    parts = axes[1].violinplot(drawn, showmeans=True, showextrema=True, showmedians=False)
    for body in parts["bodies"]:
        body.set_facecolor("#4C72B0")
        body.set_alpha(0.55)
    axes[1].set_xticks(np.arange(1, len(labels) + 1))
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("State ID", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Dwell time (ms)", fontsize=12, fontweight="bold")
    axes[1].set_title("Dwell-time distribution per state", fontsize=13, fontweight="bold")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.suptitle(f"{display}  |  Mode occupancy & dwell", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.name}", flush=True)


def main() -> None:
    if not gcp.DATASETS:
        raise SystemExit("No dataset was selected.")
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in gcp.DATASETS:
        name = ds["name"]
        display = gcp.ds_display_name(name)
        bundle = Path(ds["bundle"])
        if not bundle.exists():
            raise FileNotFoundError(f"Missing bundle: {bundle}")
        print(f"\n===== {name} =====", flush=True)
        print(f"Out {OUT}", flush=True)
        table = summarize(*load_reps(bundle))
        x = table["contrast_pct"].to_numpy(dtype=float)
        line_plot(
            x, table["mean_dwell_ms"], table["std_dwell_ms"],
            "Contrast (%)", "Mean dwell time (ms)",
            f"{display}: Dwell Time",
            OUT / "mode_summary_dwell_time_vs_contrast.png",
            "#D35400",
        )
        line_plot(
            x, table["mean_mode_count"], table["std_mode_count"],
            "Contrast (%)", "Mean mode count",
            f"{display}: Mode Count",
            OUT / "mode_summary_mode_count_vs_contrast.png",
            "#1f77b4",
        )
        occupancy_plot(bundle, display, OUT / f"{name}_occupancy_dwell.png")


if __name__ == "__main__":
    main()
