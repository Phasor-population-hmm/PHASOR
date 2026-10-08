#!/usr/bin/env python3
"""
viterbi_mea_subtype_heatmaps.py

For every registered dataset and every contrast (C50-C90), plots the
spatial distribution of ON, OFF and Biphasic cells on the 120-channel
MEA grid.

Data path:
  <run_dir>/clustering/filtered/c<N>_f2Hz/Stage5_Cell_Fourier_Analysis/cell_fourier_results.csv
  <run_dir>/label_index_maps/c<N>_txtcells_label_index_map.csv

Mapping:
  cell_fourier_results.cell_idx  ==  label_index_map.kept_index
  → txt_channel  (1-indexed, matches CHANNEL_INFO_120 order)

Output, written directly into PHASOR_OUTPUT_ROOT:
  mea_<subtype>_C50-C90.png
  mea_<subtype>_C50.png ... mea_<subtype>_C90.png
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import os as _os
_OUT_ROOT = Path(_os.environ.get("PHASOR_OUTPUT_ROOT",
                 str(Path(__file__).resolve().parent.parent / "phasor_output")))
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

_ROOT = Path(__file__).resolve().parent.parent
_GCP_PATH = _ROOT / "05_viterbi" / "generate_contrast_progression_png.py"
_gs = importlib.util.spec_from_file_location("gcp", _GCP_PATH)
gcp = importlib.util.module_from_spec(_gs)
_gs.loader.exec_module(gcp)

CONTRASTS  = [50, 60, 70, 80, 90]
SUBTYPES   = ["ON", "OFF", "Biphasic"]

ST_CMAPS = {
    "ON":       mcolors.LinearSegmentedColormap.from_list("on_cmap",  ["#FFFFFF", "#E69F00", "#B36B00"]),
    "OFF":      mcolors.LinearSegmentedColormap.from_list("off_cmap", ["#FFFFFF", "#56B4E9", "#0072B2"]),
    "Biphasic": mcolors.LinearSegmentedColormap.from_list("bip_cmap", ["#FFFFFF", "#4ECCA3", "#009E73"]),
}

EXPORT_DPI = 300
OUT_ROOT = _OUT_ROOT

CHANNEL_INFO_120: List[Tuple[Tuple[int, int], str]] = [
    ((5, 6), "F7"), ((5, 7), "F8"), ((5, 11), "F12"), ((5, 10), "F11"), ((5, 9), "F10"), ((5, 8), "F9"),
    ((4, 11), "E12"), ((4, 10), "E11"), ((4, 9), "E10"), ((4, 8), "E9"),
    ((3, 11), "D12"), ((3, 10), "D11"), ((3, 9), "D10"), ((3, 8), "D9"),
    ((2, 10), "C11"), ((2, 9), "C10"), ((1, 9), "B10"), ((4, 7), "E8"),
    ((2, 8), "C9"), ((1, 8), "B9"), ((0, 8), "A9"), ((3, 7), "D8"), ((2, 7), "C8"), ((1, 7), "B8"), ((0, 7), "A8"),
    ((3, 6), "D7"), ((2, 6), "C7"), ((1, 6), "B7"), ((0, 6), "A7"), ((4, 6), "E7"),
    ((5, 5), "F6"), ((4, 5), "E6"), ((0, 5), "A6"), ((1, 5), "B6"), ((2, 5), "C6"), ((3, 5), "D6"),
    ((0, 4), "A5"), ((1, 4), "B5"), ((2, 4), "C5"), ((3, 4), "D5"),
    ((0, 3), "A4"), ((1, 3), "B4"), ((2, 3), "C4"), ((3, 3), "D4"),
    ((1, 2), "B3"), ((2, 2), "C3"), ((2, 1), "C2"), ((4, 4), "E5"),
    ((3, 2), "D3"), ((3, 1), "D2"), ((3, 0), "D1"), ((4, 3), "E4"),
    ((4, 2), "E3"), ((4, 1), "E2"), ((4, 0), "E1"),
    ((5, 3), "F4"), ((5, 2), "F3"), ((5, 1), "F2"), ((5, 0), "F1"), ((5, 4), "F5"),
    ((6, 5), "G6"), ((6, 4), "G5"), ((6, 0), "G1"), ((6, 1), "G2"), ((6, 2), "G3"), ((6, 3), "G4"),
    ((7, 0), "H1"), ((7, 1), "H2"), ((7, 2), "H3"), ((7, 3), "H4"),
    ((8, 0), "J1"), ((8, 1), "J2"), ((8, 2), "J3"), ((8, 3), "J4"),
    ((9, 1), "K2"), ((9, 2), "K3"), ((10, 2), "L3"), ((7, 4), "H5"),
    ((9, 3), "K4"), ((10, 3), "L4"), ((11, 3), "M4"),
    ((8, 4), "J5"), ((9, 4), "K5"), ((10, 4), "L5"), ((11, 4), "M5"),
    ((8, 5), "J6"), ((9, 5), "K6"), ((10, 5), "L6"), ((11, 5), "M6"),
    ((7, 5), "H6"), ((6, 6), "G7"), ((7, 6), "H7"), ((11, 6), "M7"), ((10, 6), "L7"), ((9, 6), "K7"),
    ((8, 6), "J7"), ((11, 7), "M8"), ((10, 7), "L8"), ((9, 7), "K8"), ((8, 7), "J8"),
    ((11, 8), "M9"), ((10, 8), "L9"), ((9, 8), "K9"), ((8, 8), "J9"),
    ((10, 9), "L10"), ((9, 9), "K10"), ((9, 10), "K11"), ((7, 7), "H8"),
    ((8, 9), "J10"), ((8, 10), "J11"), ((8, 11), "J12"), ((7, 8), "H9"),
    ((7, 9), "H10"), ((7, 10), "H11"), ((7, 11), "H12"),
    ((6, 8), "G9"), ((6, 9), "G10"), ((6, 10), "G11"), ((6, 11), "G12"), ((6, 7), "G8"),
]

CHANNEL_MAP: Dict[int, Tuple[int, int]] = {
    idx + 1: coords for idx, (coords, _) in enumerate(CHANNEL_INFO_120)
}
GRID_ROWS = max(r for r, _ in CHANNEL_MAP.values()) + 1
GRID_COLS = max(c for _, c in CHANNEL_MAP.values()) + 1

VALID_POSITIONS = set(CHANNEL_MAP.values())


def _run_dir(ds: dict) -> Path:
    return gcp.ds_run_dir(ds)


def _fourier_csv(run_dir: Path, contrast: int) -> Optional[Path]:
    """
    Locate cell_fourier_results.csv for a given contrast, handling all known
    folder naming variants:
      - clustering/filtered/c50_f2Hz/Stage5_Cell_Fourier_Analysis/   (wt8/wt14/wt15)
      - Filtered_Data/c50/Stage5_Cell_Fourier_Analysis/               (wt2/wt11/wt12/wt13)
      - Filtered_Data/C-50/Stage5_Cell_Fourier_Analysis/              (wt7)
    """
    candidates = [
        run_dir / f"clustering/filtered/c{contrast}_f2Hz/Stage5_Cell_Fourier_Analysis/cell_fourier_results.csv",
        run_dir / f"Filtered_Data/c{contrast}/Stage5_Cell_Fourier_Analysis/cell_fourier_results.csv",
        run_dir / f"Filtered_Data/C-{contrast}/Stage5_Cell_Fourier_Analysis/cell_fourier_results.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    tag = str(contrast)
    for csv in run_dir.glob("**/Stage5_Cell_Fourier_Analysis/cell_fourier_results.csv"):
        parts = [p.lower() for p in csv.parts]
        if any(tag in part for part in parts):
            return csv
    return None


def _lmap_csv(run_dir: Path, contrast: int) -> Optional[Path]:
    p = run_dir / f"label_index_maps/c{contrast}_txtcells_label_index_map.csv"
    return p if p.exists() else None


def channel_counts_per_subtype(
    run_dir: Path, contrast: int
) -> Optional[Dict[str, Dict[int, int]]]:
    """
    Returns {subtype: {channel_num: count}} or None if data missing.
    channel_num is 1-indexed (matches CHANNEL_MAP keys).
    """
    f_csv = _fourier_csv(run_dir, contrast)
    l_csv = _lmap_csv(run_dir, contrast)
    if f_csv is None or l_csv is None:
        return None

    try:
        df_f = pd.read_csv(f_csv)
        df_m = pd.read_csv(l_csv)
    except Exception:
        return None

    if "label" not in df_f.columns or "kept_index" not in df_m.columns:
        return None

    df_f = df_f[df_f["label"].isin(SUBTYPES)].copy()
    if df_f.empty:
        return None

    df_m_valid = df_m.dropna(subset=["kept_index"]).copy()
    df_m_valid["kept_index"] = df_m_valid["kept_index"].astype(int)
    merged = df_f.merge(
        df_m_valid[["kept_index", "txt_channel"]],
        left_on="cell_idx",
        right_on="kept_index",
        how="left",
    )
    merged = merged.dropna(subset=["txt_channel"])
    merged["txt_channel"] = merged["txt_channel"].astype(int)

    result: Dict[str, Dict[int, int]] = {st: {} for st in SUBTYPES}
    for _, row in merged.iterrows():
        st = str(row["label"])
        ch = int(row["txt_channel"])
        if st in result and ch in CHANNEL_MAP:
            result[st][ch] = result[st].get(ch, 0) + 1

    return result


def _make_density_grid(channel_counts: Dict[int, int]) -> np.ndarray:
    """
    Convert channel→count dict to a Gaussian-smoothed density grid
    (same spatial logic as mea_activity_analysis.py reference).
    """
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=float)
    for ch, cnt in channel_counts.items():
        if ch in CHANNEL_MAP:
            r, c = CHANNEL_MAP[ch]
            grid[r, c] = float(cnt)

    high_res = 4
    hr_rows, hr_cols = GRID_ROWS * high_res, GRID_COLS * high_res
    density_hr = np.zeros((hr_rows, hr_cols), dtype=float)
    sigma_hr = 0.8 * high_res

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r, c] > 0:
                hr, hc = r * high_res + high_res // 2, c * high_res + high_res // 2
                y, x = np.ogrid[:hr_rows, :hr_cols]
                gaussian = np.exp(-((x - hc) ** 2 + (y - hr) ** 2) / (2 * sigma_hr ** 2))
                density_hr += grid[r, c] * gaussian

    density_hr = ndimage.gaussian_filter(density_hr, sigma=0.5 * high_res)
    density = ndimage.zoom(density_hr, 1.0 / high_res, order=1)
    return density


def _oriented_grid(grid: np.ndarray) -> np.ndarray:
    """Apply same orientation as reference: rot90 clockwise then fliplr."""
    return np.fliplr(np.rot90(grid, k=-1))


def _plot_mea_panel(
    ax: plt.Axes,
    density: np.ndarray,
    subtype: str,
    contrast: int,
    vmax: float,
    show_channels: bool = True,
) -> None:
    """Draw one MEA panel onto an existing axes."""
    cmap = ST_CMAPS[subtype]
    im = ax.imshow(
        _oriented_grid(density),
        cmap=cmap,
        vmin=0.0,
        vmax=max(vmax, 1e-9),
        aspect="equal",
        interpolation="bilinear",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"C{contrast}", fontsize=11, fontweight="bold", pad=4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.044, pad=0.02)
    cbar.set_ticks([0, max(vmax, 1)])
    cbar.set_ticklabels(["0", f"{vmax:.0f}"])
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("cell density (smoothed)", fontsize=7)

    if not show_channels:
        return

    rows = density.shape[0]
    cols = density.shape[1]
    for ch_num, (r, c) in CHANNEL_MAP.items():
        rot_r, rot_c = c, rows - 1 - r
        fin_r, fin_c = rot_r, cols - 1 - rot_c
        txt = ax.text(
            fin_c, fin_r,
            str(ch_num),
            ha="center", va="center",
            fontsize=4.5, color="#333333", fontweight="bold",
            zorder=5,
        )
        txt.set_alpha(0.65)


def save_subtype_figure(
    ds_name: str,
    subtype: str,
    per_contrast_counts: Dict[int, Optional[Dict[int, int]]],
    out_path: Path,
) -> None:
    """
    Create and save one TIFF with 5 panels side-by-side (C50..C90)
    for one subtype.
    """
    n_panels = len(CONTRASTS)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.0 * n_panels, 5.2),
        gridspec_kw={"wspace": 0.28},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{ds_name}   {subtype} cells   MEA spatial distribution",
        fontsize=13, fontweight="bold", y=1.01,
    )

    all_grids = []
    for c in CONTRASTS:
        ch_counts = per_contrast_counts.get(c)
        if ch_counts:
            all_grids.append(_make_density_grid(ch_counts))

    vmax = float(max(g.max() for g in all_grids)) if all_grids else 1.0
    vmax = max(vmax, 1.0)

    for ax, c in zip(axes, CONTRASTS):
        ch_counts = per_contrast_counts.get(c)
        density = _make_density_grid(ch_counts) if ch_counts else np.zeros((GRID_ROWS, GRID_COLS), dtype=float)
        _plot_mea_panel(ax, density, subtype, c, vmax, show_channels=True)

    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path.name}", flush=True)


def save_one_contrast(
    ds_name: str,
    subtype: str,
    contrast: int,
    channel_counts: Optional[Dict[int, int]],
    vmax: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 6.0))
    fig.patch.set_facecolor("white")
    density = _make_density_grid(channel_counts) if channel_counts else np.zeros((GRID_ROWS, GRID_COLS), dtype=float)
    _plot_mea_panel(ax, density, subtype, contrast, vmax, show_channels=True)
    fig.suptitle(f"{ds_name}   {subtype}   C{contrast}", fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path.name}", flush=True)


def main() -> None:
    args = sys.argv[1:]
    name_filter = None
    if args:
        name_filter = {a.lower() for a in args}

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for ds in gcp.DATASETS:
        name = ds["name"]
        if name_filter and not any(name.lower().startswith(f) for f in name_filter):
            continue

        bundle = Path(ds["bundle"])
        if not bundle.exists():
            continue

        run_dir = _run_dir(ds)

        per_contrast: Dict[int, Optional[Dict[str, Dict[int, int]]]] = {}
        for c in CONTRASTS:
            per_contrast[c] = channel_counts_per_subtype(run_dir, c)
            if per_contrast[c] is None:
                pass
            else:
                totals = {st: sum(per_contrast[c][st].values()) for st in SUBTYPES}

        if all(v is None for v in per_contrast.values()):
            continue

        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"\n===== {name} =====", flush=True)
        print(f"Out {OUT_ROOT}", flush=True)

        for subtype in SUBTYPES:
            st_counts: Dict[int, Optional[Dict[int, int]]] = {
                c: (per_contrast[c][subtype] if per_contrast[c] else None)
                for c in CONTRASTS
            }
            grids = [_make_density_grid(st_counts[c]) for c in CONTRASTS if st_counts.get(c)]
            vmax = float(max(g.max() for g in grids)) if grids else 1.0
            vmax = max(vmax, 1.0)
            save_subtype_figure(name, subtype, st_counts, OUT_ROOT / f"mea_{subtype}_C50-C90.png")
            for contrast in CONTRASTS:
                save_one_contrast(
                    name,
                    subtype,
                    contrast,
                    st_counts.get(contrast),
                    vmax,
                    OUT_ROOT / f"mea_{subtype}_C{contrast}.png",
                )


if __name__ == "__main__":
    main()
