#!/usr/bin/env python3
"""
viterbi_polar_12states_pizza.py

Same dataset/contrast logic as viterbi_polar_aggregated_states_summary.py,
but with two key differences:

  1.  12 equal sectors (30° each) instead of 8 → M1 ... M12
  2.  Stacked "pizza" bars instead of side-by-side:
        ON (bottom) / OFF (middle) / Biphasic (top)

Output groups
─────────────
  mean_active_count/         ON+OFF+Biphasic stacked  –  raw counts
  mean_active_percent/       ON+OFF+Biphasic stacked  –  composition %
  biphasic_count/            Biphasic only             –  raw counts
  biphasic_percent/          Biphasic only             –  composition %

  + one combined-comparison figure per group
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import os as _os
_OUT_ROOT = Path(_os.environ.get("PHASOR_OUTPUT_ROOT",
                 str(Path(__file__).resolve().parent.parent / "phasor_output")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


_ROOT    = Path(__file__).resolve().parent.parent
_VB_PATH = _ROOT / "05_viterbi" / "viterbi_bands_wt22.py"
_spec    = importlib.util.spec_from_file_location("vb", _VB_PATH)
vb       = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)

_GCP_PATH = _ROOT / "05_viterbi" / "generate_contrast_progression_png.py"
_gs       = importlib.util.spec_from_file_location("gcp", _GCP_PATH)
gcp       = importlib.util.module_from_spec(_gs)
_gs.loader.exec_module(gcp)


INCLUDE_DATASET_NAMES = {
    "wt1_run1", "wt1_run2",
    "wt2_run1", "wt2_run2",
    "wt3_run1", "wt3_run2",
    "wt4_run1", "wt4_run2", "wt4_ap4_run1", "wt4_washout_run1",
    "wt5_run1", "wt5_run2",
    "wt6_run1", "wt6_run2",
    "wt7_run1", "wt7_run2",
    "wt8_run1", "wt8_run2", "wt8_ap4_run1", "wt8_wash_run1", "wt8_wash_run2",
    "rd1_1", "rd1_2", "rd1_3", "rd1_1_post",
    "wt9_run1", "wt9_run2", "wt9_ap4_run1", "wt9_wash_run1",
}
OVERALL_DATASET_NAMES = {
    "wt1_run1", "wt1_run2",
    "wt2_run1", "wt2_run2",
    "wt3_run1", "wt3_run2",
    "wt4_run1", "wt4_run2",
    "wt5_run1", "wt5_run2",
    "wt6_run1", "wt6_run2",
    "wt7_run1", "wt7_run2",
    "wt8_run1", "wt8_run2",
}
if gcp.USING_DATASET_OVERRIDE:
    DATASETS = list(gcp.DATASETS)
    OVERALL_DATASETS = list(gcp.DATASETS)
else:
    DATASETS = [ds for ds in gcp.DATASETS if ds["name"] in INCLUDE_DATASET_NAMES]
    OVERALL_DATASETS = [ds for ds in gcp.DATASETS if ds["name"] in OVERALL_DATASET_NAMES]

BIN_MS         = 10
FREQ_HZ        = 2.0
BINS_PER_CYCLE = int(round(1000.0 / (FREQ_HZ * BIN_MS)))
SEG_LEN_BINS   = int(vb.SEG_LEN_BINS)
SKIP_BINS      = BINS_PER_CYCLE
N_CYCLES       = SEG_LEN_BINS // BINS_PER_CYCLE - 1
CONTRASTS      = [50, 60, 70, 80, 90]

COL_ON    = "#E69F00"
COL_OFF   = "#56B4E9"
COL_BIP   = "#009E73"
TYPE_ORDER  = ("ON", "OFF", "Biphasic")
TYPE_COLORS = (COL_ON, COL_OFF, COL_BIP)

N_AGG_STATES  = 12
AGG_LABELS    = [f"M{i}" for i in range(1, N_AGG_STATES + 1)]

R0         = 0.20
R_BAR_MAX  = 0.66
EXPORT_DPI = 900

_GRAD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "grad_border",
    [(0.00, "#081220"),
     (0.28, "#1f3b5a"),
     (0.52, "#3f6d86"),
     (0.76, "#8aa08e"),
     (1.00, "#f2e8c9")],
)
_GRAD_NORM = mcolors.Normalize(vmin=-1.0, vmax=1.0)

_PHASE = {
    0:   "Rising\n0°",
    90:  "Peak\n90°",
    180: "Falling\n180°",
    270: "Trough\n270°",
}

OUT_ROOT = _OUT_ROOT


def _sector_bins(sector_idx: int) -> np.ndarray:
    lo      = sector_idx * BINS_PER_CYCLE / N_AGG_STATES
    hi      = (sector_idx + 1) * BINS_PER_CYCLE / N_AGG_STATES
    centers = np.arange(BINS_PER_CYCLE, dtype=float) + 0.5
    return np.flatnonzero((centers >= lo) & (centers < hi)).astype(int)


SECTOR_BINS = [_sector_bins(i) for i in range(N_AGG_STATES)]


def _load_cd(ds: dict, contrast: int):
    try:
        cd = vb.load_condition_data(
            gcp.ds_run_dir(ds),
            vb.condition_name_from_contrast(contrast),
            FREQ_HZ,
        )
        return cd.flat_to_fourier, cd.raw_flat, cd.valid_flat_mask
    except Exception:
        return None, None, None


def _split3(cell_set: set[int], fm) -> tuple[int, int, int]:
    if fm is None:
        return len(cell_set), 0, 0
    on  = sum(1 for fi in cell_set if fm.get(fi) == "ON")
    off = sum(1 for fi in cell_set if fm.get(fi) == "OFF")
    bip = sum(1 for fi in cell_set if fm.get(fi) == "Biphasic")
    return on, off, bip


def _active_set(raw, valid, rep_idx: int, bins: np.ndarray) -> set[int]:
    if raw is None or len(bins) == 0:
        return set()
    active = np.any(raw[rep_idx][bins] > 0, axis=0) & valid
    return {int(fi) for fi in np.flatnonzero(active)}


def collect_contrast_stats_for_datasets(contrast: int, datasets: list[dict]) -> dict:
    """
    Returns dict with:
      counts  shape (12, 3)  mean active cell counts   per sector × subtype
      pct     shape (12, 3)  mean composition %         per sector × subtype
      n_datasets int
    """
    count_samples = [[[] for _ in TYPE_ORDER] for _ in range(N_AGG_STATES)]
    pct_samples   = [[[] for _ in TYPE_ORDER] for _ in range(N_AGG_STATES)]
    n_datasets    = 0

    for ds in datasets:
        bundle = Path(ds["bundle"])
        if not bundle.exists():
            continue
        fm, raw, valid = _load_cd(ds, contrast)
        if raw is None:
            continue
        n_datasets += 1
        for rep_idx in range(raw.shape[0]):
            for cycle_i in range(N_CYCLES):
                cyc0 = SKIP_BINS + cycle_i * BINS_PER_CYCLE
                for si, sbins in enumerate(SECTOR_BINS):
                    bins  = cyc0 + sbins
                    aset  = _active_set(raw, valid, rep_idx, bins)
                    on, off, bip = _split3(aset, fm)
                    total = max(on + off + bip, 1)
                    for ti, val in enumerate((on, off, bip)):
                        count_samples[si][ti].append(float(val))
                        pct_samples[si][ti].append(float(val) / total * 100.0)

    counts = np.zeros((N_AGG_STATES, 3), dtype=float)
    pct    = np.zeros_like(counts)
    for si in range(N_AGG_STATES):
        for ti in range(3):
            if count_samples[si][ti]:
                counts[si, ti] = float(np.mean(count_samples[si][ti]))
                pct[si, ti]    = float(np.mean(pct_samples[si][ti]))

    return {"counts": counts, "pct": pct, "n_datasets": n_datasets}


def collect_contrast_stats(contrast: int) -> dict:
    """Use OVERALL_DATASETS (baseline WT only) for cross-dataset averages."""
    return collect_contrast_stats_for_datasets(contrast, OVERALL_DATASETS)


def build_dataset_stats_cache(
    available: list[dict],
) -> tuple[dict[str, dict[int, dict]], float, float]:
    """
    Load each dataset once (all contrasts), compute global count scale, cache for plotting.
    Percent plots always use 75%.
    """
    cache: dict[str, dict[int, dict]] = {}
    max_stacked = 0.0
    max_any = 0.0
    n = len(available)
    for i, ds in enumerate(available, 1):
        name = ds["name"]
        per_c: dict[int, dict] = {}
        for c in CONTRASTS:
            st = collect_contrast_stats_for_datasets(c, [ds])
            per_c[c] = st
            if st["n_datasets"] == 0:
                continue
            max_stacked = max(
                max_stacked,
                float(np.max(np.sum(st["counts"], axis=1))),
            )
            max_any = max(max_any, float(np.max(st["counts"])))
        cache[name] = per_c
    peak = max(max_stacked, max_any)
    count_scale = max(75.0, float(np.ceil(peak / 25.0) * 25.0))
    return cache, count_scale, 75.0


def _draw_outer_gradient_border(ax, r_inner: float, r_outer: float):
    n  = 720
    tt = np.linspace(0.0, 2 * np.pi, n + 1)
    for i in range(n):
        z = np.sin((tt[i] + tt[i + 1]) * 0.5)
        ax.fill_between(
            [tt[i], tt[i + 1]],
            [r_inner, r_inner],
            [r_outer, r_outer],
            color=_GRAD_CMAP(_GRAD_NORM(z)),
            linewidth=0, zorder=0,
        )


def _draw_scale_rings(ax, scale_max: float, percent: bool):
    tt   = np.linspace(0, 2 * np.pi, 720)
    vals = [25.0, 50.0, 75.0]
    lbls = ["25%", "50%", "75%"] if percent else ["25", "50", "75"]
    for val, lbl in zip(vals, lbls):
        if val > scale_max:
            continue
        r = R0 + (val / scale_max) * R_BAR_MAX
        ax.plot(tt, np.full_like(tt, r),
                ls=(0, (2.0, 2.2)), lw=1.1, color="#777777", alpha=0.65, zorder=1)
        ax.text(
            np.radians(90), r, lbl,
            ha="center", va="center", fontsize=11.0, fontweight="bold",
            color="#111111",
            path_effects=[pe.withStroke(linewidth=1.4, foreground="#ffffff", alpha=0.75)],
            zorder=7,
        )


def _draw_phase_labels(ax, r_top: float):
    r_lbl = r_top + 0.28
    for deg, label in _PHASE.items():
        th = np.radians(deg)
        ax.text(
            th, r_lbl, label,
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#111111", zorder=8,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="#ffffff", alpha=0.7)],
        )


def draw_pizza_polar(
    ax,
    values: np.ndarray,
    scale_max: float,
    title: str,
    percent: bool,
    bip_only: bool = False,
    single_type: str | None = None,
):
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_axis_off()

    r_top = R0 + R_BAR_MAX
    _draw_outer_gradient_border(ax, r_top + 0.02, r_top + 0.07)
    _draw_scale_rings(ax, scale_max, percent)

    tt = np.linspace(0, 2 * np.pi, 720)
    ax.plot(tt, np.full_like(tt, R0),
            color="#999999", lw=1.1, ls="--", alpha=0.8, zorder=2)

    sector_width = 2 * np.pi / N_AGG_STATES

    for si, label in enumerate(AGG_LABELS):
        th0     = si * sector_width
        th_mid  = th0 + sector_width * 0.5
        th_bar  = th_mid

        if single_type is not None:
            ti_map = {"ON": 0, "OFF": 1, "Biphasic": 2}
            ti = ti_map[single_type]
            col = TYPE_COLORS[ti]
            val = float(values[si, ti] if values.ndim == 2 else values[si])
            h = min(max(val, 0.0), scale_max) / scale_max * R_BAR_MAX
            if h > 0:
                ax.bar(th_bar, h, width=sector_width * 0.80,
                       bottom=R0, color=col,
                       edgecolor="white", linewidth=0.35, alpha=0.94, zorder=4)
        elif bip_only:
            val = float(values[si]) if values.ndim == 1 else float(values[si, 2])
            h   = min(max(val, 0.0), scale_max) / scale_max * R_BAR_MAX
            if h > 0:
                ax.bar(th_bar, h, width=sector_width * 0.80,
                       bottom=R0, color=COL_BIP,
                       edgecolor="white", linewidth=0.35, alpha=0.94, zorder=4)
        else:
            bottom = R0
            for ti, col in enumerate(TYPE_COLORS):
                val = float(values[si, ti])
                h   = min(max(val, 0.0), scale_max) / scale_max * R_BAR_MAX
                if h <= 0:
                    continue
                ax.bar(th_bar, h, width=sector_width * 0.80,
                       bottom=bottom, color=col,
                       edgecolor="white", linewidth=0.35, alpha=0.94, zorder=4)
                bottom += h

        ax.plot([th0, th0], [0, r_top + 0.03],
                color="#d0d0d0", lw=0.65, alpha=0.9, zorder=2)
        ax.text(th_mid, r_top + 0.18, label,
                ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#111111")

    _draw_phase_labels(ax, r_top)
    ax.set_ylim(0, r_top + 0.52)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)


def make_single_figure(contrast: int, stats: dict, values_key: str,
                       scale_max: float, title: str,
                       percent: bool, bip_only: bool = False) -> plt.Figure:
    fig = plt.figure(figsize=(9.4, 9.4), facecolor="white")
    ax  = fig.add_subplot(111, projection="polar")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.05)

    draw_pizza_polar(ax, stats[values_key], scale_max, title, percent, bip_only)

    if bip_only:
        handles = [mpatches.Patch(facecolor=COL_BIP, label="Biphasic")]
    else:
        handles = [mpatches.Patch(facecolor=c, label=t)
                   for c, t in zip(TYPE_COLORS, TYPE_ORDER)]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.975, 0.965), fontsize=9, framealpha=0.95)
    fig.suptitle(
        f"All datasets  C{contrast}%  |  Aggregated States M1-M12  (30° each)",
        fontsize=12, fontweight="bold", y=0.97,
    )
    return fig


def make_combined_figure(all_stats: dict, values_key: str, scale_max: float,
                         title_prefix: str, percent: bool,
                         bip_only: bool = False) -> plt.Figure:
    fig = plt.figure(figsize=(26.0, 6.4), facecolor="white")
    gs  = fig.add_gridspec(1, 5, left=0.03, right=0.97,
                           top=0.86, bottom=0.10, wspace=0.22)
    for idx, contrast in enumerate(CONTRASTS):
        ax = fig.add_subplot(gs[0, idx], projection="polar")
        draw_pizza_polar(
            ax, all_stats[contrast][values_key],
            scale_max, f"C{contrast}", percent, bip_only,
        )

    if bip_only:
        handles = [mpatches.Patch(facecolor=COL_BIP, label="Biphasic")]
    else:
        handles = [mpatches.Patch(facecolor=c, label=t)
                   for c, t in zip(TYPE_COLORS, TYPE_ORDER)]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.985, 0.985), fontsize=10, framealpha=0.95)
    fig.suptitle(
        f"All datasets  |  12 Aggregated States  |  {title_prefix}",
        fontsize=16, fontweight="bold", y=0.975,
    )
    return fig


def make_combined_figure_single_type(all_stats: dict, values_key: str, scale_max: float,
                                     title_prefix: str, subtype: str) -> plt.Figure:
    fig = plt.figure(figsize=(26.0, 6.4), facecolor="white")
    gs = fig.add_gridspec(1, 5, left=0.03, right=0.97, top=0.86, bottom=0.10, wspace=0.22)
    for idx, contrast in enumerate(CONTRASTS):
        ax = fig.add_subplot(gs[0, idx], projection="polar")
        draw_pizza_polar(
            ax, all_stats[contrast][values_key],
            scale_max, f"C{contrast}", percent=False,
            single_type=subtype,
        )
    ti_map = {"ON": 0, "OFF": 1, "Biphasic": 2}
    col = TYPE_COLORS[ti_map[subtype]]
    fig.legend(
        handles=[mpatches.Patch(facecolor=col, label=subtype)],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        fontsize=10,
        framealpha=0.95,
    )
    fig.suptitle(f"{title_prefix} | {subtype} only", fontsize=16, fontweight="bold", y=0.975)
    return fig


def _save(fig, path: Path):
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    global EXPORT_DPI
    args = list(sys.argv[1:])
    if "--fast" in args:
        EXPORT_DPI = 220

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    available = [ds for ds in DATASETS if Path(ds["bundle"]).exists()]
    if not available:
        raise SystemExit("No bundle was found for this recording.")

    ds_cache, count_scale, _percent_scale = build_dataset_stats_cache(available)
    for ds in available:
        name = ds["name"]
        disp = gcp.ds_display_name(name)
        ds_stats = ds_cache[name]
        for contrast in CONTRASTS:
            stats = ds_stats[contrast]
            if stats["n_datasets"] == 0:
                continue
            _save(
                make_single_figure(contrast, stats, "counts", count_scale,
                                   f"{disp}  C{contrast} count", percent=False),
                OUT_ROOT / f"{name}_c{contrast}_count.png",
            )
        _save(
            make_combined_figure(ds_stats, "counts", count_scale,
                                 f"{disp} count", percent=False),
            OUT_ROOT / f"{name}_all_contrasts_count.png",
        )
        for subtype, stem in (("ON", "on"), ("OFF", "off"), ("Biphasic", "biphasic")):
            _save(
                make_combined_figure_single_type(ds_stats, "counts", count_scale, disp, subtype),
                OUT_ROOT / f"{name}_all_contrasts_{stem}_count.png",
            )
        print(f"saved count PNGs for {name} → {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
