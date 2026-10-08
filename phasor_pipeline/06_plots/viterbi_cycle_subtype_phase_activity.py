#!/usr/bin/env python3
"""
Cycle-averaged phase activity by Fourier subtype (ON / OFF / Biphasic).

**Mean population activity** at each phase bin:
  (sum of spike counts over all valid cells of that subtype) / (number of those cells),
averaged over repetitions, cycles, and contrasts C50 to C90.

Each subtype gets a **half-maximum band** along phase (contiguous bins ≥ 50% of
that subtype's own maximum): that gives **onset / offset in degrees and ms** and
**duration** for “when this subtype is elevated.” Mean population activity per
bin is the right intensity metric for that story; the band summarizes *where*
and *how long* without relying on a single noisy peak bin.

Dashed band edges (half-max range) are drawn **on each subtype row** of the
heatmap (ON / OFF / Biphasic in row colors). A **range strip** below the sine
shows the same half-max interval per subtype (three rows, not a blend).
Stimulus sine and range strip **share x** with the heatmap; the colorbar is
appended beside the heatmap so panel widths stay aligned for vertical readout.

Outputs: TIFF per dataset + OVERALL, dominant-segment CSV, band-summary CSV.

Run standalone or from viterbi_two_questions.main().
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
import os as _os
_OUT_ROOT = Path(_os.environ.get("PHASOR_OUTPUT_ROOT",
                 str(Path(__file__).resolve().parent.parent / "phasor_output")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
VB_PATH = ROOT / "05_viterbi" / "viterbi_bands_wt22.py"
spec = importlib.util.spec_from_file_location("vb", VB_PATH)
vb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vb)

GCP_PATH = ROOT / "05_viterbi" / "generate_contrast_progression_png.py"
gs = importlib.util.spec_from_file_location("gcp", GCP_PATH)
gcp = importlib.util.module_from_spec(gs)
gs.loader.exec_module(gcp)

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
DATASETS = (
    list(gcp.DATASETS)
    if gcp.USING_DATASET_OVERRIDE
    else [ds for ds in gcp.DATASETS if ds["name"] in INCLUDE_DATASET_NAMES]
)

PER_CONTRAST_NAMES = {
    "wt4_run1", "wt4_run2", "wt4_ap4_run1", "wt4_washout_run1",
    "wt8_run1", "wt8_run2", "wt8_ap4_run1", "wt8_wash_run1", "wt8_wash_run2",
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

FREQ_HZ = 2.0
BIN_MS = 10
BINS_PER_CYCLE = int(round(1000.0 / (FREQ_HZ * BIN_MS)))
SEG_LEN_BINS = int(vb.SEG_LEN_BINS)
SKIP_BINS = BINS_PER_CYCLE
N_CYCLES = SEG_LEN_BINS // BINS_PER_CYCLE - 1
DEG_PER_BIN = 360.0 / BINS_PER_CYCLE
CONTRASTS = [50, 60, 70, 80, 90]
SUBTYPES = ["ON", "OFF", "Biphasic"]

ACTIVITY_UNIT = "spikes · cell⁻¹ · (10 ms)⁻¹"

ST_COL = {"ON": "#E69F00", "OFF": "#56B4E9", "Biphasic": "#009E73"}

OUT_DIR = _OUT_ROOT


def _load_cd(ds, contrast):
    try:
        cd = vb.load_condition_data(
            gcp.ds_run_dir(ds),
            vb.condition_name_from_contrast(contrast),
            FREQ_HZ,
        )
        return cd.flat_to_fourier, cd.raw_flat, cd.valid_flat_mask
    except Exception:
        return None, None, None


def _indices_by_subtype(fm: dict) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {st: [] for st in SUBTYPES}
    for fi, lab in (fm or {}).items():
        if lab in out:
            out[lab].append(int(fi))
    return {st: np.array(v, dtype=int) for st, v in out.items()}


def _cell_counts(
    raw: np.ndarray,
    valid: np.ndarray,
    fm: dict,
    lo_bins: dict[str, int],
    hi_bins: dict[str, int],
) -> dict[str, tuple[int, int, int]]:
    """
    For each subtype return (active_in_band, active_outside_band, total_subtype).

    active_in_band     — valid subtype cells that fired in ≥1 bin INSIDE half-max band
                         across any rep/cycle.
    active_outside_band— valid subtype cells that fired in ≥1 bin OUTSIDE half-max band
                         across any rep/cycle.
    total_subtype      — all valid cells classified as that subtype.
    Percentages are computed relative to total_subtype.
    """
    idx = _indices_by_subtype(fm)
    n_rep, n_t, _ = raw.shape
    out: dict[str, tuple[int, int, int]] = {}
    for st in SUBTYPES:
        ind = idx[st]
        good = ind[valid[ind]] if ind.size > 0 else np.array([], dtype=int)
        total_st = int(good.size)
        if good.size == 0:
            out[st] = (0, 0, total_st)
            continue
        lo = lo_bins[st]
        hi = hi_bins[st]
        fired_in = np.zeros(good.size, dtype=bool)
        fired_out = np.zeros(good.size, dtype=bool)
        for rep in range(n_rep):
            for cy in range(N_CYCLES):
                for b in range(BINS_PER_CYCLE):
                    t_abs = SKIP_BINS + cy * BINS_PER_CYCLE + b
                    if t_abs >= n_t:
                        continue
                    spikes = raw[rep, t_abs, good] > 0
                    if lo <= b <= hi:
                        fired_in |= spikes
                    else:
                        fired_out |= spikes
        out[st] = (int(np.sum(fired_in)), int(np.sum(fired_out)), total_st)
    return out


def _population_mean_at_bin(
    raw: np.ndarray,
    valid: np.ndarray,
    indices: np.ndarray,
    phase_bin: int,
) -> float:
    if indices.size == 0:
        return 0.0
    good = indices[valid[indices]]
    if good.size == 0:
        return 0.0
    n_cells = float(good.size)
    vals: list[float] = []
    n_rep, n_t, _ = raw.shape
    for rep in range(n_rep):
        for cy in range(N_CYCLES):
            t_abs = SKIP_BINS + cy * BINS_PER_CYCLE + phase_bin
            if t_abs >= n_t:
                continue
            col = raw[rep, t_abs, good]
            vals.append(float(np.sum(col)) / n_cells)
    return float(np.mean(vals)) if vals else 0.0


def profile_one_contrast(raw, valid, fm) -> np.ndarray:
    idx = _indices_by_subtype(fm or {})
    P = np.zeros((BINS_PER_CYCLE, len(SUBTYPES)), dtype=float)
    for j, st in enumerate(SUBTYPES):
        ind = idx[st]
        for b in range(BINS_PER_CYCLE):
            P[b, j] = _population_mean_at_bin(raw, valid, ind, b)
    return P


def profile_dataset_one_contrast(ds, contrast) -> tuple[np.ndarray, dict[str, tuple[int, int, int]]] | tuple[None, None]:
    """Like profile_dataset but for a single contrast only."""
    if not Path(ds["bundle"]).exists():
        return None, None
    fm, raw, valid = _load_cd(ds, contrast)
    if raw is None or valid is None:
        return None, None
    P = profile_one_contrast(raw, valid, fm or {})
    lo_bins, hi_bins = {}, {}
    for j, st in enumerate(SUBTYPES):
        lo, hi = half_max_bin_range(P[:, j])
        lo_bins[st] = lo
        hi_bins[st] = hi
    counts = _cell_counts(raw, valid, fm or {}, lo_bins, hi_bins)
    return P, counts


def profile_dataset(ds) -> tuple[np.ndarray, dict[str, tuple[int, int, int]]] | tuple[None, None]:
    """Return (P, counts) where counts maps subtype -> (active_in_band, total_subtype, total_all_valid).

    P is averaged across contrasts C50-C90.
    counts are derived from the averaged P's half-max bands, using the last
    successfully loaded contrast's raw/valid/fm (cell identity is stable across contrasts).
    """
    if not Path(ds["bundle"]).exists():
        return None, None
    mats = []
    last_raw = last_valid = last_fm = None
    for c in CONTRASTS:
        fm, raw, valid = _load_cd(ds, c)
        if raw is None or valid is None:
            continue
        mats.append(profile_one_contrast(raw, valid, fm or {}))
        last_raw, last_valid, last_fm = raw, valid, fm or {}
    if not mats:
        return None, None
    P = np.mean(np.stack(mats, axis=0), axis=0)
    lo_bins = {}
    hi_bins = {}
    for j, st in enumerate(SUBTYPES):
        lo, hi = half_max_bin_range(P[:, j])
        lo_bins[st] = lo
        hi_bins[st] = hi
    counts = _cell_counts(last_raw, last_valid, last_fm, lo_bins, hi_bins)
    return P, counts


def dominant_segments(P: np.ndarray, min_bins: int = 2):
    dom = np.argmax(P, axis=1)
    segs = []
    s = 0
    prev_subtype = None
    while s < BINS_PER_CYCLE:
        k = int(dom[s])
        e = s
        while e + 1 < BINS_PER_CYCLE and int(dom[e + 1]) == k:
            e += 1
        if e - s + 1 >= min_bins:
            st = SUBTYPES[k]
            segs.append(
                {
                    "subtype": st,
                    "from_subtype": prev_subtype if prev_subtype is not None else "",
                    "to_subtype": st,
                    "bin_start": s,
                    "bin_end": e,
                    "deg_start": s * DEG_PER_BIN,
                    "deg_end_exclusive": (e + 1) * DEG_PER_BIN,
                    "time_start_ms": s * BIN_MS,
                    "time_end_ms": (e + 1) * BIN_MS,
                    "duration_ms": (e - s + 1) * BIN_MS,
                }
            )
            prev_subtype = st
        s = e + 1
    return segs


def half_max_bin_range(profile: np.ndarray) -> tuple[int, int]:
    """Bins lo..hi inclusive: contiguous band containing argmax, value >= half max."""
    peak = float(np.max(profile))
    bpk = int(np.argmax(profile))
    if peak <= 0:
        return bpk, bpk
    thr = 0.5 * peak
    lo = bpk
    while lo > 0 and float(profile[lo - 1]) >= thr:
        lo -= 1
    hi = bpk
    while hi < len(profile) - 1 and float(profile[hi + 1]) >= thr:
        hi += 1
    return lo, hi


def band_stats(prof: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """Mean and max population activity within inclusive bin range."""
    seg = prof[lo : hi + 1]
    if seg.size == 0:
        return 0.0, 0.0
    return float(np.mean(seg)), float(np.max(seg))


def _sanitize(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def band_summary_rows(dataset: str, P: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for j, st in enumerate(SUBTYPES):
        prof = P[:, j]
        lo, hi = half_max_bin_range(prof)
        mean_b, max_b = band_stats(prof, lo, hi)
        dur_ms = float((hi - lo + 1) * BIN_MS)
        rows.append(
            {
                "dataset": dataset,
                "subtype": st,
                "range_bin_lo": lo,
                "range_bin_hi": hi,
                "range_deg_start": round(float(lo * DEG_PER_BIN), 2),
                "range_deg_end": round(float((hi + 1) * DEG_PER_BIN), 2),
                "range_ms_start": round(float(lo * BIN_MS), 2),
                "range_ms_end": round(float((hi + 1) * BIN_MS), 2),
                "duration_ms": round(dur_ms, 2),
                "mean_population_activity_in_band": round(mean_b, 6),
                "max_population_activity_in_band": round(max_b, 6),
                "activity_unit": ACTIVITY_UNIT,
            }
        )
    return rows


def _draw_subtype_band_edges_on_heatmap(ax, P: np.ndarray) -> None:
    """Dashed verticals at each subtype's half-max band, only across that subtype's row."""
    for j, st in enumerate(SUBTYPES):
        lo, hi = half_max_bin_range(P[:, j])
        x0 = float(lo * BIN_MS)
        x1 = float((hi + 1) * BIN_MS)
        c = ST_COL[st]
        y0 = float(j)
        y1 = float(j + 1)
        ax.plot(
            [x0, x0],
            [y0, y1],
            color=c,
            ls="--",
            lw=2.0,
            alpha=0.95,
            zorder=5,
            clip_on=False,
        )
        ax.plot(
            [x1, x1],
            [y0, y1],
            color=c,
            ls="--",
            lw=2.0,
            alpha=0.95,
            zorder=5,
            clip_on=False,
        )


def _band_edge_vlines(P: np.ndarray) -> list[tuple[float, str]]:
    """(x_ms, subtype_key) for each half-max band edge, for aligned vlines on sine/range."""
    out: list[tuple[float, str]] = []
    for st in SUBTYPES:
        j = SUBTYPES.index(st)
        lo, hi = half_max_bin_range(P[:, j])
        out.append((float(lo * BIN_MS), st))
        out.append((float((hi + 1) * BIN_MS), st))
    return out


def _half_max_range_rgb_strip_shifted(P_orig: np.ndarray, phase_offset_bins: int) -> np.ndarray:
    """
    Range-strip image that correctly handles bands wrapping across the display boundary.
    Band positions are derived from the ORIGINAL (unrolled) P, then shifted to display
    coordinates — same logic as edge_vlines so colors and boundary lines always match.
    """
    white = np.ones(3, dtype=float)
    H = np.ones((len(SUBTYPES), BINS_PER_CYCLE, 3), dtype=float)
    for j, st in enumerate(SUBTYPES):
        lo_o, hi_o = half_max_bin_range(P_orig[:, j])
        lo_d = (lo_o - phase_offset_bins) % BINS_PER_CYCLE
        hi_d = (hi_o - phase_offset_bins) % BINS_PER_CYCLE
        rgb = np.asarray(to_rgb(ST_COL[st]), dtype=float)
        for b in range(BINS_PER_CYCLE):
            if lo_d <= hi_d:
                inside = lo_d <= b <= hi_d
            else:
                inside = b >= lo_d or b <= hi_d
            H[j, b] = rgb if inside else white
    return H


def _plot_profile(title: str, P: np.ndarray, out_path: Path, counts: dict[str, tuple[int, int, int]] | None = None):
    cycle_ms = float(BINS_PER_CYCLE * BIN_MS)

    PHASE_OFFSET_BINS = int(round(270.0 / 360.0 * BINS_PER_CYCLE))
    PHASE_OFFSET_DEG  = 270.0
    P_disp = np.roll(P, -PHASE_OFFSET_BINS, axis=0)

    xe = np.linspace(0.0, cycle_ms, BINS_PER_CYCLE + 1)
    ye_hm = np.linspace(0.0, float(len(SUBTYPES)), len(SUBTYPES) + 1)
    Z = P_disp.T

    def _shift_ms(xm_orig):
        """Shift an original-phase ms position into the rolled display ms."""
        return (xm_orig - PHASE_OFFSET_BINS * BIN_MS) % cycle_ms

    fig = plt.figure(figsize=(15.0, 12.0))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=2,
        width_ratios=[1.0, 0.038],
        height_ratios=[1.10, 1.45, 0.34, 0.52],
        wspace=0.04,
        hspace=0.12,
        left=0.07,
        right=0.96,
        top=0.91,
        bottom=0.07,
    )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis("off")

    show_counts = counts is not None
    COL_HEADERS = [
        "Subtype",
        "Phase\nrange (°)",
        "Time in\ncycle (ms)",
        "Duration\n(ms)",
        f"Mean in band\n({ACTIVITY_UNIT})",
        f"Max in band\n({ACTIVITY_UNIT})",
    ]
    if show_counts:
        COL_HEADERS += [
            "Active in band\nn  (%  of subtype)",
            "Active outside band\nn  (%  of subtype)",
        ]
    NCOLS = len(COL_HEADERS)

    table_rows = []
    for j, st in enumerate(SUBTYPES):
        prof = P[:, j]
        lo, hi = half_max_bin_range(prof)
        mean_b, max_b = band_stats(prof, lo, hi)
        dg0 = lo * DEG_PER_BIN
        dg1 = (hi + 1) * DEG_PER_BIN
        t0 = lo * BIN_MS
        t1 = (hi + 1) * BIN_MS
        dur = (hi - lo + 1) * BIN_MS

        row = [
            st,
            f"{dg0:.0f}–{dg1:.0f}",
            f"{t0:.0f}–{t1:.0f}",
            f"{dur:.0f}",
            f"{mean_b:.4f}",
            f"{max_b:.4f}",
        ]
        if show_counts and st in counts:
            n_in, n_out, total_st = counts[st]
            pct_in  = (100.0 * n_in  / total_st) if total_st > 0 else 0.0
            pct_out = (100.0 * n_out / total_st) if total_st > 0 else 0.0
            row += [f"{n_in} ({pct_in:.1f}%)", f"{n_out} ({pct_out:.1f}%)"]
        table_rows.append(row)

    tbl = ax1.table(
        cellText=table_rows,
        colLabels=COL_HEADERS,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 2.6)
    tbl.auto_set_column_width(list(range(NCOLS)))

    for j in range(NCOLS):
        cell = tbl[(0, j)]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        cell.set_height(cell.get_height() * 1.5)

    for i in range(1, len(table_rows) + 1):
        face = "#ECF0F1" if i % 2 == 0 else "white"
        for j in range(NCOLS):
            tbl[(i, j)].set_facecolor(face)

    subtitle = "Half-max band (≥50% of subtype cycle-max)."
    if show_counts:
        subtitle += "  Active counts are relative to that subtype's total valid cells."
    ax1.set_title(
        subtitle,
        fontsize=8.5,
        pad=6,
    )

    ax2 = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])
    vmax = max(float(P.max()), 1e-9)
    pcm = ax2.pcolormesh(
        xe,
        ye_hm,
        Z,
        shading="flat",
        cmap="inferno",
        vmin=0.0,
        vmax=vmax * 1.02,
        linewidth=0,
    )
    ax2.set_xlim(0.0, cycle_ms)
    ax2.set_ylim(0.0, float(len(SUBTYPES)))
    ax2.set_yticks(np.arange(len(SUBTYPES)) + 0.5)
    ax2.set_yticklabels(SUBTYPES, fontsize=12)
    for i, st in enumerate(SUBTYPES):
        ax2.get_yticklabels()[i].set_color(ST_COL[st])
        ax2.get_yticklabels()[i].set_fontweight("bold")
    ax2.set_ylabel("Subtype")
    ax2.tick_params(axis="x", labelbottom=False)
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label(f"Mean population activity\n({ACTIVITY_UNIT})", fontsize=9)

    edge_vlines_orig = _band_edge_vlines(P)
    edge_vlines = [(_shift_ms(xm), st_key) for xm, st_key in edge_vlines_orig]

    for xm_shifted, st_key in edge_vlines:
        j = SUBTYPES.index(st_key)
        ax2.plot(
            [xm_shifted, xm_shifted],
            [float(j), float(j + 1)],
            color=ST_COL[st_key],
            ls="--",
            lw=2.0,
            alpha=0.95,
            zorder=5,
            clip_on=True,
        )

    ax_sin = fig.add_subplot(gs[2, 0], sharex=ax2)
    t_s = np.linspace(0.0, cycle_ms / 1000.0, 2000, endpoint=False)
    start_phase_rad = 2.0 * np.pi * PHASE_OFFSET_DEG / 360.0
    ax_sin.plot(t_s * 1000.0,
                np.sin(2.0 * np.pi * FREQ_HZ * t_s + start_phase_rad),
                color="#222222", lw=1.6, zorder=2)
    ax_sin.set_ylim(-1.2, 1.2)
    ax_sin.set_ylabel("Stimulus sine (a.u.)", fontsize=9)
    ax_sin.set_yticks([-1, 0, 1])
    ax_sin.tick_params(axis="x", labelbottom=False)
    for xm_shifted, st_key in edge_vlines:
        ax_sin.axvline(xm_shifted, color=ST_COL[st_key], ls="--", lw=1.5, alpha=0.9, zorder=1)
    for step in range(5):
        orig_deg = (PHASE_OFFSET_DEG + step * 90.0) % 360.0
        xm = step * (cycle_ms / 4.0)
        sine_val = np.sin(2.0 * np.pi * FREQ_HZ * (xm / 1000.0) + start_phase_rad)
        ytxt = 1.08 if abs(sine_val) < 0.5 else -1.14
        va = "bottom" if ytxt > 0 else "top"
        ax_sin.annotate(
            f"{int(orig_deg)}°",
            xy=(xm, sine_val),
            xytext=(xm, ytxt),
            textcoords="data",
            ha="center", va=va, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#666666", linewidth=0.8),
        )

    ax_rng = fig.add_subplot(gs[3, 0], sharex=ax2)
    rng_img = _half_max_range_rgb_strip_shifted(P, PHASE_OFFSET_BINS)
    ax_rng.imshow(
        rng_img,
        aspect="auto",
        extent=(0.0, cycle_ms, 0.0, float(len(SUBTYPES))),
        origin="lower",
        interpolation="nearest",
        zorder=1,
    )
    ax_rng.set_ylim(0.0, float(len(SUBTYPES)))
    ax_rng.set_yticks(np.arange(len(SUBTYPES)) + 0.5)
    ax_rng.set_yticklabels([f"{s}\n(≥50% max)" for s in SUBTYPES], fontsize=9)
    for i, st in enumerate(SUBTYPES):
        ax_rng.get_yticklabels()[i].set_color(ST_COL[st])
    ax_rng.tick_params(axis="y", length=0)
    ax_rng.set_ylabel("Half-max range", fontsize=10)
    ms_ticks = np.array([step * cycle_ms / 4.0 for step in range(5)])
    deg_labels = [f"{int((PHASE_OFFSET_DEG + step * 90.0) % 360.0)}°" for step in range(5)]
    ax_rng.set_xticks(ms_ticks)
    ax_rng.set_xticklabels(deg_labels, fontsize=10)
    ax_rng.set_xlabel(
        "Phase in one 2 Hz cycle. Display starts at 270° (trough). "
        "Dashed lines: half-max band edges per subtype.",
        fontsize=10,
    )
    for xm_shifted, st_key in edge_vlines:
        ax_rng.axvline(xm_shifted, color=ST_COL[st_key], ls="--", lw=1.5, alpha=0.9, zorder=2)
    for x in ms_ticks:
        ax_rng.axvline(x, color="#BBBBBB", lw=0.75, ls=":", alpha=0.75, zorder=0)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per_contrast",
        action="store_true",
        help="Also generate one TIFF per contrast for datasets in PER_CONTRAST_NAMES",
    )
    args, _ = ap.parse_known_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    band_rows: list[dict] = []
    profiles: dict[str, np.ndarray] = {}

    for ds in DATASETS:
        name = ds["name"]
        P, counts = profile_dataset(ds)
        if P is None:
            continue
        profiles[name] = P
        band_rows.extend(band_summary_rows(name, P))
        segs = dominant_segments(P)
        for seg in segs:
            rows.append({"dataset": name, **seg})

        _plot_profile(
            f"{name}: subtype vs phase (mean C50-C90)",
            P,
            OUT_DIR / f"cycle_subtype_{_sanitize(name)}.png",
            counts=counts,
        )

        if args.per_contrast and name in PER_CONTRAST_NAMES:
            per_c_dir = OUT_DIR / "per_contrast" / _sanitize(name)
            per_c_dir.mkdir(parents=True, exist_ok=True)
            for c in CONTRASTS:
                P_c, counts_c = profile_dataset_one_contrast(ds, c)
                if P_c is None:
                    continue
                _plot_profile(
                    f"{name}  C{c}: subtype vs phase",
                    P_c,
                    per_c_dir / f"cycle_subtype_{_sanitize(name)}_c{c}.png",
                    counts=counts_c,
                )

    overall_profiles = {n: p for n, p in profiles.items() if n in OVERALL_DATASET_NAMES}
    if len(overall_profiles) > 1:
        stack = np.stack(list(overall_profiles.values()), axis=0)
        P_all = np.nanmean(stack, axis=0)
        n_ds = len(overall_profiles)
        band_rows.extend(band_summary_rows("OVERALL", P_all))
        for seg in dominant_segments(P_all):
            rows.append({"dataset": "OVERALL", **seg})
        _plot_profile(
            f"All baseline datasets (n={n_ds} run1/run2): subtype vs phase (mean C50-C90)",
            P_all,
            OUT_DIR / "cycle_subtype_OVERALL.png",
            counts=None,
        )

    if rows:
        df = pd.DataFrame(rows)
        csv_path = OUT_DIR / "cycle_subtype_dominant_segments.csv"
        df.to_csv(csv_path, index=False)
    if band_rows:
        pdf = pd.DataFrame(band_rows)
        pcsv = OUT_DIR / "cycle_subtype_band_summary.csv"
        pdf.to_csv(pcsv, index=False)


if __name__ == "__main__":
    main()
