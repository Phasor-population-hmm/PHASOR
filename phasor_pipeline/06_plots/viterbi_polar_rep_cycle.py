#!/usr/bin/env python3
"""
Per-dataset polar plots for Rep 1 / Cycle 3 — Viterbi MODE segments.

For each contrast (first C0, then C50–C90):
  1. Take Viterbi mode path for Rep 1 inside that contrast window
  2. Restrict to cycle 3 (bins 100–149 of the 300-bin / 6-cycle segment)
  3. Split the cycle into contiguous MODE runs (segments_from_states)
  4. Each polar wedge:
       - angular width ∝ number of time bins in that mode run
       - outer ring colored with that MODE's color (build_state_color_map)
       - stacked bar = active ON / OFF / Biphasic cells during those bins
  5. Legend BELOW: mode colors (+ ON/OFF/Biphasic)

Output (6 PNGs per dataset):
  Viterbi_results/<dataset>/viterbi_polar_rep1_cycle3/
      <dataset>_c{0,50,60,70,80,90}_rep1_cycle3.png
"""
from __future__ import annotations

import argparse
import importlib.util
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

def _setup_arial():
    """Use real Arial if installed; else Liberation Sans (Arial-compatible)."""
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    path = next((c for c in candidates if Path(c).exists()), None)
    if path is None:
        path = fm.findfont(fm.FontProperties(family="Liberation Sans"))
    fm.fontManager.addfont(path)
    family = fm.FontProperties(fname=path).get_name()
    plt.rcParams.update({
        "font.family": family,
        "font.size": 20,
        "axes.titlesize": 14,
        "axes.labelsize": 20,
        "font.weight": "bold",
    })
    return path, family


_FONT_PATH, _FONT_FAMILY = _setup_arial()
print(f"[font] {_FONT_FAMILY}  ←  {_FONT_PATH}", flush=True)

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "vb", _ROOT / "05_viterbi" / "viterbi_bands_wt22.py"
)
vb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)

_gs = importlib.util.spec_from_file_location(
    "gcp", _ROOT / "05_viterbi" / "generate_contrast_progression_png.py"
)
gcp = importlib.util.module_from_spec(_gs)
_gs.loader.exec_module(gcp)

import os as _os
OUT_ROOT = Path(_os.environ.get("PHASOR_OUTPUT_ROOT", "phasor_output"))

BIN_MS = 10
FREQ_HZ = 2.0
BINS_PER_CYCLE = int(round(1000.0 / (FREQ_HZ * BIN_MS)))  # 50
SEG_LEN_BINS = int(vb.SEG_LEN_BINS)  # 300
REP_LEN_BINS = int(vb.REP_LEN_BINS)  # 3000

REP_IDX = 0
CYCLE_1BASED = 3
CYCLE0 = (CYCLE_1BASED - 1) * BINS_PER_CYCLE  # 100
CONTRASTS = [0, 50, 60, 70, 80, 90]

COL_ON = "#D4860A"
COL_OFF = "#1060A8"
COL_BIP = "#009E73"
TYPE_ORDER = ("ON", "OFF", "Biphasic")
TYPE_COLORS = (COL_ON, COL_OFF, COL_BIP)

R0 = 0.18
R_BAR_MAX = 0.62
R_OUTER = R0 + R_BAR_MAX
EXPORT_DPI = 400
# panel = 5-across row look; single = one full-figure polar (scaled up to match look)
FONTS = {
    "panel":  {"scale": 11.0, "mode": 10.5, "phase": 11.5, "title": 13, "stroke": 1.4},
    "single": {"scale": 26.0, "mode": 26.0, "phase": 26.0, "title": 22, "stroke": 2.8},
}
FONT = {"fontname": "Arial", "fontsize": 26, "fontweight": "bold"}


def nice_count_scale(peak: float) -> float:
    """Match polar_12states_averages: rings at 25/50/75, max ≥ 75 in steps of 25."""
    return max(75.0, float(np.ceil(float(peak) / 25.0) * 25.0))


def _resolve_dataset(ds: dict) -> dict:
    return dict(ds)


def load_z_reps_and_segments(bundle: Path):
    data = np.load(bundle, allow_pickle=True)
    modes_train = np.asarray(data["modes_train"])
    modes_test = np.asarray(data["modes_test"])
    if modes_train.shape[0] % REP_LEN_BINS or modes_test.shape[0] % REP_LEN_BINS:
        raise ValueError(f"Bad modes length in {bundle}")
    z_train = modes_train.reshape(-1, REP_LEN_BINS)
    z_test = modes_test.reshape(-1, REP_LEN_BINS)
    z_reps = np.concatenate([z_train, z_test], axis=0)

    if "contrast" in data and np.asarray(data["contrast"]).shape[0] >= REP_LEN_BINS:
        segments_s = vb.infer_segments_from_contrast(
            np.asarray(data["contrast"]), REP_LEN_BINS, BIN_MS
        )
    else:
        segments_s = list(vb.SEG_STRUCTURE_S)
    return z_reps, segments_s


def contrast_window_bins(segments_s, contrast: int) -> tuple[int, int]:
    """Return [start, end) bins in a rep for the requested contrast.
    For C0, use the FIRST C0 window only.
    """
    target = float(contrast)
    for ta, tb, c in segments_s:
        if abs(float(c) - target) < 1e-6:
            sa = int(round(ta * 1000.0 / BIN_MS))
            sb = int(round(tb * 1000.0 / BIN_MS))
            return sa, sb
    raise KeyError(f"No window for C{contrast}")


def _split3(cell_set: set[int], fm) -> np.ndarray:
    counts = np.zeros(3, dtype=float)
    if not cell_set or fm is None:
        return counts
    for fi in cell_set:
        lab = fm.get(fi)
        if lab == "ON":
            counts[0] += 1.0
        elif lab == "OFF":
            counts[1] += 1.0
        elif lab == "Biphasic":
            counts[2] += 1.0
    return counts


def _active_in_bins(raw, valid, rep_idx: int, bins: np.ndarray) -> set[int]:
    if raw is None or len(bins) == 0:
        return set()
    # condition raw is per-contrast (not full rep); use local bins
    bins = bins[(bins >= 0) & (bins < raw.shape[1])]
    if len(bins) == 0:
        return set()
    active = np.any(raw[rep_idx][bins] > 0, axis=0) & valid
    return {int(fi) for fi in np.flatnonzero(active)}


def collect_mode_segments(ds: dict, z_reps, segments_s, contrast: int, state_to_color: dict):
    """Build mode-segment stats for Rep1 / Cycle3 of one contrast.

    Cycle is treated as 0→360°. If a mode run continues from the previous
    cycle or into the next cycle, only the bins that fall inside THIS cycle
    are used for wedge width and for ON/OFF/Biphasic cell counts.
    """
    sa, sb = contrast_window_bins(segments_s, contrast)
    if sb - sa < CYCLE0 + BINS_PER_CYCLE:
        raise ValueError(f"C{contrast} window too short: {sa}-{sb}")

    if REP_IDX >= z_reps.shape[0]:
        raise ValueError("Rep 1 missing")

    # Strictly this cycle only (no previous / next cycle bins)
    abs_cyc0 = sa + CYCLE0
    abs_cyc1 = abs_cyc0 + BINS_PER_CYCLE
    mode_cyc = np.asarray(z_reps[REP_IDX, abs_cyc0:abs_cyc1]).astype(int)
    mode_segs = vb.segments_from_states(mode_cyc)  # already clipped to this cycle

    run_dir = gcp.ds_run_dir(ds)
    cd = vb.load_condition_data(
        run_dir, vb.condition_name_from_contrast(contrast), FREQ_HZ
    )
    raw, valid, fm = cd.raw_flat, cd.valid_flat_mask, cd.flat_to_fourier
    if raw is None or raw.shape[0] < 1:
        raise ValueError("No spike data")

    rows = []
    for lo, hi, mode in mode_segs:
        # Clip hard to [0, BINS_PER_CYCLE) — first/last mode may be edge-truncated
        lo_c = max(0, int(lo))
        hi_c = min(BINS_PER_CYCLE, int(hi))
        if hi_c <= lo_c:
            continue
        n_bins = hi_c - lo_c
        # Condition spike array is the 300-bin contrast segment → cycle-local bins
        local_bins = np.arange(CYCLE0 + lo_c, CYCLE0 + hi_c, dtype=int)
        aset = _active_in_bins(raw, valid, REP_IDX, local_bins)
        on_off_bip = _split3(aset, fm)
        rows.append({
            "mode": int(mode),
            "lo": lo_c,
            "hi": hi_c,
            "n_bins": n_bins,
            "counts": on_off_bip,
            "color": state_to_color.get(int(mode), "#BBBBBB"),
        })
    return {
        "segments": rows,
        "mode_cyc": mode_cyc,
        "window": (sa, sb),
        "inventory": {
            "ON": int(sum(r["counts"][0] for r in rows)),
            "OFF": int(sum(r["counts"][1] for r in rows)),
            "Biphasic": int(sum(r["counts"][2] for r in rows)),
        },
    }


def draw_mode_polar(ax, segments: list[dict], scale_max: float, layout: str = "single"):
    """Mode wedges + side-by-side ON / OFF / Biphasic count bars.

    - Angular WIDTH of each mode wedge = mode duration in this cycle
    - Inside each wedge: ON, OFF, Biphasic side-by-side (NOT stacked)
    - Radial HEIGHT of each bar = active-cell count (shared scale rings)
    - Outer rim = mode color (same as viterbi bands)
    """
    fs = FONTS[layout]
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_axis_off()

    total_bins = sum(max(s["n_bins"], 0) for s in segments) or BINS_PER_CYCLE
    theta = 0.0  # 0° → 360° through this cycle only
    r_mode_rim = 0.08
    r_bar_max = R_OUTER - r_mode_rim
    scale_max = max(float(scale_max), 1.0)
    n_types = len(TYPE_ORDER)

    # Count scale rings — fixed 25 / 50 / 75 like polar_12states_averages
    tt = np.linspace(0, 2 * np.pi, 720)
    for val in (25.0, 50.0, 75.0):
        if val > scale_max:
            continue
        r = R0 + (val / scale_max) * (r_bar_max - R0)
        ax.plot(
            tt, np.full_like(tt, r),
            ls=(0, (2.0, 2.4)), lw=0.9, color="#777777", alpha=0.55, zorder=1,
        )
        ax.text(
            np.radians(90), r, f"{int(val)}",
            ha="center", va="center",
            fontfamily=_FONT_FAMILY, fontsize=fs["scale"], fontweight="bold", color="#111111",
            path_effects=[pe.withStroke(linewidth=fs["stroke"], foreground="#ffffff", alpha=0.75)],
            zorder=7,
        )
    ax.plot(
        tt, np.full_like(tt, R0),
        color="#666666", lw=0.9, ls="--", alpha=0.55, zorder=2,
    )

    label_r = R_OUTER + (0.22 if layout == "single" else 0.18)
    phase_r = R_OUTER + (0.34 if layout == "single" else 0.28)
    for seg in segments:
        # Mode angular WIDTH (fills its time share; no gap to next mode)
        width = 2 * np.pi * (seg["n_bins"] / total_bins)
        th_mid = theta + width * 0.5
        mode_col = seg["color"]

        # Light mode floor across the full wedge
        ax.bar(
            th_mid, r_bar_max,
            width=width, bottom=0.0,
            color=mode_col, edgecolor="white", linewidth=0.8, alpha=0.20, zorder=2,
            align="center",
        )

        # Side-by-side ON / OFF / Biphasic — split the mode WIDTH into 3 bars
        bar_w = width / n_types
        for ti, col in enumerate(TYPE_COLORS):
            val = max(float(seg["counts"][ti]), 0.0)
            h = (val / scale_max) * (r_bar_max - R0)
            if h <= 0:
                continue
            th_bar = theta + (ti + 0.5) * bar_w
            ax.bar(
                th_bar, h, width=bar_w * 0.92, bottom=R0,
                color=col, edgecolor="white", linewidth=0.35, alpha=1.0, zorder=4,
                align="center",
            )

        # Outer rim = mode color (viterbi-band identity)
        ax.bar(
            th_mid, r_mode_rim,
            width=width, bottom=r_bar_max,
            color=mode_col, edgecolor="white", linewidth=0.8, alpha=1.0, zorder=5,
            align="center",
        )

        ax.text(
            th_mid, label_r,
            f"M{seg['mode']}",
            ha="center", va="center",
            fontfamily=_FONT_FAMILY, fontsize=fs["mode"], fontweight="bold", color="#111111",
            zorder=8,
        )
        theta += width

    for deg, label in ((0, "Rising\n0°"), (180, "Falling\n180°")):
        ax.text(
            np.radians(deg), phase_r, label,
            ha="center", va="center",
            fontfamily=_FONT_FAMILY, fontsize=fs["phase"], fontweight="bold", color="#111111",
            path_effects=[pe.withStroke(linewidth=fs["stroke"], foreground="#ffffff", alpha=0.7)],
            zorder=9,
        )

    ax.set_ylim(0, R_OUTER + (0.62 if layout == "single" else 0.52))


def make_figure(dataset: str, contrast: int, stats: dict, scale_max: float,
                state_to_color: dict, all_modes: list[int]) -> plt.Figure:
    fig = plt.figure(figsize=(9.4, 9.4), facecolor="white")
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.88], projection="polar")
    draw_mode_polar(ax, stats["segments"], scale_max, layout="single")
    ax.set_title(f"C{contrast}", fontsize=FONTS["single"]["title"], fontweight="bold", pad=10)
    return fig


def run_dataset(ds: dict) -> bool:
    ds = _resolve_dataset(ds)
    name = ds["name"]
    bundle = Path(ds["bundle"])
    out = OUT_ROOT

    if not bundle.exists():
        print(f"SKIP {name}: missing bundle", flush=True)
        return False

    z_reps, segments_s = load_z_reps_and_segments(bundle)
    state_to_color, _, all_modes = vb.build_state_color_map(z_reps, legend_top_n=None)

    out.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {name} =====", flush=True)
    print(f"Bundle  {bundle}", flush=True)
    print(f"Run dir {gcp.ds_run_dir(ds)}", flush=True)
    print(f"Out     {out}", flush=True)
    print(f"Modes   {all_modes}", flush=True)

    all_stats: dict[int, dict] = {}
    for c in CONTRASTS:
        print(f"Collecting C{c} (rep1, cycle3, mode segments) ...", flush=True)
        try:
            st = collect_mode_segments(ds, z_reps, segments_s, c, state_to_color)
        except Exception as exc:
            print(f"  WARN C{c}: {exc}", flush=True)
            continue
        all_stats[c] = st
        seg_txt = ", ".join(
            f"M{s['mode']}={s['n_bins']}b(ON{s['counts'][0]:.0f}/OFF{s['counts'][1]:.0f}/B{s['counts'][2]:.0f})"
            for s in st["segments"]
        )
        print(f"  {seg_txt}", flush=True)

    if not all_stats:
        print(f"SKIP {name}: no usable contrasts", flush=True)
        return False

    # Side-by-side bars → scale by max single-type count (not stacked sum)
    max_bar = 0.0
    for st in all_stats.values():
        for seg in st["segments"]:
            max_bar = max(max_bar, float(np.max(seg["counts"])))
    scale = nice_count_scale(max_bar)
    print(f"count_scale={scale} (max_bar={max_bar:.2f})", flush=True)

    for c in CONTRASTS:
        if c not in all_stats:
            print(f"  skip PNG C{c}", flush=True)
            continue
        fig = make_figure(name, c, all_stats[c], scale, state_to_color, all_modes)
        path = out / f"{name}_c{c}_rep{REP_IDX + 1}_cycle{CYCLE_1BASED}.png"
        fig.savefig(path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved {path.name}", flush=True)

    (out / "README.md").write_text(
        f"""# {name} — Rep1 / Cycle3 polar by Viterbi MODE segments

- Rep 1, cycle {CYCLE_1BASED}, contrasts C0 (first) + C50–C90
- Cycle = 0°→360°; wedges = contiguous modes (width ∝ duration in this cycle only)
- First/last mode: only bins inside this cycle (not previous/next cycle)
- Segment background = mode color; stack = ON / OFF / Biphasic
- Labels: M# and Rising/Falling in Arial 12; no legends / no bin counts
"""
    )
    print(f"Done → {out}", flush=True)
    return True


def main():
    global EXPORT_DPI
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--cycle", type=int, default=3)
    args = parser.parse_args()

    global REP_IDX, CYCLE_1BASED, CYCLE0
    REP_IDX = max(0, args.rep - 1)
    CYCLE_1BASED = max(1, args.cycle)
    CYCLE0 = (CYCLE_1BASED - 1) * BINS_PER_CYCLE
    print(f"Rep {args.rep}  cycle {args.cycle}", flush=True)

    if args.fast:
        EXPORT_DPI = 220
        print(f"[FAST] dpi={EXPORT_DPI}")

    if args.all or args.missing_only:
        datasets = list(gcp.DATASETS)
    elif args.dataset:
        datasets = [next(d for d in gcp.DATASETS if d["name"] == args.dataset)]
    else:
        datasets = list(gcp.DATASETS)

    ok, skipped, failed = [], [], []
    for ds in datasets:
        name = ds["name"]
        if args.missing_only:
            marker = (
                OUT_ROOT / name / "viterbi_polar_rep1_cycle3" / f"{name}_c50_rep1_cycle3.png"
            )
            if marker.exists():
                # force rebuild — previous version was wrong (phase pizza, not modes)
                pass
        try:
            if run_dataset(ds):
                ok.append(name)
            else:
                skipped.append(name)
        except Exception as exc:
            failed.append(name)
            print(f"FAIL {name}: {exc}", flush=True)
            traceback.print_exc()

    print("\n========== SUMMARY ==========")
    print(f"OK ({len(ok)}): {', '.join(ok)}")
    print(f"SKIP ({len(skipped)}): {', '.join(skipped)}")
    print(f"FAIL ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
