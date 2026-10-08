#!/usr/bin/env python3
"""
Polar plots: ON/OFF × Sustained/Transient active-cell composition.

Uses Stage6 Sustained/Transient labels + Viterbi mode timing
(same 12-sector pizza layout as viterbi_polar_12states_pizza.py).

Supports both Stage6 folder layouts:
  Stage6_SustainedTransient_CellLevel/cell_level_sustained_transient_KEPT.csv
  Stage6_Cell_SustainedTransient/cell_level_sustained_transient.csv

Usage:
  python3 wt1_run1_polar_sustained_transient.py --dataset wt1_run1 [--fast]
  python3 wt1_run1_polar_sustained_transient.py --all [--fast]

Outputs under:
  Viterbi_results/<dataset>/viterbi_polar_sustained_transient/
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Arial resolves to Liberation Sans on this Linux system.
plt.rcParams["font.family"] = "Liberation Sans"

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

BIN_MS = 10
FREQ_HZ = 2.0
BINS_PER_CYCLE = int(round(1000.0 / (FREQ_HZ * BIN_MS)))  # 50
SEG_LEN_BINS = int(vb.SEG_LEN_BINS)
SKIP_BINS = BINS_PER_CYCLE
N_CYCLES = SEG_LEN_BINS // BINS_PER_CYCLE - 1
CONTRASTS = [50, 60, 70, 80, 90]

TYPE_ORDER = (
    "ON Transient",
    "ON Sustained",
    "OFF Transient",
    "OFF Sustained",
)
TYPE_COLORS = {
    "ON Transient": "#E69F00",
    "ON Sustained": "#D55E00",
    "OFF Transient": "#56B4E9",
    "OFF Sustained": "#0072B2",
}

N_AGG_STATES = 12
AGG_LABELS = [f"P{i}" for i in range(1, N_AGG_STATES + 1)]
R0 = 0.20
R_BAR_MAX = 0.66
EXPORT_DPI = 400
import os as _os
OUT_ROOT = Path(_os.environ.get("PHASOR_OUTPUT_ROOT", "phasor_output"))


def _resolve_dataset(ds: dict) -> dict:
    return dict(ds)


_GRAD_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "grad_border",
    [(0.00, "#081220"),
     (0.28, "#1f3b5a"),
     (0.52, "#3f6d86"),
     (0.76, "#8aa08e"),
     (1.00, "#f2e8c9")],
)
_GRAD_NORM = mcolors.Normalize(vmin=-1.0, vmax=1.0)
_PHASE = {0: "Rising\n0°", 180: "Falling\n180°"}

# Mutated per dataset run
DATASET = ""
RUN_DIR = Path(".")
OUT = Path(".")


def _sector_bins(sector_idx: int) -> np.ndarray:
    lo = sector_idx * BINS_PER_CYCLE / N_AGG_STATES
    hi = (sector_idx + 1) * BINS_PER_CYCLE / N_AGG_STATES
    centers = np.arange(BINS_PER_CYCLE, dtype=float) + 0.5
    return np.flatnonzero((centers >= lo) & (centers < hi)).astype(int)


SECTOR_BINS = [_sector_bins(i) for i in range(N_AGG_STATES)]


def _find_stage6_csv(run_dir: Path, contrast: int) -> Path | None:
    """Locate Stage6 ST CSV across Filtered_Data naming variants."""
    run_dir = Path(run_dir)
    filtered_roots = [
        run_dir / "Filtered_Data",
        run_dir / "Filtered Data",
        run_dir / "Filtered data",
    ]
    cond_variants = [f"c{contrast}", f"C-{contrast}", f"c-{contrast}", f"C{contrast}"]
    folders = [
        "Stage6_SustainedTransient_CellLevel",
        "Stage6_Cell_SustainedTransient",
    ]
    csv_names = [
        "cell_level_sustained_transient_KEPT.csv",
        "cell_level_sustained_transient.csv",
    ]
    extra_roots = [
        run_dir / "clustering" / "filtered",
    ]
    extra_conds = [f"c{contrast}_f2Hz", f"c{contrast}_f4Hz"]
    for root in filtered_roots + extra_roots:
        if not root.exists():
            continue
        conds = list(cond_variants)
        if root in extra_roots:
            conds = extra_conds + conds
        for cond in conds:
            for folder in folders:
                for csv_name in csv_names:
                    p = root / cond / folder / csv_name
                    if p.exists() and p.stat().st_size > 20:
                        return p
    return None


def dataset_has_stage6(run_dir: Path) -> bool:
    return any(_find_stage6_csv(run_dir, c) is not None for c in CONTRASTS)


def load_type_flat_map(contrast: int) -> dict[int, str]:
    """flat_index -> Stage6 Sustained/Transient class (Biphasic excluded)."""
    s6_path = _find_stage6_csv(RUN_DIR, contrast)
    lm_path = RUN_DIR / "label_index_maps" / f"c{contrast}_txtcells_label_index_map.csv"
    if s6_path is None or not lm_path.exists():
        return {}
    if s6_path.stat().st_size < 20:
        return {}

    try:
        s6 = pd.read_csv(s6_path)
    except (pd.errors.EmptyDataError, ValueError):
        return {}
    if s6.empty:
        return {}

    cell_col = "cell_index" if "cell_index" in s6.columns else "cell_idx"
    if cell_col not in s6.columns or "subtype" not in s6.columns or "polarity" not in s6.columns:
        return {}

    try:
        lm = pd.read_csv(lm_path).dropna(subset=["kept_index", "flat_index"]).copy()
    except (pd.errors.EmptyDataError, ValueError):
        return {}
    if lm.empty:
        return {}
    lm["kept_index"] = lm["kept_index"].astype(int)
    lm["flat_index"] = lm["flat_index"].astype(int)

    keep = s6[s6["subtype"].isin(["Sustained", "Transient"])].copy()
    keep = keep[keep["polarity"].isin(["ON", "OFF"])]
    joined = keep.merge(
        lm[["kept_index", "flat_index"]],
        left_on=cell_col,
        right_on="kept_index",
        how="inner",
    )
    out: dict[int, str] = {}
    for _, row in joined.iterrows():
        key = f"{row['polarity']} {row['subtype']}"
        out[int(row["flat_index"])] = key
    return out


def _active_set(raw, valid, rep_idx: int, bins: np.ndarray) -> set[int]:
    if raw is None or len(bins) == 0:
        return set()
    active = np.any(raw[rep_idx][bins] > 0, axis=0) & valid
    return {int(fi) for fi in np.flatnonzero(active)}


def _split_types(cell_set: set[int], type_map: dict[int, str]) -> np.ndarray:
    counts = np.zeros(len(TYPE_ORDER), dtype=float)
    for fi in cell_set:
        lab = type_map.get(fi)
        if lab is None:
            continue
        counts[TYPE_ORDER.index(lab)] += 1.0
    return counts


def _empty_stats() -> dict:
    return {
        "counts": np.zeros((N_AGG_STATES, len(TYPE_ORDER)), dtype=float),
        "pct": np.zeros((N_AGG_STATES, len(TYPE_ORDER)), dtype=float),
        "inventory": {k: 0 for k in TYPE_ORDER},
    }


def collect_contrast_stats(contrast: int) -> dict:
    type_map = load_type_flat_map(contrast)
    if not type_map:
        return _empty_stats()

    try:
        cd = vb.load_condition_data(RUN_DIR, f"c{contrast}", FREQ_HZ)
    except Exception as exc:
        print(f"  WARN C{contrast} load_condition_data failed: {exc}", flush=True)
        return _empty_stats()

    raw, valid = cd.raw_flat, cd.valid_flat_mask
    if raw is None:
        return _empty_stats()

    count_samples = [[] for _ in range(N_AGG_STATES)]
    pct_samples = [[] for _ in range(N_AGG_STATES)]

    for rep_idx in range(raw.shape[0]):
        for cycle_i in range(N_CYCLES):
            cyc0 = SKIP_BINS + cycle_i * BINS_PER_CYCLE
            for si, sbins in enumerate(SECTOR_BINS):
                bins = cyc0 + sbins
                aset = _active_set(raw, valid, rep_idx, bins)
                counts = _split_types(aset, type_map)
                total = max(float(counts.sum()), 1.0)
                count_samples[si].append(counts)
                pct_samples[si].append(counts / total * 100.0)

    counts = np.zeros((N_AGG_STATES, len(TYPE_ORDER)), dtype=float)
    pct = np.zeros_like(counts)
    for si in range(N_AGG_STATES):
        if count_samples[si]:
            counts[si] = np.mean(np.stack(count_samples[si], axis=0), axis=0)
            pct[si] = np.mean(np.stack(pct_samples[si], axis=0), axis=0)

    inv = {k: 0 for k in TYPE_ORDER}
    for lab in type_map.values():
        inv[lab] += 1

    return {"counts": counts, "pct": pct, "inventory": inv}


def nice_count_scale(peak: float) -> float:
    """Match polar_12states_averages: rings at 25/50/75, max ≥ 75 in steps of 25."""
    return max(75.0, float(np.ceil(float(peak) / 25.0) * 25.0))


def _draw_outer_gradient_border(ax, r_inner: float, r_outer: float):
    n = 720
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


# Font sizes: panel = 5-across combined (match wt8_wt10_baseline);
# single = one polar filling the figure (scaled so labels look the same relative size).
FONTS = {
    "panel":  {"scale": 11.0, "sector": 10.5, "phase": 11.5, "title": 13, "legend": 10, "stroke": 1.4},
    "single": {"scale": 18.0, "sector": 18.0, "phase": 18.0, "title": 20, "legend": 13, "stroke": 2.2},
}


def _draw_scale_rings(ax, scale_max: float, percent: bool, layout: str = "panel"):
    """Fixed 25 / 50 / 75 rings — same as viterbi_polar_12states_pizza.py."""
    fs = FONTS[layout]
    tt = np.linspace(0, 2 * np.pi, 720)
    vals = [25.0, 50.0, 75.0]
    lbls = ["25%", "50%", "75%"] if percent else ["25", "50", "75"]
    for val, lbl in zip(vals, lbls):
        if val > scale_max:
            continue
        r = R0 + (val / scale_max) * R_BAR_MAX
        ax.plot(
            tt, np.full_like(tt, r),
            ls=(0, (2.0, 2.2)), lw=1.1, color="#777777", alpha=0.65, zorder=1,
        )
        ax.text(
            np.radians(90), r, lbl,
            ha="center", va="center", fontsize=fs["scale"], fontweight="bold",
            color="#111111",
            path_effects=[pe.withStroke(linewidth=fs["stroke"], foreground="#ffffff", alpha=0.75)],
            zorder=7,
        )


def _draw_phase_labels(ax, r_top: float, layout: str = "panel"):
    fs = FONTS[layout]
    r_lbl = r_top + (0.32 if layout == "single" else 0.28)
    for deg, label in _PHASE.items():
        ax.text(
            np.radians(deg), r_lbl, label,
            ha="center", va="center",
            fontsize=fs["phase"], fontweight="bold", color="#111111", zorder=8,
            path_effects=[pe.withStroke(linewidth=fs["stroke"], foreground="#ffffff", alpha=0.7)],
        )


def draw_pizza(ax, values: np.ndarray, scale_max: float, title: str, percent: bool,
               single_type: str | None = None, layout: str = "panel"):
    fs = FONTS[layout]
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_axis_off()

    r_top = R0 + R_BAR_MAX
    _draw_outer_gradient_border(ax, r_top + 0.02, r_top + 0.07)
    _draw_scale_rings(ax, scale_max, percent, layout=layout)
    tt = np.linspace(0, 2 * np.pi, 720)
    ax.plot(tt, np.full_like(tt, R0), color="#999999", lw=1.1, ls="--", alpha=0.8, zorder=2)

    sector_width = 2 * np.pi / N_AGG_STATES
    label_r = r_top + (0.22 if layout == "single" else 0.18)
    for si, label in enumerate(AGG_LABELS):
        th0 = si * sector_width
        th_mid = th0 + sector_width * 0.5
        th_bar = th_mid

        if single_type is not None:
            ti = TYPE_ORDER.index(single_type)
            val = max(float(values[si, ti]), 0.0)
            h = min(val, scale_max) / scale_max * R_BAR_MAX
            if h > 0:
                ax.bar(
                    th_bar, h, width=sector_width * 0.80, bottom=R0,
                    color=TYPE_COLORS[single_type],
                    edgecolor="white", linewidth=0.35, alpha=0.94, zorder=4,
                    clip_on=True,
                )
        else:
            bottom = R0
            for ti, tname in enumerate(TYPE_ORDER):
                val = max(float(values[si, ti]), 0.0)
                h = min(val, scale_max) / scale_max * R_BAR_MAX
                if h <= 0:
                    continue
                remaining = (R0 + R_BAR_MAX) - bottom
                if remaining <= 0:
                    break
                h = min(h, remaining)
                ax.bar(
                    th_bar, h, width=sector_width * 0.80, bottom=bottom,
                    color=TYPE_COLORS[tname],
                    edgecolor="white", linewidth=0.35, alpha=0.94, zorder=4,
                    clip_on=True,
                )
                bottom += h

        ax.plot([th0, th0], [0, r_top + 0.03], color="#d0d0d0", lw=0.65, alpha=0.9, zorder=2)
        ax.text(
            th_mid, label_r, label,
            ha="center", va="center", fontsize=fs["sector"], fontweight="bold", color="#111111",
        )

    _draw_phase_labels(ax, r_top, layout=layout)
    ax.set_ylim(0, r_top + (0.62 if layout == "single" else 0.52))
    if title:
        ax.set_title(title, fontsize=fs["title"], fontweight="bold", pad=10)


def _legend_handles(single_type: str | None = None):
    if single_type is not None:
        return [mpatches.Patch(facecolor=TYPE_COLORS[single_type], label=single_type)]
    return [mpatches.Patch(facecolor=TYPE_COLORS[t], label=t) for t in TYPE_ORDER]


def make_single(contrast: int, stats: dict, values_key: str, scale_max: float,
                title: str, percent: bool, single_type: str | None = None) -> plt.Figure:
    fig = plt.figure(figsize=(9.4, 9.4), facecolor="white")
    ax = fig.add_subplot(111, projection="polar")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.05)
    draw_pizza(
        ax, stats[values_key], scale_max, f"C{contrast}", percent, single_type,
        layout="single",
    )
    fig.legend(
        handles=_legend_handles(single_type),
        loc="upper right", bbox_to_anchor=(0.975, 0.965),
        prop={"family": "Liberation Sans", "weight": "bold",
              "size": FONTS["single"]["legend"]},
        framealpha=0.95,
    )
    return fig


def make_combined(all_stats: dict, values_key: str, scale_max: float,
                  title_prefix: str, percent: bool,
                  single_type: str | None = None) -> plt.Figure:
    fig = plt.figure(figsize=(26.0, 6.6), facecolor="white")
    gs = fig.add_gridspec(1, 5, left=0.03, right=0.97, top=0.82, bottom=0.08, wspace=0.22)
    for idx, contrast in enumerate(CONTRASTS):
        ax = fig.add_subplot(gs[0, idx], projection="polar")
        draw_pizza(
            ax, all_stats[contrast][values_key],
            scale_max, f"C{contrast}", percent, single_type,
            layout="panel",
        )
    fig.legend(
        handles=_legend_handles(single_type),
        loc="upper right", bbox_to_anchor=(0.985, 0.985),
        prop={"family": "Liberation Sans", "weight": "bold",
              "size": FONTS["panel"]["legend"]},
        framealpha=0.95,
    )
    return fig


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.name}", flush=True)


def run_dataset(ds: dict) -> bool:
    global DATASET, RUN_DIR, OUT

    ds = _resolve_dataset(ds)
    DATASET = ds["name"]
    RUN_DIR = gcp.ds_run_dir(ds)
    bundle = Path(ds["bundle"])
    OUT = OUT_ROOT

    if not bundle.exists():
        print(f"SKIP {DATASET}: missing bundle {bundle}", flush=True)
        return False
    if not dataset_has_stage6(RUN_DIR):
        print(f"SKIP {DATASET}: no Stage6 Sustained/Transient CSVs", flush=True)
        return False

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n===== {DATASET} =====", flush=True)
    print(f"Bundle  {bundle}", flush=True)
    print(f"Run dir {RUN_DIR}", flush=True)
    print(f"Out     {OUT}", flush=True)

    all_stats = {}
    inv_rows = []
    for c in CONTRASTS:
        print(f"Collecting C{c} ...", flush=True)
        st = collect_contrast_stats(c)
        all_stats[c] = st
        row = {"contrast": c, **st["inventory"]}
        inv_rows.append(row)
        print(f"  inventory {st['inventory']}", flush=True)

    inv_df = pd.DataFrame(inv_rows)
    inv_df.to_csv(OUT / f"{DATASET}_sustained_transient_cell_counts.csv", index=False)

    max_stacked = max(
        float(np.max(np.sum(s["counts"], axis=1))) for s in all_stats.values()
    )
    count_scale = nice_count_scale(max_stacked)
    print(f"max_stacked={max_stacked:.2f}  count_scale={count_scale}", flush=True)

    for c in CONTRASTS:
        st = all_stats[c]
        _save(
            make_single(c, st, "counts", count_scale, "Mean active count", False),
            OUT / f"{DATASET}_c{c}_stacked_count.png",
        )

    _save(
        make_combined(all_stats, "counts", count_scale,
                      "stacked ON/OFF Sustained/Transient counts", False),
        OUT / f"{DATASET}_all_contrasts_stacked_count.png",
    )

    for tname in TYPE_ORDER:
        ti = TYPE_ORDER.index(tname)
        tmax = max(float(np.max(s["counts"][:, ti])) for s in all_stats.values())
        tscale = nice_count_scale(tmax)
        stem = tname.lower().replace(" ", "_")
        _save(
            make_combined(
                all_stats, "counts", tscale,
                f"{tname} active count", False, single_type=tname,
            ),
            OUT / f"{DATASET}_all_contrasts_{stem}_count.png",
        )

    bip = OUT / f"{DATASET}_all_contrasts_biphasic_count.png"
    if bip.exists():
        bip.unlink()

    readme = f"""# {DATASET} — Sustained / Transient polar plots

Classes (Stage6; Intermediate + Biphasic excluded):
- ON Transient / ON Sustained / OFF Transient / OFF Sustained

Colors match Stage6_SustainedTransient.

## Cell inventory
```
{inv_df.to_string(index=False)}
```
"""
    (OUT / "README.md").write_text(readme)
    print(f"Done → {OUT}", flush=True)
    return True


def main():
    global EXPORT_DPI
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--missing-only", action="store_true",
                        help="Skip datasets that already have stacked_count outputs")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

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
            done = OUT_ROOT / f"{name}_all_contrasts_stacked_count.png"
            if done.exists():
                print(f"SKIP {name}: already has outputs", flush=True)
                skipped.append(name)
                continue
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
