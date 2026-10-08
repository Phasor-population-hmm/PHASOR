#!/usr/bin/env python3
"""
generate_contrast_progression_png.py

Outputs per dataset:

  1. <dataset>_contrast_all_reps.png
     Union active (≥1 spike in an occurrence): table + bars + subtype lines.

  1b. <dataset>_contrast_all_reps_min50pct_bins.png
     Stricter active: in every rep and every occurrence of that contrast, the cell
     must spike in ≥50% of that window's time bins (intersection across all slices).

  2. contrast_progression_pngs/<dataset>/rep_01.png  …  rep_NN.png
     One clean PNG per repetition — 10 blocks following actual stimulus order:
       c0 #1 → c50 → c0 #2 → c60 → c0 #3 → c70 → c0 #4 → c80 → c0 #5 → c90
     Each stimulus block: Active count(%), ON / OFF / Bip count (%).
     c0 blocks: grey — active count only.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import os as _os
_OUT_ROOT = Path(_os.environ.get("PHASOR_OUTPUT_ROOT",
                 str(Path(__file__).resolve().parent.parent / "phasor_output")))
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpec

import os
_VB = Path(os.environ.get("PHASOR_VITERBI_BANDS",
                          str(Path(__file__).resolve().parent / "viterbi_bands_wt22.py")))
_spec = importlib.util.spec_from_file_location("vb", _VB)
vb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vb)

_BUILTIN_DATASETS: List[Dict] = []


# Datasets can be overridden by the app via a JSON file: [{"name": ..., "bundle": ...}, ...]
import json as _json
_DS_OVERRIDE = os.environ.get("PHASOR_DATASETS_JSON")
USING_DATASET_OVERRIDE = bool(_DS_OVERRIDE and Path(_DS_OVERRIDE).exists())
if USING_DATASET_OVERRIDE:
    DATASETS: List[Dict] = [
        {
            "name": d["name"],
            "bundle": Path(d["bundle"]),
            **({"run_dir": Path(d["run_dir"])} if d.get("run_dir") else {}),
        }
        for d in _json.loads(Path(_DS_OVERRIDE).read_text())
    ]
else:
    DATASETS = _BUILTIN_DATASETS


DISPLAY_NAME_OVERRIDES: Dict[str, str] = {
    "wt1_run1":         "Wt1 Run 1",
    "wt1_run2":         "Wt1 Run 2",
    "wt2_run1":         "Wt2 Run 1",
    "wt2_run2":         "Wt2 Run 2",
    "wt3_run1":         "Wt3 Run 1",
    "wt3_run2":         "Wt3 Run 2",
    "wt4_run1":         "Wt4 Run 1",
    "wt4_run2":         "Wt4 Run 2",
    "wt4_ap4_run1":     "Wt4 AP4 Run 1",
    "wt4_washout_run1": "Wt4 Washout Run 1",
    "wt5_run1":         "Wt5 Run 1",
    "wt5_run2":         "Wt5 Run 2",
    "wt6_run1":         "Wt6 Run 1",
    "wt6_run2":         "Wt6 Run 2",
    "wt7_run1":         "Wt7 Run 1",
    "wt7_run2":         "Wt7 Run 2",
    "wt8_run1":         "Wt8 Run 1",
    "wt8_run2":         "Wt8 Run 2",
    "wt8_ap4_run1":     "Wt8 AP4 Run 1",
    "wt8_wash_run1":    "Wt8 Washout Run 1",
    "wt8_wash_run2":    "Wt8 Washout Run 2",
    "rd1_1":            "Rd1-1",
    "rd1_2":            "Rd1-2",
    "rd1_3":            "Rd1-3",
    "rd1_1_post":       "Rd1-1 Post",
    "wt9_run1":         "Wt9 Run 1",
    "wt9_run2":         "Wt9 Run 2",
    "wt9_ap4_run1":     "Wt9 AP4 Run 1",
    "wt9_wash_run1":    "Wt9 Washout 2.5 Run 1",
    "wt9_ap4_20_run1":  "Wt9 AP4 20 µM Run 1",
    "wt9_wash_20_run1": "Wt9 Washout 20 Run 1",
    "wt10_run1":         "Wt10 Run 1",
    "wt10_run2":         "Wt10 Run 2",
    "wt10_ap4_run1":     "Wt10 AP4 Run 1",
    "wt10_wash_run1":    "Wt10 Washout 2.5 Run 1",
    "wt10_ap4_20_run1":  "Wt10 AP4 20 µM Run 1",
    "wt10_wash_20_run1": "Wt10 Washout 20 Run 1",
    "wt10_no_projection_baseline": "No Projection",
    "wt11_run2":      "Wt11 Run 2",
    "wt11_ap4_run1":  "Wt11 AP4 Run 1",
    "wt11_wash_run1": "Wt11 Washout Run 1",
    "wt11_wash_run2": "Wt11 Washout Run 2",
}
OUT_DIR = _OUT_ROOT / "contrast_progression_pngs"

CONTRAST_ORDER = [0, 50, 60, 70, 80, 90]

EVERY_REP_MIN_BIN_FRACTION = 0.5

STIM_SEQ = [
    (0, 0), (50, 0),
    (0, 1), (60, 0),
    (0, 2), (70, 0),
    (0, 3), (80, 0),
    (0, 4), (90, 0),
]

C_FACE = {0: "#e0e0e0", 50: "#c7e2ff", 60: "#81bef7",
          70: "#3d8ee0", 80: "#1a4fa8", 90: "#06125e"}
C_TEXT = {0: "#666",    50: "#153580", 60: "#0a2860",
          70: "#ffffff", 80: "#ffffff", 90: "#ffffff"}
C_EDGE = {0: "#aaa",    50: "#3a6ab0", 60: "#1a4a90",
          70: "#1a6ab0", 80: "#0a2a7a", 90: "#080e3a"}

FOUC  = {"ON": "#e8a000", "OFF": "#1a72c0", "Biphasic": "#8030c0"}
FOUL  = {"ON": "#9a6800", "OFF": "#0a3a80", "Biphasic": "#4a1080"}


def pct_str(n: int, d: int, dec: int = 0) -> str:
    return f"{n / d * 100:.{dec}f}%" if d > 0 else "—"


def ds_run_dir(ds: dict) -> Path:
    """Directory containing Filtered_Data / clustering for a dataset."""
    if "run_dir" in ds:
        return Path(ds["run_dir"])
    return Path(ds["bundle"]).parent


def ds_display_name(raw: str) -> str:
    """Return the human-readable display name for an internal dataset name."""
    if raw in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[raw]
    mapping = {"ap4": "AP4", "washout": "Washout"}
    result = []
    for p in raw.replace("_", " ").split():
        if p in mapping:
            result.append(mapping[p])
        elif p.startswith("wt") and len(p) > 2:
            result.append("Wt" + p[2:])
        elif p.startswith("run") and len(p) > 3:
            result.append("Run " + p[3:])
        else:
            result.append(p.capitalize())
    return " ".join(result)


def ds_name_from_bundle(bundle_path) -> Optional[str]:
    """Resolve a bundle path back to its registered internal name."""
    bp = Path(bundle_path).resolve()
    for ds in DATASETS:
        if Path(ds["bundle"]).resolve() == bp:
            return ds["name"]
    return None


def ds_display_name_from_bundle(bundle_path) -> Optional[str]:
    """Return the display name for a dataset given its bundle path."""
    name = ds_name_from_bundle(bundle_path)
    return ds_display_name(name) if name is not None else None


def n_reps_from_bundle(p: Path) -> int:
    d = np.load(p, allow_pickle=True)
    rl = int(vb.REP_LEN_BINS)
    return (np.asarray(d["modes_train"]).shape[0] +
            np.asarray(d["modes_test"]).shape[0]) // rl


def load_cd(base: Path, c: int) -> Optional[vb.ConditionData]:
    try:
        return vb.load_condition_data(base, vb.condition_name_from_contrast(c), 2.0)
    except Exception:
        return None


def fourier_of(flat_set: Set[int], cd: vb.ConditionData) -> Dict[str, int]:
    counts = {"ON": 0, "OFF": 0, "Biphasic": 0}
    for fi in flat_set:
        lbl = cd.flat_to_fourier.get(fi)
        if lbl in counts:
            counts[lbl] += 1
    return counts


def active_in(cd: vb.ConditionData, occ: int) -> Set[int]:
    if occ >= len(cd.raw_flat):
        return set()
    fired = np.any(cd.raw_flat[occ] > 0, axis=0) & cd.kept_flat_mask
    return set(int(x) for x in np.flatnonzero(fired))


def cells_ge_bin_fraction_active(
    cd: vb.ConditionData, occ: int, min_frac: float
) -> Set[int]:
    """Kept cells that fire in ≥ min_frac of time bins in this occurrence window."""
    if occ >= len(cd.raw_flat):
        return set()
    win = cd.raw_flat[occ]
    n_bins = int(win.shape[0])
    if n_bins <= 0:
        return set()
    active_bins = np.sum(win > 0, axis=0).astype(np.float64) / float(n_bins)
    mask = (active_bins >= min_frac) & cd.kept_flat_mask
    return set(int(x) for x in np.flatnonzero(mask))


def per_rep_stats(n_reps: int, cond_map: Dict) -> Tuple[Dict, Dict]:
    occ_per_rep: Dict[int, int] = {}
    for c, cd in cond_map.items():
        total = len(cd.raw_flat)
        occ_per_rep[c] = total // n_reps if n_reps > 0 else 0

    per_rep: Dict = {}
    for ri in range(n_reps):
        per_rep[ri] = {}
        for c in CONTRAST_ORDER:
            cd = cond_map.get(c)
            k  = occ_per_rep.get(c, 0)
            if cd is None or k == 0:
                continue
            for oi in range(k):
                global_occ = ri * k + oi
                act  = active_in(cd, global_occ)
                four = fourier_of(act, cd) if c != 0 else {}
                per_rep[ri][(c, oi)] = {"active": act, "fourier": four}
    return per_rep, occ_per_rep


def all_reps_union(n_reps: int, cond_map: Dict, occ_per_rep: Dict) -> Dict:
    result: Dict = {}
    for c in CONTRAST_ORDER:
        cd = cond_map.get(c)
        k  = occ_per_rep.get(c, 0)
        if cd is None or k == 0:
            result[c] = {"active": set(), "silent": set(), "fourier": {}}
            continue
        ua: Set[int] = set()
        for ri in range(n_reps):
            for oi in range(k):
                ua |= active_in(cd, ri * k + oi)
        never = np.ones(cd.universe_size, dtype=bool)
        for i in range(len(cd.raw_flat)):
            never &= ~np.any(cd.raw_flat[i] > 0, axis=0)
        us = set(int(x) for x in np.flatnonzero(cd.kept_flat_mask & never))
        four = fourier_of(ua, cd) if c != 0 else {}
        result[c] = {"active": ua, "silent": us, "fourier": four}
    return result


def all_reps_every_occurrence_ge_bin_fraction(
    n_reps: int,
    cond_map: Dict,
    occ_per_rep: Dict,
    min_frac: float,
) -> Dict:
    """
    Per contrast: cells that satisfy (spike in ≥ min_frac of bins) in *every*
    rep and *every* occurrence slice for that contrast. Silent column unchanged
    from union definition (never fired in any occurrence of this condition).
    """
    result: Dict = {}
    for c in CONTRAST_ORDER:
        cd = cond_map.get(c)
        k  = occ_per_rep.get(c, 0)
        if cd is None or k == 0:
            result[c] = {"active": set(), "silent": set(), "fourier": {}}
            continue
        kept = set(int(x) for x in np.flatnonzero(cd.kept_flat_mask))
        ua = set(kept)
        for ri in range(n_reps):
            for oi in range(k):
                global_occ = ri * k + oi
                slice_ok = cells_ge_bin_fraction_active(cd, global_occ, min_frac)
                ua &= slice_ok
        never = np.ones(cd.universe_size, dtype=bool)
        for i in range(len(cd.raw_flat)):
            never &= ~np.any(cd.raw_flat[i] > 0, axis=0)
        us = set(int(x) for x in np.flatnonzero(cd.kept_flat_mask & never))
        four = fourier_of(ua, cd) if c != 0 else {}
        result[c] = {"active": ua, "silent": us, "fourier": four}
    return result


def make_single_rep_png(
    ds_name: str,
    rep_idx: int,
    n_reps: int,
    rep_data: Dict,
    occ_per_rep: Dict,
    total_cells: int,
) -> plt.Figure:
    """
    One row of 10 blocks following the actual stimulus sequence.
    Big, spacious blocks with Active count + ON/OFF/Bip chips.
    """
    seq = [(c, oi) for (c, oi) in STIM_SEQ if occ_per_rep.get(c, 0) > oi]
    n_cols = len(seq)

    BW    = 1.0
    ROW_H = 1.0

    fig_w = max(20, n_cols * 2.4 + 2.0)
    fig_h = 4.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#f0f3fa")
    ax.set_xlim(-0.08, n_cols * BW + 0.08)
    ax.set_ylim(-0.15, ROW_H + 0.85)
    ax.set_axis_off()

    for ci, (c, oi) in enumerate(seq):
        xc    = ci * BW + BW / 2
        label = f"c{c}%" if c != 0 else f"c0 #{oi + 1}"
        ax.add_patch(mpatches.FancyBboxPatch(
            (ci * BW + 0.05, ROW_H + 0.12), BW - 0.10, 0.52,
            boxstyle="round,pad=0.04", linewidth=0,
            facecolor=C_FACE[c], clip_on=False,
        ))
        ax.text(xc, ROW_H + 0.38, label,
                ha="center", va="center", fontsize=11,
                fontweight="bold", color=C_TEXT[c], clip_on=False)

    for ci, (c, oi) in enumerate(seq):
        x0    = ci * BW
        xc    = x0 + BW / 2
        stats = rep_data.get((c, oi))
        na    = len(stats["active"]) if stats else 0
        four  = stats["fourier"]     if stats else {}

        pad = 0.06
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0 + pad, 0.05), BW - 2 * pad, ROW_H - 0.10,
            boxstyle="round,pad=0.05",
            linewidth=1.0, edgecolor=C_EDGE[c], facecolor=C_FACE[c], zorder=2,
        ))

        if c == 0:
            ax.text(xc, 0.55,
                    f"{na}\n({pct_str(na, total_cells, 0)})",
                    ha="center", va="center", fontsize=14,
                    fontweight="bold", color="#555", linespacing=1.4, zorder=3)
        else:
            on_n  = four.get("ON",  0)
            off_n = four.get("OFF", 0)
            bip_n = four.get("Biphasic", 0)

            ax.text(xc, 0.78,
                    f"Active  {na}  ({pct_str(na, total_cells, 0)})",
                    ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=C_TEXT[c], zorder=3)

            chip_h  = 0.30
            chip_w  = (BW - 0.20) / 3
            chip_y0 = 0.10
            for ki, (lbl, fc, ec) in enumerate([
                (f"ON {on_n}\n({pct_str(on_n, na)})",        FOUC["ON"],       FOUL["ON"]),
                (f"OFF {off_n}\n({pct_str(off_n, na)})",      FOUC["OFF"],      FOUL["OFF"]),
                (f"Bip {bip_n}\n({pct_str(bip_n, na)})",      FOUC["Biphasic"], FOUL["Biphasic"]),
            ]):
                cx = x0 + 0.10 + ki * chip_w
                ax.add_patch(mpatches.FancyBboxPatch(
                    (cx, chip_y0), chip_w - 0.04, chip_h,
                    boxstyle="round,pad=0.025",
                    facecolor=fc, edgecolor=ec, linewidth=0.7, zorder=3,
                ))
                ax.text(cx + (chip_w - 0.04) / 2, chip_y0 + chip_h / 2,
                        lbl, ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color="white",
                        linespacing=1.35, zorder=4)

    disp = ds_display_name(ds_name)
    ax.set_title(
        f"Contrast Progression  ·  {disp}  ·  Rep {rep_idx + 1} / {n_reps}"
        f"   (total kept cells: {total_cells})",
        fontsize=13, fontweight="bold", color="#1a1a2e", pad=36,
    )

    legend_handles = [
        mpatches.Patch(facecolor=FOUC["ON"],       edgecolor=FOUL["ON"],       label="ON"),
        mpatches.Patch(facecolor=FOUC["OFF"],      edgecolor=FOUL["OFF"],      label="OFF"),
        mpatches.Patch(facecolor=FOUC["Biphasic"], edgecolor=FOUL["Biphasic"], label="Biphasic"),
        mpatches.Patch(facecolor="#e0e0e0",         edgecolor="#aaa",           label="c0 – no stimulus"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(1.0, -0.04),
              fontsize=9, framealpha=0.92, ncol=4, edgecolor="#ccc")

    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])
    return fig


def make_all_reps_png(
    ds_name: str,
    all_reps: Dict,
    total_cells: int,
    *,
    criterion_note: str = "",
) -> plt.Figure:
    contrasts = CONTRAST_ORDER
    actives  = [len(all_reps.get(c, {}).get("active", set())) for c in contrasts]
    silents  = [len(all_reps.get(c, {}).get("silent", set())) for c in contrasts]
    fouriers = [all_reps.get(c, {}).get("fourier", {})        for c in contrasts]

    fig = plt.figure(figsize=(16, 11), facecolor="#f0f3fa")
    gs  = GridSpec(2, 2, figure=fig,
                   height_ratios=[1.0, 1.5],
                   width_ratios=[1, 1],
                   hspace=0.40, wspace=0.30)
    ax_tbl   = fig.add_subplot(gs[0, :])
    ax_bar   = fig.add_subplot(gs[1, 0])
    ax_trend = fig.add_subplot(gs[1, 1])
    ax_tbl.set_axis_off()

    col_hdrs = ["Contrast", "Active", "Active %", "Silent", "Silent %",
                "ON  (% of Active)", "OFF  (% of Active)", "Biphasic  (% of Active)"]
    tbl_data = []
    for c, na, ns, f in zip(contrasts, actives, silents, fouriers):
        on  = f.get("ON",  0)
        off = f.get("OFF", 0)
        bip = f.get("Biphasic", 0)
        tbl_data.append([
            f"c{c}%",
            str(na),  pct_str(na, total_cells, 1),
            str(ns),  pct_str(ns, total_cells, 1),
            f"{on}  ({pct_str(on, na)})"   if c != 0 else "—",
            f"{off}  ({pct_str(off, na)})" if c != 0 else "—",
            f"{bip}  ({pct_str(bip, na)})" if c != 0 else "—",
        ])

    cell_clrs = [[to_rgba(C_FACE[c], alpha=0.60)] * len(col_hdrs) for c in contrasts]
    hdr_clr   = [[to_rgba("#1e1e2e")] * len(col_hdrs)]
    all_clrs  = hdr_clr + cell_clrs

    tbl = ax_tbl.table(
        cellText=[col_hdrs] + tbl_data,
        loc="center", cellLoc="center", cellColours=all_clrs,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 2.6)

    for ci in range(len(col_hdrs)):
        tbl[0, ci].set_text_props(color="white", fontweight="bold", fontsize=10)
    for ri in range(1, len(contrasts) + 1):
        c     = contrasts[ri - 1]
        txt_c = C_TEXT[c] if C_TEXT[c] != "#ffffff" else "#111111"
        for ci in range(len(col_hdrs)):
            tbl[ri, ci].set_text_props(color=txt_c, fontsize=10)

    disp = ds_display_name(ds_name)
    tbl_title = (
        f"Contrast Progression for All Reps  ·  {disp}   (total kept cells: {total_cells})"
    )
    if criterion_note:
        tbl_title += f"\n{criterion_note}"
    ax_tbl.set_title(tbl_title, fontsize=13, fontweight="bold", color="#1a1a2e", pad=14)

    x     = np.arange(len(contrasts))
    w     = 0.62
    on_v  = [f.get("ON",  0) if c != 0 else 0 for c, f in zip(contrasts, fouriers)]
    off_v = [f.get("OFF", 0) if c != 0 else 0 for c, f in zip(contrasts, fouriers)]
    bip_v = [f.get("Biphasic", 0) if c != 0 else 0 for c, f in zip(contrasts, fouriers)]
    unlbl = [max(0, na - on - off - bip)
             for na, on, off, bip in zip(actives, on_v, off_v, bip_v)]

    ax_bar.bar(x, on_v,  w, label="ON",               color=FOUC["ON"],   edgecolor=FOUL["ON"],       linewidth=0.7)
    ax_bar.bar(x, off_v, w, bottom=on_v,               label="OFF",        color=FOUC["OFF"],  edgecolor=FOUL["OFF"],      linewidth=0.7)
    bots  = [a + b for a, b in zip(on_v, off_v)]
    ax_bar.bar(x, bip_v, w, bottom=bots,               label="Biphasic",   color=FOUC["Biphasic"], edgecolor=FOUL["Biphasic"], linewidth=0.7)
    bots2 = [a + b + c_ for a, b, c_ in zip(on_v, off_v, bip_v)]
    ax_bar.bar(x, unlbl, w, bottom=bots2,              label="Unlabelled", color="#dddddd",    edgecolor="#aaaaaa",        linewidth=0.5)

    max_act = max(actives) if actives else 1
    for xi, na in zip(x, actives):
        if na > 0:
            ax_bar.text(xi, na + max_act * 0.015, str(na),
                        ha="center", va="bottom", fontsize=9.5,
                        fontweight="bold", color="#1a3a7a")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"c{c}%" for c in contrasts], fontsize=10)
    ax_bar.set_ylabel("Cell count", fontsize=10)
    ax_bar.set_title("Active Cell Breakdown", fontsize=11, fontweight="bold", color="#1a1a2e")
    ax_bar.set_facecolor("white")
    ax_bar.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.18),
        fontsize=8.5, framealpha=0.9, ncol=2,
    )
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    stim_idx = [i for i, c in enumerate(contrasts) if c != 0]
    stim_c   = [contrasts[i] for i in stim_idx]
    xs       = np.arange(len(stim_idx))

    on_pct  = [on_v[i]  / max(actives[i], 1) * 100 for i in stim_idx]
    off_pct = [off_v[i] / max(actives[i], 1) * 100 for i in stim_idx]
    bip_pct = [bip_v[i] / max(actives[i], 1) * 100 for i in stim_idx]

    ax_trend.plot(xs, on_pct,  "o-", color=FOUC["ON"],       lw=2.5, ms=9,
                  label="ON",       mec=FOUL["ON"],       mew=1.5)
    ax_trend.plot(xs, off_pct, "s-", color=FOUC["OFF"],      lw=2.5, ms=9,
                  label="OFF",      mec=FOUL["OFF"],      mew=1.5)
    ax_trend.plot(xs, bip_pct, "^-", color=FOUC["Biphasic"], lw=2.5, ms=9,
                  label="Biphasic", mec=FOUL["Biphasic"], mew=1.5)

    for xs_i, (op, fp, bp) in enumerate(zip(on_pct, off_pct, bip_pct)):
        ax_trend.annotate(f"{op:.0f}%", (xs_i, op),
                          xytext=(0, 7), textcoords="offset points",
                          ha="center", fontsize=8.5, color=FOUL["ON"], fontweight="bold")
        ax_trend.annotate(f"{fp:.0f}%", (xs_i, fp),
                          xytext=(0, -13), textcoords="offset points",
                          ha="center", fontsize=8.5, color=FOUL["OFF"], fontweight="bold")
        ax_trend.annotate(f"{bp:.0f}%", (xs_i, bp),
                          xytext=(7, 2), textcoords="offset points",
                          ha="center", fontsize=8.5, color=FOUL["Biphasic"], fontweight="bold")

    ax_trend.set_xticks(xs)
    ax_trend.set_xticklabels([f"c{c}%" for c in stim_c], fontsize=10)
    ax_trend.set_ylabel("% of active cells", fontsize=10)
    ax_trend.set_ylim(0, 105)
    ax_trend.set_title("Subtype Transition Across Contrasts\n(% of active cells per contrast)",
                        fontsize=11, fontweight="bold", color="#1a1a2e")
    ax_trend.set_facecolor("white")
    ax_trend.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_trend.spines["top"].set_visible(False)
    ax_trend.spines["right"].set_visible(False)

    sup = f"Contrast Progression for All Reps  ·  {disp}"
    if criterion_note:
        sup += f"  ·  {criterion_note}"
    fig.suptitle(sup, fontsize=15, fontweight="bold", color="#1a1a2e", y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 1.0])
    return fig


def process_dataset(ds: dict, out_dir: Path) -> None:
    name, bundle = ds["name"], ds["bundle"]
    if not bundle.exists():
        return
    base  = ds_run_dir(ds)
    n_reps = n_reps_from_bundle(bundle)
    cond_map = {c: load_cd(base, c) for c in CONTRAST_ORDER}
    cond_map = {c: v for c, v in cond_map.items() if v is not None}
    if not cond_map:
        return
    total_cells = int(next(iter(cond_map.values())).valid_flat_mask.sum())
    per_rep, occ_per_rep = per_rep_stats(n_reps, cond_map)
    all_reps = all_reps_union(n_reps, cond_map, occ_per_rep)
    all_reps_strict = all_reps_every_occurrence_ge_bin_fraction(
        n_reps, cond_map, occ_per_rep, EVERY_REP_MIN_BIN_FRACTION
    )
    pct_int = int(round(EVERY_REP_MIN_BIN_FRACTION * 100))
    strict_note = (
        f"Active = ≥{pct_int}% of bins with a spike in every rep & every occurrence "
        f"of that contrast"
    )

    try:
        fig = make_all_reps_png(name, all_reps, total_cells)
        p   = out_dir / f"{name}_contrast_all_reps.png"
        fig.savefig(p, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception as e:
        import traceback; traceback.print_exc()

    try:
        fig = make_all_reps_png(
            name, all_reps_strict, total_cells, criterion_note=strict_note
        )
        p   = out_dir / f"{name}_contrast_all_reps_min{pct_int}pct_bins.png"
        fig.savefig(p, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception as e:
        import traceback; traceback.print_exc()

    rep_dir = out_dir / name
    rep_dir.mkdir(parents=True, exist_ok=True)
    for ri in range(n_reps):
        try:
            fig = make_single_rep_png(
                name, ri, n_reps, per_rep.get(ri, {}), occ_per_rep, total_cells
            )
            p   = rep_dir / f"rep_{ri + 1:02d}.png"
            fig.savefig(p, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
        except Exception as e:
            import traceback; traceback.print_exc()


def main() -> None:
    filt = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        if filt and ds["name"] not in filt:
            continue
        process_dataset(ds, OUT_DIR)


if __name__ == "__main__":
    main()
