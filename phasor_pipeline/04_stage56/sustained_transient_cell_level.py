#!/usr/bin/env python3
"""
Cell-level sustained/transient labeling from Fourier ON/OFF cells.

Inputs:
- Stage4_Visualization/X_processed.npy
- Stage5_Cell_Fourier_Analysis/cell_fourier_results.csv

Output:
- Stage6_Cell_SustainedTransient/ (CSV + plots)

Method:
- Analyze cycles 2-6 by default.
- Keep ON/OFF cells that pass SNR and phase-lock checks.
- Compute per-cycle effective response width and average it.
- Label by width threshold: Transient (< transient_ms) or Sustained (>= sustained_ms).
- Biphasic cells are excluded.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.stats import circstd

EPS = 1e-9


def _baseline_subtract(x: np.ndarray) -> np.ndarray:
    return (x - float(np.mean(x))).astype(np.float32)


def _window_bins(bin_ms: float, polarity: str, on_window_ms: Tuple[float, float], off_window_ms: Tuple[float, float]) -> Tuple[int, int, bool]:
    if polarity.upper() == "OFF":
        w0, w1 = off_window_ms
    else:
        w0, w1 = on_window_ms
    wrap = bool(w0 > w1)
    b0 = int(np.floor(w0 / bin_ms))
    b1 = int(np.ceil(w1 / bin_ms))
    b0 = max(0, min(49, b0))
    b1 = max(0, min(50, b1))
    if not wrap:
        b1 = max(b0 + 1, b1)
    return b0, b1, wrap


def _window_slice(x: np.ndarray, b0: int, b1: int, wrap: bool) -> np.ndarray:
    if not wrap:
        return x[b0:b1]
    return np.concatenate([x[b0:], x[:b1]])


def _dominant_deflection_sign(w: np.ndarray) -> int:
    if w.size == 0:
        return 1
    pos = float(np.max(w))
    neg = float(np.max(-w))
    return 1 if pos >= neg else -1


def _longest_true_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _cycle_a1_phi(cycle_signed: np.ndarray, period_bins: int) -> Tuple[float, float]:
    F = fft(cycle_signed)
    c1 = F[1] / period_bins
    a1 = float(np.abs(c1))
    phi = float(np.angle(c1, deg=True)) % 360.0
    return a1, phi


def _robust_std(x: np.ndarray) -> float:
    if x.size == 0:
        return 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s < EPS:
        s = float(np.std(x))
    if not np.isfinite(s) or s < EPS:
        s = 1.0
    return float(s)


def _close_small_gaps(mask: np.ndarray, max_gap_bins: int) -> np.ndarray:
    if max_gap_bins <= 0 or mask.size == 0:
        return mask
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        gap_len = j - i
        if i > 0 and j < n and gap_len <= max_gap_bins:
            out[i:j] = True
        i = j
    return out


def _cycle_effective_width_metrics(
    cycle_signed: np.ndarray,
    b0: int,
    b1: int,
    wrap: bool,
    bin_ms: float,
    peak_floor: float,
) -> Dict[str, float]:
    w = _window_slice(cycle_signed, b0, b1, wrap)
    if w.size == 0:
        return {
            "peak": 0.0,
            "baseline": 0.0,
            "auc_ms": 0.0,
            "width_ms": 0.0,
        }

    peak = float(np.max(w))

    if not wrap:
        bg = np.concatenate([cycle_signed[:b0], cycle_signed[b1:]])
    else:
        bg = cycle_signed[b1:b0]
    if bg.size < 5:
        bg = cycle_signed
    baseline = float(np.median(bg))
    noise = _robust_std(bg - baseline)

    if peak < peak_floor:
        return {
            "peak": peak,
            "baseline": baseline,
            "auc_ms": 0.0,
            "width_ms": 0.0,
        }

    resp = w - baseline
    resp_pos = np.maximum(resp, 0.0)
    auc_ms = float(np.sum(resp_pos) * bin_ms)
    peak_above = float(max(0.0, peak - baseline))
    width_ms = float(auc_ms / (peak_above + EPS))

    return {
        "peak": peak,
        "baseline": baseline,
        "auc_ms": auc_ms,
        "width_ms": width_ms,
        "noise_std": float(noise),
    }


def analyze_condition(
    condition_dir: Path,
    snr_min: float,
    transient_ms: float,
    sustained_ms: float,
    a1_min: float,
    peak_floor: float,
    phase_jitter_deg: float,
    on_window_ms: Tuple[float, float],
    off_window_ms: Tuple[float, float],
    c0_dir: Optional[Path] = None,
) -> Path:
    condition_dir = Path(condition_dir)
    stage4 = condition_dir / "Stage4_Visualization"
    stage5_cell = condition_dir / "Stage5_Cell_Fourier_Analysis"

    X = np.load(stage4 / "X_processed.npy")

    if c0_dir is not None:
        c0_Xp = Path(c0_dir) / "Stage4_Visualization" / "X_processed.npy"
        if c0_Xp.exists():
            X_c0 = np.load(c0_Xp)
            if X_c0.shape == X.shape:
                X = X - X_c0

    df = pd.read_csv(stage5_cell / "cell_fourier_results.csv")

    df = df[df["label"].isin(["ON", "OFF"])]
    df = df[df["snr"] >= float(snr_min)]

    out_dir = condition_dir / "Stage6_Cell_SustainedTransient"
    if out_dir.exists():
        for f in out_dir.iterdir():
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)
    out_dir.mkdir(exist_ok=True)

    period_bins = 50
    bin_ms = 10.0
    stim_freq = 2.0
    n_analysis = 5

    rows: List[Dict] = []

    for _, r in df.iterrows():
        i = int(r["cell_idx"])
        pol = str(r["label"])
        cid = str(r.get("cluster_id", ""))

        psth = X[i]
        cycles = psth.reshape(6, period_bins)
        analysis = cycles[1:6]

        b0, b1, wrap = _window_bins(bin_ms, pol, on_window_ms=on_window_ms, off_window_ms=off_window_ms)

        per_width = []
        per_auc = []
        per_peak_above = []
        per_noise = []
        per_peak = []
        per_phi = []
        per_a1 = []
        locked = True

        for c in range(n_analysis):
            cyc = _baseline_subtract(analysis[c])
            w = _window_slice(cyc, b0, b1, wrap)
            sign = _dominant_deflection_sign(w)
            cyc_signed = (sign * cyc).astype(np.float32)

            m = _cycle_effective_width_metrics(
                cyc_signed,
                b0=b0,
                b1=b1,
                wrap=wrap,
                bin_ms=bin_ms,
                peak_floor=peak_floor,
            )
            peak = float(m["peak"])
            a1, phi = _cycle_a1_phi(cyc_signed, period_bins)

            baseline = float(m.get("baseline", 0.0))
            per_width.append(float(m["width_ms"]))
            per_auc.append(float(m["auc_ms"]))
            per_peak_above.append(float(max(0.0, peak - baseline)))
            per_noise.append(float(m.get("noise_std", np.nan)))
            per_peak.append(peak)
            per_a1.append(a1)
            per_phi.append(phi)

            if (a1 < a1_min) or (peak < peak_floor):
                locked = False

        jitter = float(np.degrees(circstd(np.radians(per_phi)))) if locked else float("nan")
        if locked and (jitter > phase_jitter_deg):
            locked = False

        if not locked:
            continue

        width_mean = float(np.mean(per_width))
        if width_mean < transient_ms:
            subtype = "Transient"
        elif width_mean >= sustained_ms:
            subtype = "Sustained"
        else:
            subtype = "Transient"

        row_out: Dict = {
            "cell_idx": i,
            "polarity": pol,
            "subtype": subtype,
            "duration_mean_ms": width_mean,
            "auc_mean_ms": float(np.mean(per_auc)),
            "peak_above_mean": float(np.mean(per_peak_above)),
            "noise_std_mean": float(np.nanmean(per_noise)),
            "phase_jitter_deg": jitter,
            "cluster_id": cid,
            "snr": float(r["snr"]),
        }
        for c in range(n_analysis):
            cn = c + 1
            row_out[f"duration_cycle{cn}_ms"]   = float(per_width[c])
            row_out[f"auc_cycle{cn}_ms"]         = float(per_auc[c])
            row_out[f"peak_above_cycle{cn}"]     = float(per_peak_above[c])
        rows.append(row_out)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "cell_level_sustained_transient.csv", index=False)

    if len(out) > 0:
        cells_dir = out_dir / "cells_by_subtype"
        cells_dir.mkdir(exist_ok=True)
        for pol, st in [("ON", "Transient"), ("ON", "Sustained"), ("OFF", "Transient"), ("OFF", "Sustained")]:
            sub = out[(out["polarity"] == pol) & (out["subtype"] == st)].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values(["subtype", "duration_mean_ms", "cell_idx"], ascending=[True, True, True])
            sub.to_csv(cells_dir / f"cells_{pol}_{st}.csv", index=False)

        _plot_mean_traces_with_sine(out, X, out_dir, bin_ms, on_window_ms, off_window_ms)

        _plot_psth_all_cycles(out, X, out_dir, bin_ms, on_window_ms, off_window_ms)

        _plot_individual_profiles(out, X, out_dir, bin_ms, on_window_ms, off_window_ms)

        mapping = out.groupby(["cluster_id", "polarity", "subtype"]).size().reset_index(name="n_cells")
        mapping.to_csv(out_dir / "cluster_to_subtype_counts.csv", index=False)

        _plot_pie_subtypes(out, out_dir)
        _plot_heatmaps_subtypes(out, X, out_dir, bin_ms)
        _plot_polar_duration(out, out_dir, on_window_ms, off_window_ms)

    return out_dir


def _plot_polar_duration(
    out: pd.DataFrame, out_dir: Path,
    on_window_ms: Tuple[float, float], off_window_ms: Tuple[float, float],
) -> None:
    """Plot mean subtype duration on a polar phase axis."""
    if len(out) == 0:
        return
    colors = {
        ("ON", "Transient"): "#E69F00",
        ("ON", "Sustained"): "#D55E00",
        ("OFF", "Transient"): "#56B4E9",
        ("OFF", "Sustained"): "#0072B2",
    }

    def ms_to_theta(ms: float) -> float:
        return 2.0 * np.pi * (ms / 500.0)

    def window_ms_length(s: float, e: float, wrap: bool) -> float:
        if not wrap:
            return max(1.0, e - s)
        return max(1.0, (500.0 - s) + e)

    on_s, on_e = on_window_ms[0], on_window_ms[1]
    off_s, off_e = off_window_ms[0], off_window_ms[1]
    on_wrap = on_s > on_e
    off_wrap = off_s > off_e

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 1.75)

    def shade_window(s: float, e: float, wrap: bool, color: str, label: str) -> None:
        def _sector(theta0: float, theta1: float) -> None:
            width = float(theta1 - theta0)
            center = float(theta0 + width / 2.0)
            ax.bar(
                [center],
                height=0.22,
                width=width,
                bottom=0.92,
                color=color,
                alpha=0.20,
                edgecolor="none",
                align="center",
                zorder=0,
            )

        if not wrap:
            _sector(ms_to_theta(s), ms_to_theta(e))
        else:
            _sector(ms_to_theta(s), ms_to_theta(500.0))
            _sector(ms_to_theta(0.0), ms_to_theta(e))

        mid_ms = (s + (e - s) / 2.0) if not wrap else ((s + window_ms_length(s, e, True) / 2.0) % 500.0)
        ax.text(
            ms_to_theta(mid_ms),
            1.65,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9),
        )

    shade_window(on_s, on_e, on_wrap, "#D55E00", "ON window")
    shade_window(off_s, off_e, off_wrap, "#0072B2", "OFF window")

    for pol in ["ON", "OFF"]:
        s, e, wrap = (on_s, on_e, on_wrap) if pol == "ON" else (off_s, off_e, off_wrap)
        wlen = window_ms_length(s, e, wrap)
        for st in ["Transient", "Sustained"]:
            d = out[(out["polarity"] == pol) & (out["subtype"] == st)]
            if len(d) == 0:
                continue
            mean_dur = float(d["duration_mean_ms"].mean())
            frac = float(np.clip(mean_dur / wlen, 0.0, 1.0))
            arc_ms = frac * wlen
            if not wrap:
                th = np.linspace(ms_to_theta(s), ms_to_theta(s + arc_ms), 200)
            else:
                end_ms = s + arc_ms
                if end_ms <= 500.0:
                    th = np.linspace(ms_to_theta(s), ms_to_theta(end_ms), 200)
                else:
                    th1 = np.linspace(ms_to_theta(s), ms_to_theta(500.0), 200)
                    th2 = np.linspace(ms_to_theta(0.0), ms_to_theta(end_ms - 500.0), 200)
                    th = np.concatenate([th1, th2])
            r = 1.0 if st == "Sustained" else 0.84
            ax.plot(th, np.ones_like(th) * r, color=colors[(pol, st)], linewidth=10, solid_capstyle="round",
                    label=f"{pol}-{st} mean={mean_dur:.0f}ms (n={len(d)})")

    ax.set_yticklabels([])
    ax.set_title("Mean response duration mapped onto ON/OFF phase windows", pad=25, fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05), fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "04_polar_mean_duration.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_individual_profiles(
    out: pd.DataFrame,
    X: np.ndarray,
    out_dir: Path,
    bin_ms: float,
    on_window_ms: Tuple[float, float],
    off_window_ms: Tuple[float, float],
) -> None:
    """Save individual 3000 ms PSTHs by subtype."""
    if len(out) == 0:
        return

    colors = {
        ("ON", "Transient"): "#E69F00",
        ("ON", "Sustained"): "#D55E00",
        ("OFF", "Transient"): "#56B4E9",
        ("OFF", "Sustained"): "#0072B2",
    }
    on_s, on_e = on_window_ms[0], on_window_ms[1]
    off_s, off_e = off_window_ms[0], off_window_ms[1]

    n_bins = X.shape[1]
    t_ms = np.arange(n_bins) * bin_ms
    lum = 0.8 * np.sin(2 * np.pi * 2.0 * (t_ms / 1000.0))

    prof_dir = out_dir / "individual_profiles"
    if prof_dir.exists():
        shutil.rmtree(prof_dir)
    prof_dir.mkdir(exist_ok=True)

    groups = [("ON", "Transient"), ("ON", "Sustained"), ("OFF", "Transient"), ("OFF", "Sustained")]
    for pol, st in groups:
        sub = out[(out["polarity"] == pol) & (out["subtype"] == st)].copy()
        if len(sub) == 0:
            continue
        gdir = prof_dir / f"{pol}_{st}"
        gdir.mkdir(exist_ok=True)

        sub = sub.sort_values(["duration_mean_ms", "cell_idx"], ascending=[True, True])

        for _, r in sub.iterrows():
            i = int(r["cell_idx"])
            psth = X[i]

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(14, 5.6),
                sharex=True,
                height_ratios=[0.22, 0.78],
                gridspec_kw={"hspace": 0.12},
            )
            ax_sine = axes[0]
            ax_psth = axes[1]

            ax_sine.plot(t_ms, lum, color="gray", ls="--", lw=1.2, alpha=0.8)
            ax_sine.set_ylabel("Stim", fontsize=10)
            ax_sine.set_yticks([-0.8, 0, 0.8])
            ax_sine.set_ylim(-1.1, 1.1)
            ax_sine.tick_params(axis="x", labelbottom=False)
            for c in range(6):
                base = c * 500
                ax_sine.axvspan(base + off_s, base + off_e, color="#0072B2", alpha=0.12)
                if on_s > on_e:
                    ax_sine.axvspan(base, base + on_e, color="#D55E00", alpha=0.12)
                    ax_sine.axvspan(base + on_s, base + 500, color="#D55E00", alpha=0.12)
                else:
                    ax_sine.axvspan(base + on_s, base + on_e, color="#D55E00", alpha=0.12)

            ax_psth.plot(t_ms, psth, color=colors[(pol, st)], lw=2.0)
            for c in range(6):
                base = c * 500
                ax_psth.axvspan(base + off_s, base + off_e, color="#0072B2", alpha=0.08)
                if on_s > on_e:
                    ax_psth.axvspan(base, base + on_e, color="#D55E00", alpha=0.08)
                    ax_psth.axvspan(base + on_s, base + 500, color="#D55E00", alpha=0.08)
                else:
                    ax_psth.axvspan(base + on_s, base + on_e, color="#D55E00", alpha=0.08)
            ax_psth.set_ylabel("Z-score", fontsize=10)
            ax_psth.set_xlabel("Time (ms)", fontsize=10)
            ax_psth.grid(True, alpha=0.15)

            cid = str(r.get("cluster_id", ""))
            title = (
                f"cell_idx={i}  {pol}-{st}  "
                f"dur_mean={float(r['duration_mean_ms']):.1f}ms  "
                f"dur(c3,c4,c5)=({float(r['duration_cycle3_ms']):.0f},{float(r['duration_cycle4_ms']):.0f},{float(r['duration_cycle5_ms']):.0f})ms  "
                f"jitter={float(r['phase_jitter_deg']):.1f}°  "
                f"snr={float(r['snr']):.2f}"
            )
            if cid and cid != "nan":
                title += f"  cluster_id={cid}"
            ax_psth.set_title(title, fontsize=10, pad=8)
            ax_psth.axvspan(1000, 2500, color="black", alpha=0.03, zorder=0)

            fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.1)
            plt.savefig(gdir / f"cell_{i:04d}.png", dpi=200, bbox_inches="tight")
            plt.close()


def _plot_mean_traces_with_sine(
    out: pd.DataFrame, X: np.ndarray, out_dir: Path,
    bin_ms: float, on_window_ms: Tuple[float, float], off_window_ms: Tuple[float, float],
) -> None:
    """Plot subtype mean traces across 3000 ms."""
    substypes = [("ON", "Transient"), ("ON", "Sustained"), ("OFF", "Transient"), ("OFF", "Sustained")]
    colors = {
        ("ON", "Transient"): "#E69F00",
        ("ON", "Sustained"): "#D55E00",
        ("OFF", "Transient"): "#56B4E9",
        ("OFF", "Sustained"): "#0072B2",
    }
    t_ms = np.arange(X.shape[1]) * bin_ms
    lum = 0.8 * np.sin(2 * np.pi * 2.0 * (t_ms / 1000.0))
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, height_ratios=[0.25, 0.75], gridspec_kw={"hspace": 0.08})
    ax_sine = axes[0]
    ax_psth = axes[1]
    ax_sine.plot(t_ms, lum, color="gray", ls="--", lw=1.2, alpha=0.8)
    ax_sine.set_ylabel("Stim")
    ax_sine.set_yticks([-0.8, 0, 0.8])
    ax_sine.set_ylim(-1.1, 1.1)
    ax_sine.spines["top"].set_visible(False)
    on_s, on_e = on_window_ms[0], on_window_ms[1]
    off_s, off_e = off_window_ms[0], off_window_ms[1]
    for c in range(6):
        base = c * 500
        ax_sine.axvspan(base + off_s, base + off_e, color="#0072B2", alpha=0.12)
        if on_s > on_e:
            ax_sine.axvspan(base, base + on_e, color="#D55E00", alpha=0.12)
            ax_sine.axvspan(base + on_s, base + 500, color="#D55E00", alpha=0.12)
        else:
            ax_sine.axvspan(base + on_s, base + on_e, color="#D55E00", alpha=0.12)
    for (pol, st) in substypes:
        sub = out[(out["polarity"] == pol) & (out["subtype"] == st)]
        if len(sub) == 0:
            continue
        idx = sub["cell_idx"].astype(int).values
        m = X[idx].mean(axis=0)
        ax_psth.plot(t_ms, m, color=colors[(pol, st)], lw=2.0, label=f"{pol}-{st} (n={len(idx)})")
    for c in range(6):
        base = c * 500
        ax_psth.axvspan(base + off_s, base + off_e, color="#0072B2", alpha=0.08)
        if on_s > on_e:
            ax_psth.axvspan(base, base + on_e, color="#D55E00", alpha=0.08)
            ax_psth.axvspan(base + on_s, base + 500, color="#D55E00", alpha=0.08)
        else:
            ax_psth.axvspan(base + on_s, base + on_e, color="#D55E00", alpha=0.08)
    ax_psth.set_ylabel("Z-score")
    ax_psth.set_xlabel("Time (ms)")
    ax_psth.set_title("Mean traces (3000ms), ON window=orange, OFF window=blue")
    ax_psth.grid(True, alpha=0.15)
    ax_psth.legend(fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_dir / "02_mean_traces_3000ms.png", dpi=200)
    plt.close()


def _plot_psth_all_cycles(
    out: pd.DataFrame, X: np.ndarray, out_dir: Path,
    bin_ms: float, on_window_ms: Tuple[float, float], off_window_ms: Tuple[float, float],
) -> None:
    """Plot per-subtype PSTHs across 3000 ms."""
    substypes = [("ON", "Transient"), ("ON", "Sustained"), ("OFF", "Transient"), ("OFF", "Sustained")]
    colors = {
        ("ON", "Transient"): "#E69F00",
        ("ON", "Sustained"): "#D55E00",
        ("OFF", "Transient"): "#56B4E9",
        ("OFF", "Sustained"): "#0072B2",
    }
    n_bins = X.shape[1]
    t_ms = np.arange(n_bins) * bin_ms
    lum = 0.8 * np.sin(2 * np.pi * 2.0 * (t_ms / 1000.0))
    on_s, on_e = on_window_ms[0], on_window_ms[1]
    off_s, off_e = off_window_ms[0], off_window_ms[1]
    for pol, st in substypes:
        sub = out[(out["polarity"] == pol) & (out["subtype"] == st)]
        if len(sub) == 0:
            continue
        idx = sub["cell_idx"].astype(int).values
        mat = X[idx]
        m = mat.mean(axis=0)
        sem = mat.std(axis=0) / np.sqrt(mat.shape[0])
        fig, axes = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True, height_ratios=[0.22, 0.78], gridspec_kw={"hspace": 0.12})
        ax_sine = axes[0]
        ax_psth = axes[1]
        ax_sine.plot(t_ms, lum, color="gray", ls="--", lw=1.2, alpha=0.8)
        ax_sine.set_ylabel("Stim", fontsize=10)
        ax_sine.set_yticks([-0.8, 0, 0.8])
        ax_sine.set_ylim(-1.1, 1.1)
        ax_sine.tick_params(axis="x", labelbottom=False)
        for c in range(6):
            base = c * 500
            ax_sine.axvspan(base + off_s, base + off_e, color="#0072B2", alpha=0.12)
            if on_s > on_e:
                ax_sine.axvspan(base, base + on_e, color="#D55E00", alpha=0.12)
                ax_sine.axvspan(base + on_s, base + 500, color="#D55E00", alpha=0.12)
            else:
                ax_sine.axvspan(base + on_s, base + on_e, color="#D55E00", alpha=0.12)
        ax_psth.plot(t_ms, m, color=colors[(pol, st)], lw=2.5, label=f"mean (n={len(sub)})")
        ax_psth.fill_between(t_ms, m - sem, m + sem, color=colors[(pol, st)], alpha=0.25)
        for c in range(6):
            base = c * 500
            ax_psth.axvspan(base + off_s, base + off_e, color="#0072B2", alpha=0.08)
            if on_s > on_e:
                ax_psth.axvspan(base, base + on_e, color="#D55E00", alpha=0.08)
                ax_psth.axvspan(base + on_s, base + 500, color="#D55E00", alpha=0.08)
            else:
                ax_psth.axvspan(base + on_s, base + on_e, color="#D55E00", alpha=0.08)
        ax_psth.set_ylabel("Z-score", fontsize=10)
        ax_psth.set_xlabel("Time (ms)", fontsize=10)
        ax_psth.set_title(f"{pol}-{st} (n={len(sub)})", fontsize=11, pad=8)
        ax_psth.grid(True, alpha=0.15)
        ax_psth.legend(loc="upper right", fontsize=9, framealpha=0.9)
        fig.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.1)
        plt.savefig(out_dir / f"psth_all_cycles_{pol}_{st}.png", dpi=200, bbox_inches="tight")
        plt.close()


def _plot_pie_subtypes(out: pd.DataFrame, out_dir: Path) -> None:
    """Plot subtype proportions."""
    if len(out) == 0:
        return
    order = ["ON-Transient", "ON-Sustained", "OFF-Transient", "OFF-Sustained"]
    lab = out["polarity"] + "-" + out["subtype"]
    counts = lab.value_counts().reindex(order).fillna(0).astype(int)
    counts = counts[counts > 0]
    if counts.empty:
        return
    colors = {
        "ON-Transient": "#E69F00",
        "ON-Sustained": "#D55E00",
        "OFF-Transient": "#56B4E9",
        "OFF-Sustained": "#0072B2",
    }
    cols = [colors.get(k, "#9E9E9E") for k in counts.index]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        startangle=90,
        colors=cols,
        labels=None,
        autopct="%1.1f%%",
        pctdistance=0.85,
        wedgeprops={"linewidth": 1.2, "edgecolor": "white"},
    )
    ax.set_aspect("equal")
    ax.set_title("Sustained / Transient (ON/OFF)")
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(10)
        t.set_fontweight("bold")
    legend_labels = [f"{k}  (n={int(v)})" for k, v in counts.items()]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, frameon=False)
    fig.subplots_adjust(right=0.70)
    plt.savefig(out_dir / "03_pie_subtypes.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_heatmaps_subtypes(out: pd.DataFrame, X: np.ndarray, out_dir: Path, bin_ms: float) -> None:
    """Plot subtype heatmaps (units x time)."""
    if len(out) == 0:
        return
    heat_dir = out_dir / "heatmaps_subtypes"
    if heat_dir.exists():
        shutil.rmtree(heat_dir)
    heat_dir.mkdir(exist_ok=True)
    groups = [("ON", "Transient"), ("ON", "Sustained"), ("OFF", "Transient"), ("OFF", "Sustained")]
    n_bins = X.shape[1]
    time_s = np.arange(n_bins) * bin_ms / 1000.0
    for pol, st in groups:
        sub = out[(out["polarity"] == pol) & (out["subtype"] == st)]
        if len(sub) == 0:
            continue
        idx = sub["cell_idx"].astype(int).values
        mat = X[idx]
        vmin = float(np.percentile(mat, 2.0))
        vmax = float(np.percentile(mat, 98.0))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = float(np.min(mat)), float(np.max(mat))
        fig, ax = plt.subplots(figsize=(12, 6.2))
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            extent=[float(time_s[0]), float(time_s[-1]), 0, mat.shape[0]],
            cmap="cividis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"Heatmap: {pol}-{st} (n={len(sub)})", pad=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Units")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Z-score")
        plt.tight_layout()
        plt.savefig(heat_dir / f"heatmap_{pol}_{st}.png", dpi=250, bbox_inches="tight")
        plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("condition_dir", type=str)
    p.add_argument("--snr_min",       type=float, default=0.3)
    p.add_argument("--transient_ms",  type=float, default=95.0, help="Transient if mean effective width < this (ms)")
    p.add_argument("--sustained_ms",  type=float, default=95.0, help="Sustained if mean effective width >= this (ms)")
    p.add_argument("--a1_min",        type=float, default=0.01)
    p.add_argument("--peak_floor",    type=float, default=0.5, help="Min peak (z-score units) inside expected window")
    p.add_argument("--phase_jitter_deg", type=float, default=35.0, help="Max phase jitter across all 6 cycles (deg)")
    p.add_argument("--on_window_ms",  type=str, default="375,125")
    p.add_argument("--off_window_ms", type=str, default="125,375")
    p.add_argument("--c0_dir", type=str, default=None,
                   help="Path to c0_f2Hz condition dir; X_processed is subtracted as spontaneous baseline")
    args = p.parse_args()

    def _parse(s: str) -> Tuple[float, float]:
        a, b = s.split(",")
        return float(a), float(b)

    out = analyze_condition(
        Path(args.condition_dir),
        snr_min=float(args.snr_min),
        transient_ms=float(args.transient_ms),
        sustained_ms=float(args.sustained_ms),
        a1_min=float(args.a1_min),
        peak_floor=float(args.peak_floor),
        phase_jitter_deg=float(args.phase_jitter_deg),
        on_window_ms=_parse(args.on_window_ms),
        off_window_ms=_parse(args.off_window_ms),
        c0_dir=Path(args.c0_dir) if args.c0_dir else None,
    )


if __name__ == "__main__":
    main()
