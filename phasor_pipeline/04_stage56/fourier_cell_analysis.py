#!/usr/bin/env python3
"""
Cell-level Fourier classification of individual cells.

Input:
- Stage4_Visualization/X_processed.npy

Output:
- Stage5_Cell_Fourier_Analysis/ (CSV + plots)

Method:
- Analyze cycles 2-6 by default.
- Split Biphasic vs ON/OFF using A2/A1.
- Enforce SNR, per-cycle amplitude, and phase-jitter checks.
- ON/OFF label is a per-cycle majority vote (tie-break by mean X).
- Optional c0 subtraction via --c0_dir.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fft import fft, ifft
from scipy.stats import circstd
from condition_layout import resolve_cluster_dir, sibling_c0

EPS = 1e-9
N_CYCLES = 6

CB_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
SUBTYPE_COLORS = {"ON": "#E69F00", "OFF": "#56B4E9", "Biphasic": "#009E73"}


def circular_jitter_deg(phases_deg: np.ndarray) -> float:
    return float(np.degrees(circstd(np.radians(phases_deg))))


def harmonic_recon_snr(r_tau: np.ndarray, harmonics: Tuple[int, ...] = (1, 2)) -> Tuple[float, float, float]:
    """Compute reconstruction SNR from selected Fourier harmonics."""
    F = fft(r_tau)
    Ff = np.zeros_like(F)
    for k in harmonics:
        Ff[k] = F[k]
        Ff[-k] = F[-k]
    recon = np.real(ifft(Ff))
    resid = r_tau - recon
    sig_std = float(np.std(recon))
    noise_std = float(np.std(resid))
    snr = sig_std / (noise_std + EPS)
    return snr, sig_std, noise_std


def cycle_fft_features(cycle: np.ndarray, period_bins: int) -> Dict[str, float]:
    F = fft(cycle)
    c1 = F[1] / period_bins
    c2 = F[2] / period_bins
    a1, a2 = float(np.abs(c1)), float(np.abs(c2))
    phi1 = float(np.angle(c1, deg=True)) % 360.0
    phi2 = float(np.angle(c2, deg=True)) % 360.0
    return {
        "A1": a1,
        "A2": a2,
        "phi1": phi1,
        "phi2": phi2,
        "X": float(a1 * np.cos(np.radians(phi1))),
        "Y": float(a1 * np.sin(np.radians(phi1))),
    }


def first_harmonic_peak_bin(c1: complex, period_bins: int) -> int:
    """Return the peak bin of the first-harmonic reconstruction."""
    n = np.arange(period_bins)
    recon = 2.0 * np.real(c1 * np.exp(1j * 2.0 * np.pi * n / period_bins))
    return int(np.argmax(recon))


def peak_vote_from_bin(peak_bin: int, period_bins: int) -> str:
    """Return ON/OFF from the peak position in one cycle."""
    off_start = int(np.floor(0.25 * period_bins))
    off_end   = int(np.ceil(0.75 * period_bins))
    if off_start <= peak_bin < off_end:
        return "OFF"
    return "ON"


def _cycle_defaults(n: int) -> Dict:
    """Build default per-cycle output fields."""
    d: Dict = {}
    for c in range(1, n + 1):
        d[f"A1_cycle{c}"]    = 0.0
        d[f"X_cycle{c}"]     = 0.0
        d[f"phi_cycle{c}"]   = 0.0
        d[f"A2_cycle{c}"]    = 0.0
        d[f"phi2_cycle{c}"]  = 0.0
        d[f"peakbin_cycle{c}"] = -1
        d[f"vote_cycle{c}"]  = ""
    return d


class CellFourierAnalyzer:
    def __init__(
        self,
        condition_dir: Path,
        stim_freq: float = 2.0,
        sampling_rate: float = 100.0,
        start_cycle: int = 1,
        end_cycle: int = 6,
        onoff_threshold: float = 0.6,
        snr_threshold: float = 0.3,
        a1_min: float = 0.01,
        a2_min: float = 0.01,
        x_min: float = 0.01,
        phase_jitter_threshold_deg: float = 35.0,
        phase_jitter_biphasic_deg: float = 35.0,
        output_subdir: Optional[str] = None,
        c0_dir: Optional[Path] = None,
    ):
        self.requested_dir = Path(condition_dir)
        self.cluster_dir = resolve_cluster_dir(self.requested_dir)
        self.condition_dir = self.cluster_dir
        self.stage4_dir = self.cluster_dir / "Stage4_Visualization"
        out_name = output_subdir if output_subdir else "Stage5_Cell_Fourier_Analysis"
        self.out_dir = self.requested_dir / out_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if c0_dir is None:
            c0_dir = sibling_c0(self.cluster_dir)
        print(f"Clustering: {self.cluster_dir}")
        print(f"Fourier output: {self.out_dir}")
        if c0_dir is not None:
            print(f"Baseline c0: {c0_dir}")

        self.stim_freq  = float(stim_freq)
        self.fs         = float(sampling_rate)
        self.bin_ms     = 1000.0 / self.fs
        self.period_bins = int(round(self.fs / self.stim_freq))
        self.start_cycle = int(start_cycle)
        self.end_cycle   = int(end_cycle)
        self.onoff_threshold = float(onoff_threshold)
        self.snr_threshold   = float(snr_threshold)
        self.a1_min = float(a1_min)
        self.a2_min = float(a2_min)
        self.x_min  = float(x_min)
        self.phase_jitter_threshold_deg  = float(phase_jitter_threshold_deg)
        self.phase_jitter_biphasic_deg   = float(phase_jitter_biphasic_deg)
        self.c0_dir = Path(c0_dir) if c0_dir else None

        self._load_stage4()

    def _load_stage4(self) -> None:
        Xp = self.stage4_dir / "X_processed.npy"
        if not Xp.exists():
            raise FileNotFoundError(f"Missing {Xp}")
        self.X = np.load(Xp)
        self.n_cells, self.n_bins = self.X.shape

        if self.c0_dir is not None:
            c0_Xp = Path(self.c0_dir) / "Stage4_Visualization" / "X_processed.npy"
            if c0_Xp.exists():
                X_c0 = np.load(c0_Xp)
                if X_c0.shape == self.X.shape:
                    self.X = self.X - X_c0

        Lp = self.stage4_dir / "cluster_labels.npy"
        self.cluster_labels = np.load(Lp) if Lp.exists() else np.full(self.n_cells, -1, dtype=int)

    def _extract_cycles(self, psth: np.ndarray) -> np.ndarray:
        n_cycles = psth.size // self.period_bins
        return psth[:n_cycles * self.period_bins].reshape(n_cycles, self.period_bins)

    def _cid(self, i: int) -> str:
        return f"C{self.cluster_labels[i]}" if self.cluster_labels[i] != -1 else "Noise"

    def _unclassified_row(self, cell_idx: int, cid_str: str, reason: str, **kwargs) -> Dict:
        defaults: Dict = {
            "snr": float("nan"), "snr_signal_std": float("nan"), "snr_noise_std": float("nan"),
            "phase_jitter_deg": float("nan"),
            "A1": 0.0, "A2": 0.0, "A2_A1": 0.0, "phi1": 0.0, "X": 0.0, "Y": 0.0,
        }
        defaults.update(_cycle_defaults(N_CYCLES))
        defaults.update(kwargs)
        return {"cell_idx": int(cell_idx), "cluster_id": cid_str, "label": f"Unclassified ({reason})", **defaults}

    def analyze(self) -> pd.DataFrame:
        rows = []
        n_analysis = self.end_cycle - self.start_cycle

        for i in range(self.n_cells):
            cycles = self._extract_cycles(self.X[i])
            if cycles.shape[0] < self.end_cycle:
                rows.append(self._unclassified_row(i, self._cid(i), "TooFewCycles"))
                continue

            analysis = cycles[self.start_cycle:self.end_cycle]
            r_tau = analysis.mean(axis=0)
            rt = cycle_fft_features(r_tau, self.period_bins)
            ratio = float(rt["A2"] / (rt["A1"] + EPS))

            per_cycle = [cycle_fft_features(analysis[c], self.period_bins) for c in range(n_analysis)]
            A1s  = np.array([p["A1"]  for p in per_cycle])
            phis = np.array([p["phi1"] for p in per_cycle])
            A2s  = np.array([p["A2"]  for p in per_cycle])
            phis2 = np.array([p["phi2"] for p in per_cycle])
            Xs   = np.array([p["X"]   for p in per_cycle])

            is_biphasic_candidate = bool(ratio >= self.onoff_threshold)

            if is_biphasic_candidate:
                snr2, sig2, noise2 = harmonic_recon_snr(r_tau, harmonics=(2,))
                if snr2 < self.snr_threshold:
                    rows.append(self._unclassified_row(
                        i, self._cid(i), "LowSNR_4Hz",
                        snr=snr2, snr_signal_std=sig2, snr_noise_std=noise2,
                        A1=float(rt["A1"]), A2=float(rt["A2"]), A2_A1=ratio,
                    ))
                    continue
                if np.any(A2s < self.a2_min):
                    rows.append(self._unclassified_row(
                        i, self._cid(i), "A2TooSmall",
                        snr=snr2, snr_signal_std=sig2, snr_noise_std=noise2,
                        A1=float(rt["A1"]), A2=float(rt["A2"]), A2_A1=ratio,
                        **{f"A2_cycle{c+1}": float(A2s[c]) for c in range(n_analysis)},
                        **{f"phi2_cycle{c+1}": float(phis2[c]) for c in range(n_analysis)},
                    ))
                    continue
                jitter2 = circular_jitter_deg(phis2)
                if jitter2 > self.phase_jitter_biphasic_deg:
                    rows.append(self._unclassified_row(
                        i, self._cid(i), "HighPhaseJitter_4Hz",
                        snr=snr2, snr_signal_std=sig2, snr_noise_std=noise2,
                        phase_jitter_deg=jitter2,
                        A1=float(rt["A1"]), A2=float(rt["A2"]), A2_A1=ratio,
                        **{f"A2_cycle{c+1}": float(A2s[c]) for c in range(n_analysis)},
                        **{f"phi2_cycle{c+1}": float(phis2[c]) for c in range(n_analysis)},
                    ))
                    continue
                label = "Biphasic"
                jitter = float(jitter2)
                snr, sig_std, noise_std = snr2, sig2, noise2
                peak_bins = [None] * n_analysis
                votes = [""] * n_analysis

            else:
                snr1, sig1, noise1 = harmonic_recon_snr(r_tau, harmonics=(1,))
                if snr1 < self.snr_threshold:
                    rows.append(self._unclassified_row(
                        i, self._cid(i), "LowSNR_2Hz",
                        snr=snr1, snr_signal_std=sig1, snr_noise_std=noise1,
                        A1=float(rt["A1"]), A2=float(rt["A2"]), A2_A1=ratio,
                    ))
                    continue
                if np.any(A1s < self.a1_min):
                    rows.append(self._unclassified_row(
                        i, self._cid(i), "A1TooSmall",
                        snr=snr1, snr_signal_std=sig1, snr_noise_std=noise1,
                        A1=float(rt["A1"]), A2=float(rt["A2"]), A2_A1=ratio,
                        **{f"A1_cycle{c+1}": float(A1s[c]) for c in range(n_analysis)},
                        **{f"phi_cycle{c+1}": float(phis[c]) for c in range(n_analysis)},
                    ))
                    continue
                jitter = circular_jitter_deg(phis)
                if jitter > self.phase_jitter_threshold_deg:
                    rows.append(self._unclassified_row(
                        i, self._cid(i), "HighPhaseJitter",
                        snr=snr1, snr_signal_std=sig1, snr_noise_std=noise1,
                        phase_jitter_deg=jitter,
                        A1=float(rt["A1"]), A2=float(rt["A2"]), A2_A1=ratio,
                        **{f"A1_cycle{c+1}": float(A1s[c]) for c in range(n_analysis)},
                        **{f"phi_cycle{c+1}": float(phis[c]) for c in range(n_analysis)},
                    ))
                    continue

                votes = []
                peak_bins = []
                for c in range(n_analysis):
                    F = fft(analysis[c])
                    c1 = F[1] / self.period_bins
                    pb = first_harmonic_peak_bin(c1, self.period_bins)
                    peak_bins.append(pb)
                    votes.append(peak_vote_from_bin(pb, self.period_bins))

                on_votes  = sum(v == "ON"  for v in votes)
                off_votes = n_analysis - on_votes
                if on_votes != off_votes:
                    label = "ON" if on_votes > off_votes else "OFF"
                else:
                    label = "ON" if float(rt["X"]) >= 0.0 else "OFF"
                snr, sig_std, noise_std = snr1, sig1, noise1

            row: Dict = {
                "cell_idx": i,
                "cluster_id": self._cid(i),
                "label": label,
                "snr": snr,
                "snr_signal_std": sig_std,
                "snr_noise_std": noise_std,
                "A1": float(rt["A1"]),
                "A2": float(rt["A2"]),
                "A2_A1": ratio,
                "phi1": float(rt["phi1"]),
                "X": float(rt["X"]),
                "Y": float(rt["Y"]),
                "phase_jitter_deg": jitter,
            }
            for c in range(n_analysis):
                cn = c + 1
                row[f"A1_cycle{cn}"]    = float(A1s[c])
                row[f"X_cycle{cn}"]     = float(Xs[c])
                row[f"phi_cycle{cn}"]   = float(phis[c])
                row[f"A2_cycle{cn}"]    = float(A2s[c])
                row[f"phi2_cycle{cn}"]  = float(phis2[c])
                row[f"peakbin_cycle{cn}"] = int(peak_bins[c]) if peak_bins[c] is not None else -1
                row[f"vote_cycle{cn}"]  = str(votes[c])
            rows.append(row)

        self.df = pd.DataFrame(rows)
        self.df.to_csv(self.out_dir / "cell_fourier_results.csv", index=False)
        return self.df

    def plot_population(self) -> None:
        df, colors = self.df, SUBTYPE_COLORS
        time_ms = np.arange(self.n_bins) * self.bin_ms

        counts = df["label"].value_counts()
        plt.figure(figsize=(10, 5))
        ax = plt.gca()
        bar_colors = [colors.get(l, "#999999") for l in counts.index]
        counts.plot(kind="bar", ax=ax, color=bar_colors)
        ax.set_title(f"Cell-level labels ({self.condition_dir.name})")
        ax.set_ylabel("# cells")
        ax.set_xlabel("Cell Type")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.out_dir / "01_cell_label_counts.png", dpi=200)
        plt.close()

        fig, ax = plt.subplots(figsize=(14, 6))
        all_traces = []
        for lab in ["ON", "OFF", "Biphasic"]:
            mask = df["label"] == lab
            if not mask.any():
                continue
            idx = df.loc[mask, "cell_idx"].astype(int).values
            for cell_idx in idx:
                ax.plot(time_ms, self.X[cell_idx], color=colors[lab], alpha=0.1, lw=0.5)
            m   = self.X[idx].mean(axis=0)
            sem = self.X[idx].std(axis=0) / np.sqrt(len(idx))
            all_traces.append(m)
            ax.plot(time_ms, m, color=colors[lab], lw=3.0, label=f"{lab} (n={len(idx)})", zorder=10)
            ax.fill_between(time_ms, m - sem, m + sem, color=colors[lab], alpha=0.2, zorder=9)

        if all_traces:
            y_min, y_max = min(np.min(t) for t in all_traces), max(np.max(t) for t in all_traces)
            ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)

        lum = 0.8 * np.sin(2 * np.pi * self.stim_freq * (time_ms / 1000.0))
        ax2 = ax.twinx()
        ax2.plot(time_ms, lum, color="gray", alpha=0.25, ls="--", lw=1.2)
        ax2.set_yticks([])
        ax.set_title(f"Subtype averages with individual cells ({self.condition_dir.name})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_ylabel("Z-score", fontsize=11)
        ax.legend(framealpha=0.9, loc="upper right")
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.savefig(self.out_dir / "02_subtype_averages_with_individuals.png", dpi=200)
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 4))
        for lab in ["ON", "OFF", "Biphasic"]:
            mask = df["label"] == lab
            if mask.any():
                idx = df.loc[mask, "cell_idx"].astype(int).values
                m   = self.X[idx].mean(axis=0)
                sem = self.X[idx].std(axis=0) / np.sqrt(len(idx))
                ax.plot(time_ms, m, color=colors[lab], lw=2.5, label=f"{lab} (n={len(idx)})")
                ax.fill_between(time_ms, m - sem, m + sem, color=colors[lab], alpha=0.2)

        if all_traces:
            y_min, y_max = min(np.min(t) for t in all_traces), max(np.max(t) for t in all_traces)
            ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)

        ax2 = ax.twinx()
        ax2.plot(time_ms, lum, color="gray", alpha=0.25, ls="--", lw=1.2)
        ax2.set_yticks([])
        ax.set_title("Subtype mean traces (3000ms)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=11)
        ax.set_ylabel("Z-score", fontsize=11)
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.savefig(self.out_dir / "03_subtype_averages_only.png", dpi=200)
        plt.close()

        subtype_order = ["ON", "OFF", "Biphasic"]
        n_subtypes = sum(1 for lab in subtype_order if (df["label"] == lab).any())
        if n_subtypes > 0:
            fig, axes = plt.subplots(n_subtypes, 1, figsize=(12, 4 * n_subtypes), sharex=True)
            axes = [axes] if n_subtypes == 1 else list(axes)
            plot_idx = 0
            for lab in subtype_order:
                mask = df["label"] == lab
                if mask.any():
                    ax = axes[plot_idx]
                    idx = df.loc[mask, "cell_idx"].astype(int).values
                    m   = self.X[idx].mean(axis=0)
                    sem = self.X[idx].std(axis=0) / np.sqrt(len(idx))
                    ax.plot(time_ms, m, color=colors[lab], lw=2.5, label=f"{lab} (n={len(idx)})")
                    ax.fill_between(time_ms, m - sem, m + sem, color=colors[lab], alpha=0.2)
                    ax2 = ax.twinx()
                    ax2.plot(time_ms, lum, color="gray", alpha=0.25, ls="--", lw=1.2)
                    ax2.set_yticks([])
                    ax.set_ylabel("Z-score", fontsize=11, fontweight="bold")
                    ax.set_title(f"{lab} Average Response (n={len(idx)} cells)", fontsize=12, fontweight="bold", color=colors[lab])
                    ax.grid(True, alpha=0.15)
                    ax.legend(framealpha=0.9, loc="upper right")
                    plot_idx += 1
            axes[-1].set_xlabel("Time (ms)", fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(self.out_dir / "04_subtype_averages_vertical.png", dpi=200)
            plt.close()

        df.groupby(["cluster_id", "label"]).size().unstack(fill_value=0).to_csv(
            self.out_dir / "05_cluster_to_cell_labels.csv"
        )

    def plot_profiles(self) -> None:
        df, colors = self.df, SUBTYPE_COLORS
        out = self.out_dir / "Individual_Profiles"
        out.mkdir(exist_ok=True)
        time_ms = np.arange(self.n_bins) * self.bin_ms
        lum_full = 0.8 * np.sin(2 * np.pi * self.stim_freq * (time_ms / 1000.0))
        c0_ms = self.start_cycle * self.period_bins * self.bin_ms
        c1_ms = self.end_cycle   * self.period_bins * self.bin_ms

        for lab in ["ON", "OFF", "Biphasic"]:
            d = df[df["label"] == lab]
            if d.empty:
                continue
            (out / lab).mkdir(exist_ok=True)
            for _, row in d.iterrows():
                i = int(row["cell_idx"])
                fig = plt.figure(figsize=(14, 6))
                gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3, figure=fig)
                ax = fig.add_subplot(gs[0])
                ax_metrics = fig.add_subplot(gs[1])
                ax_metrics.axis('off')

                ax.plot(time_ms, self.X[i], color=colors[lab], lw=2.5, alpha=0.9)
                ax.axvspan(c0_ms, c1_ms, color="gold", alpha=0.15)
                ax.set_ylabel("Z-score", fontsize=11, fontweight="bold")
                ax.set_title(
                    f"Cell {i} | {lab} | Cluster {row['cluster_id']} | "
                    f"SNR={row['snr']:.2f} | A2/A1={row['A2_A1']:.2f}",
                    fontsize=13, fontweight="bold",
                )
                ax.grid(True, alpha=0.2, linestyle="--")
                ax.set_xlim(0, time_ms[-1])
                ax.plot([], [], color=colors[lab], lw=2.5, label=f"{lab} response")
                ax.axvspan(0, 0, color="gold", alpha=0.15, label=f"Analysis cycles ({self.start_cycle+1}-{self.end_cycle})")
                ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

                ax2 = ax.twinx()
                ax2.plot(time_ms, lum_full, color="gray", alpha=0.3, ls="--", lw=1.5)
                ax2.set_yticks([])
                ax2.set_ylabel("Luminance (sine)", fontsize=9, color="gray", alpha=0.7)

                txt = (
                    f"A1={row['A1']:.3f}  |  A2={row['A2']:.3f}  |  A2/A1={row['A2_A1']:.2f}  |  "
                    f"Phi1={row['phi1']:.1f}°  |  X={row['X']:.3f}  |  "
                    f"Phase jitter={row['phase_jitter_deg']:.1f}°  |  SNR={row['snr']:.2f}"
                )
                ax_metrics.text(
                    0.5, 0.5, txt, transform=ax_metrics.transAxes, fontsize=10,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8, edgecolor="black"),
                )

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                fig.savefig(out / lab / f"cell_{i:04d}.png", dpi=200)
                plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("condition_dir", type=str)
    p.add_argument("--start_cycle",   type=int,   default=1,
                   help="First cycle to include (0-indexed). Default=1 skips the first (noisy) cycle.")
    p.add_argument("--end_cycle",     type=int,   default=6,
                   help="Cycle index (exclusive). Default=6 gives cycles 2-6 (5 cycles total).")
    p.add_argument("--onoff_threshold",         type=float, default=0.6,
                   help="Biphasic if A2/A1 >= this (default 0.6)")
    p.add_argument("--snr_threshold",           type=float, default=0.3)
    p.add_argument("--phase_jitter_deg",        type=float, default=35.0)
    p.add_argument("--phase_jitter_biphasic_deg", type=float, default=35.0)
    p.add_argument("--a1_min",  type=float, default=0.01)
    p.add_argument("--a2_min",  type=float, default=0.01)
    p.add_argument("--x_min",   type=float, default=0.01)
    p.add_argument("--output_subdir", type=str, default=None)
    p.add_argument("--c0_dir", type=str, default=None,
                   help="Path to the c0_f2Hz condition dir whose X_processed is subtracted as baseline")
    args = p.parse_args()

    try:
        resolved_c0 = Path(args.c0_dir) if args.c0_dir else None
        if resolved_c0 is not None and not (resolved_c0 / "Stage4_Visualization" / "X_processed.npy").exists():
            resolved_c0 = resolve_cluster_dir(resolved_c0)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    try:
        analyzer = CellFourierAnalyzer(
        Path(args.condition_dir),
        start_cycle=args.start_cycle,
        end_cycle=args.end_cycle,
        onoff_threshold=float(args.onoff_threshold),
        snr_threshold=float(args.snr_threshold),
        phase_jitter_threshold_deg=float(args.phase_jitter_deg),
        phase_jitter_biphasic_deg=float(args.phase_jitter_biphasic_deg),
        a1_min=float(args.a1_min),
        a2_min=float(args.a2_min),
        x_min=float(args.x_min),
        output_subdir=args.output_subdir,
        c0_dir=resolved_c0,
    )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    analyzer.analyze()
    analyzer.plot_population()
    analyzer.plot_profiles()


if __name__ == "__main__":
    main()
