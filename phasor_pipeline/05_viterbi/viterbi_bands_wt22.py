import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import gaussian_filter1d

COLORBLIND_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7",
    "#000000", "#999999", "#882255", "#AA4499",
    "#117733", "#44AA99", "#DDCC77", "#88CCEE",
    "#332288", "#661100", "#6699CC", "#AA4455",
    "#228833", "#CCBB44", "#EE6677", "#AA7744",
    "#774411", "#114477", "#771155", "#555555",
    "#99DDFF", "#77AADD",
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
    "#66A61E", "#E6AB02", "#A6761D", "#666666",
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
    "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3",
]

BIN_MS_DEFAULT   = 10
FREQ_HZ_DEFAULT  = 2.0
REP_LEN_BINS     = 3000
SEG_LEN_BINS     = 300
FOURIER_LABELS   = ("ON", "OFF", "Biphasic")
SUBTYPE_LABELS = (
    "ON-Sustained",
    "ON-Transient",
    "OFF-Sustained",
    "OFF-Transient",
)

SEG_STRUCTURE_S = [
    ( 0,  3,  0.0),
    ( 3,  6, 50.0),
    ( 6,  9,  0.0),
    ( 9, 12, 60.0),
    (12, 15,  0.0),
    (15, 18, 70.0),
    (18, 21,  0.0),
    (21, 24, 80.0),
    (24, 27,  0.0),
    (27, 30, 90.0),
]


def segments_from_states(z, force_splits=None):
    z = np.asarray(z).astype(int)
    T = len(z)
    if T == 0:
        return []
    split_points = set(force_splits or [])
    segs = []
    start = 0
    cur = z[0]
    for t in range(1, T):
        if t in split_points or z[t] != cur:
            segs.append((start, t, cur))
            start = t
            cur = z[t]
    segs.append((start, T, cur))
    return segs


def infer_segments_from_contrast(contrast, rep_len_bins, bin_ms):
    """
    Infer (start_s, end_s, contrast_percent) segments from the first rep's
    contrast array. Assumes contrast is aligned to the same binning as modes.
    """
    c = np.asarray(contrast).astype(float)
    if c.shape[0] < rep_len_bins:
        raise ValueError("contrast array shorter than one rep")
    c0 = c[:rep_len_bins]
    segs = []
    start_bin = 0
    cur = float(c0[0])
    for i in range(1, rep_len_bins):
        v = float(c0[i])
        if v != cur:
            ta = start_bin * bin_ms / 1000.0
            tb = i * bin_ms / 1000.0
            segs.append((ta, tb, cur))
            start_bin = i
            cur = v
    segs.append((start_bin * bin_ms / 1000.0, rep_len_bins * bin_ms / 1000.0, cur))
    return segs


def parse_title_from_bundle(bundle_path):
    """
    Return a human-readable title for a bundle path.

    Uses explicit path-pattern → label mapping (sequential wt renaming) first,
    then falls back to regex-based heuristics.  No circular import from gcp.
    """
    import re

    p = Path(bundle_path)
    low = str(p).lower()
    base = p.name.lower()
    parent = p.parent.name

    if "no projection" in low:
        return "No Projection"
    if "20260520_ap4" in low and "/baseline/run1/" in low:
        return "Wt10 Run 1"
    if "20260520_ap4" in low and "/baseline/run2/" in low:
        return "Wt10 Run 2"
    if "20260520_ap4" in low and "2.5um ap4" in low:
        return "Wt10 AP4 Run 1"
    if "20260520_ap4" in low and "washout for 2.5um" in low:
        return "Wt10 Washout 2.5 Run 1"
    if "20260520_ap4" in low and "20um ap4" in low:
        return "Wt10 AP4 20 µM Run 1"
    if "20260520_ap4" in low and "washout for 20um" in low:
        return "Wt10 Washout 20 Run 1"

    explicit_labels = [
        ("/10_27_2025/wt1_2hz/bundle.npz",       "Wt1 Run 1"),
        ("/10_27_2025/wt1_2hz_run2/bundle.npz",   "Wt1 Run 2"),
        ("wt21_bundle",                            "Wt2 Run 1"),
        ("wt22_bundle",                            "Wt2 Run 2"),
        ("/20260106_wt7_ffsine/2hz_run1/",         "Wt3 Run 1"),
        ("/20260106_wt7_ffsine/2hz_run2/",         "Wt3 Run 2"),
        ("/02_06_2024/wt8 (2hz)/run1/",            "Wt4 Run 1"),
        ("/02_06_2024/wt8 (2hz)/run2/",            "Wt4 Run 2"),
        ("/02_06_2024/wt8 + 20um ap4/run1/",       "Wt4 AP4 Run 1"),
        ("/02_06_2024/wt8 +ap4 washout (2hz)/run1/", "Wt4 Washout Run 1"),
        ("wt111_bundle",                           "Wt5 Run 1"),
        ("wt112_bundle",                           "Wt5 Run 2"),
        ("wt121_bundle",                           "Wt6 Run 1"),
        ("wt122_bundle",                           "Wt6 Run 2"),
        ("wt131_bundle",                           "Wt7 Run 1"),
        ("wt132_bundle",                           "Wt7 Run 2"),
        ("/20260422_ap4(2.5um)/pre_ap4/run1/",     "Wt8 Run 1"),
        ("/20260422_ap4(2.5um)/pre_ap4/run2/",     "Wt8 Run 2"),
        ("/20260422_ap4(2.5um)/treat with 2.5um ap4/", "Wt8 AP4 Run 1"),
        ("/20260422_ap4(2.5um)/wash/run1/",        "Wt8 Washout Run 1"),
        ("/20260422_ap4(2.5um)/wash/run2/",        "Wt8 Washout Run 2"),
        ("/rd1/rd1_pre/",                          "Rd1-1"),
        ("/rd1/rd1(2)_pre/",                       "Rd1-2"),
        ("/rd1/rd1(3)_pre/",                       "Rd1-3"),
        ("/rd1/rd1_post/",                         "Rd1-1 Post"),
        ("/pre_ap4_baseline/run1/",                "Wt9 Run 1"),
        ("/pre_ap4_baseline/run2/",                "Wt9 Run 2"),
        ("/2.5_ap4/run1/",                         "Wt9 AP4 Run 1"),
        ("/washout_2.5/run1/",                     "Wt9 Washout Run 1"),
        ("/20260520_ap4(2.5+20 wt17)/baseline/run1/", "Wt10 Run 1"),
        ("/20260520_ap4(2.5+20 wt17)/baseline/run2/", "Wt10 Run 2"),
        ("/20260520_ap4(2.5+20 wt17)/2.5um ap4/run1/", "Wt10 AP4 Run 1"),
        ("/20260520_ap4(2.5+20 wt17)/washout for 2.5um/run1/", "Wt10 Washout 2.5 Run 1"),
        ("/20260520_ap4(2.5+20 wt17)/20um ap4/run1/", "Wt10 AP4 20 µM Run 1"),
        ("/20260520_ap4(2.5+20 wt17)/washout for 20um/run1/", "Wt10 Washout 20 Run 1"),
        ("/no projection/baseline/",               "No Projection"),
        ("/wt18_files_final_results/wt18_run2/",   "Wt11 Run 2"),
        ("/wt18_files_final_results/wt18_ap4/",    "Wt11 AP4 Run 1"),
        ("/wt18_files_final_results/wt18_wash1/",  "Wt11 Washout Run 1"),
        ("/wt18_files_final_results/wt18_wash2/",  "Wt11 Washout Run 2"),
    ]
    for pattern, label in explicit_labels:
        if pattern in low or pattern in base:
            return label

    suffix = ""
    if "washout" in low or "wo_bundle" in base or "_wo_" in base or "/wash/" in low:
        suffix = " Washout"
    elif "ap4" in low:
        suffix = " AP4"

    if re.match(r"rd\d", parent, re.I):
        label = parent.replace("_", " ").strip()
        if label.lower().startswith("rd"):
            label = "Rd" + label[2:]
        label = re.sub(r"(Rd\d+)\(", r"\1 (", label)
        return f"{label}{suffix}"

    wt_m = re.search(r"wt(\d+)", base)
    wt_digits = wt_m.group(1) if wt_m else None
    wt_m_path = re.search(r"wt(\d+)", low)
    wt_digits_path = wt_m_path.group(1) if wt_m_path else None
    wt_digits_dir = None
    if wt_digits is None and wt_digits_path is None:
        for sib in p.parent.iterdir():
            m = re.search(r"wt(\d+)", sib.name.lower())
            if m:
                wt_digits_dir = m.group(1)
                break

    run_m = re.search(r"run(\d+)", low)
    run_num_from_path = int(run_m.group(1)) if run_m else None

    if wt_digits is None:
        wt_src = wt_digits_path if wt_digits_path is not None else wt_digits_dir
        wt_num = int(wt_src) if wt_src is not None else "?"
        run_num = run_num_from_path if run_num_from_path is not None else "?"
    else:
        if len(wt_digits) >= 2:
            wt_num = int(wt_digits[:-1])
            run_num = int(wt_digits[-1])
        else:
            wt_num = int(wt_digits)
            run_num = 1
        if run_num_from_path is not None:
            run_num = run_num_from_path

    return f"Wt{wt_num}{suffix} Run {run_num}"


def condition_name_from_contrast(contrast_value):
    return f"c{int(round(float(contrast_value)))}"


def bins_to_seconds(n_bins, bin_ms):
    return n_bins * bin_ms / 1000.0


def bins_to_milliseconds(indices, bin_ms):
    return np.asarray(indices, dtype=float) * float(bin_ms)


def safe_read_csv(path):
    try:
        if path is None:
            return pd.DataFrame()
        path = Path(path)
        if not path.exists() or not path.is_file():
            return pd.DataFrame()
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def first_existing_path(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_condition_dir(base_dir, condition_name, freq_hz):
    """
    Support multiple on-disk layouts.

    Layout A (WT22 style):
      <base>/Filtered_Data/c50/...

    Layout B (WT8 style):
      <base>/clustering/filtered/c50_f2Hz/...
    """
    base_dir = Path(base_dir)
    cond_num = "".join(ch for ch in str(condition_name) if ch.isdigit())
    filtered_roots = [
        base_dir / "Filtered_Data",
        base_dir / "Filtered Data",
        base_dir / "Filtered data",
    ]
    cond_variants = [condition_name]
    if cond_num:
        cond_variants.extend([f"C-{cond_num}", f"c-{cond_num}", f"C{cond_num}", f"c{cond_num}"])
    seen = set()
    for v in cond_variants:
        if v in seen:
            continue
        seen.add(v)
        for filtered_root_a in filtered_roots:
            if not filtered_root_a.exists():
                continue
            cand_a = filtered_root_a / v
            if cand_a.exists():
                return cand_a

    ftag = "f4Hz" if float(freq_hz) >= 3.0 else "f2Hz"
    cand_b = base_dir / "clustering" / "filtered" / f"{condition_name}_{ftag}"
    if cand_b.exists():
        return cand_b

    filtered_root = base_dir / "clustering" / "filtered"
    if filtered_root.exists():
        matches = sorted(filtered_root.glob(f"{condition_name}_f*Hz"))
        if matches:
            return matches[0]

    tried = ", ".join([str(base_dir / "Filtered_Data"), str(base_dir / "Filtered Data")])
    raise FileNotFoundError(
        f"Could not locate condition directory for {condition_name} under {base_dir}. "
        f"Tried {tried} (with name variants) and {cand_b}."
    )


@dataclass
class ConditionData:
    name: str
    raw_flat: np.ndarray
    flat_to_fourier: Dict[int, str]
    flat_to_subtype: Dict[int, str]
    subtype_available: bool
    stage4_shape: Tuple[int, int]
    valid_flat_mask: np.ndarray
    valid_flat_indices: np.ndarray
    universe_size: int
    flat_to_kept: Dict[int, int]
    kept_flat_mask: np.ndarray


@dataclass
class SegmentInfo:
    sequence_index: int
    segment_index: int
    start_bin: int
    end_bin: int
    condition: str
    contrast: float
    occurrence_index: int


@dataclass
class ModeStats:
    total_bins: int
    active_counts: np.ndarray
    active_mask: np.ndarray
    silent_mask: np.ndarray
    valid_cell_count: int


def build_state_color_map(z_reps, legend_top_n=None):
    Z = np.asarray(z_reps).astype(int)
    all_states = np.unique(Z)
    state_counts = {int(s): int(np.sum(Z == s)) for s in all_states}
    sorted_states = sorted(all_states, key=lambda s: state_counts[int(s)], reverse=True)
    if legend_top_n is not None:
        sorted_states = sorted_states[:legend_top_n]
    n_pal = len(COLORBLIND_PALETTE)
    state_to_color = {
        int(s): COLORBLIND_PALETTE[i % n_pal]
        for i, s in enumerate(sorted_states)
    }
    return state_to_color, state_counts, [int(s) for s in sorted_states]


def load_condition_data(base_dir, condition_name, freq_hz, target_universe_size=None):
    condition_dir = resolve_condition_dir(base_dir, condition_name, freq_hz)
    raw_path = next(condition_dir.glob("*_filtered_4d.npz"))
    raw = np.load(raw_path, allow_pickle=True)["array"]
    s_count, t_count, n_channels, n_units = raw.shape
    universe_size = int(n_channels * n_units)
    raw_flat = raw.reshape(s_count, t_count, universe_size)

    if target_universe_size is not None and int(target_universe_size) != universe_size:
        target_universe_size = int(target_universe_size)
        if universe_size > target_universe_size:
            raw_flat = raw_flat[:, :, :target_universe_size]
            universe_size = target_universe_size
        else:
            pad = target_universe_size - universe_size
            raw_flat = np.pad(raw_flat, ((0, 0), (0, 0), (0, pad)), mode="constant", constant_values=0)
            universe_size = target_universe_size

    valid_flat_mask = np.any(raw_flat > 0, axis=(0, 1))
    valid_flat_indices = np.flatnonzero(valid_flat_mask)

    label_map_path = Path(base_dir) / "label_index_maps" / f"{condition_name}_txtcells_label_index_map.csv"
    label_map = safe_read_csv(label_map_path)
    kept_to_flat = {}
    flat_to_kept: Dict[int, int] = {}
    if not label_map.empty:
        for row in label_map.itertuples(index=False):
            if pd.isna(row.kept_index) or pd.isna(row.flat_index):
                continue
            ki = int(row.kept_index)
            fi = int(row.flat_index)
            kept_to_flat[ki] = fi
            flat_to_kept[fi] = ki

    stage5_path = condition_dir / "Stage5_Cell_Fourier_Analysis" / "cell_fourier_results.csv"
    fourier_df = safe_read_csv(stage5_path)
    flat_to_fourier: Dict[int, str] = {}
    if not fourier_df.empty:
        for row in fourier_df.itertuples(index=False):
            flat_idx = kept_to_flat.get(int(row.cell_idx))
            label = str(row.label)
            if flat_idx is not None and label in FOURIER_LABELS and int(flat_idx) < universe_size:
                flat_to_fourier[flat_idx] = label

    stage6_dir = condition_dir / "Stage6_Cell_SustainedTransient"
    stage6_csv = first_existing_path(
        [
            stage6_dir / "cell_level_sustained_transient_KEPT.csv",
            stage6_dir / "cell_level_sustained_transient.csv",
            stage6_dir / "cell_level_sustained_transient_ALL.csv",
        ]
    )
    subtype_df = safe_read_csv(stage6_csv)
    flat_to_subtype: Dict[int, str] = {}
    subtype_available = False
    if not subtype_df.empty:
        cell_col = "cell_index" if "cell_index" in subtype_df.columns else "cell_idx" if "cell_idx" in subtype_df.columns else None
        keep_col = "keep" if "keep" in subtype_df.columns else None
        if cell_col is not None and {"polarity", "subtype"}.issubset(subtype_df.columns):
            subtype_available = True
            for row in subtype_df.itertuples(index=False):
                if keep_col is not None:
                    keep_value = getattr(row, keep_col)
                    if pd.notna(keep_value) and int(keep_value) != 1:
                        continue
                kept_index = int(getattr(row, cell_col))
                flat_idx = kept_to_flat.get(kept_index)
                polarity = str(getattr(row, "polarity"))
                subtype = str(getattr(row, "subtype"))
                if (
                    flat_idx is not None
                    and int(flat_idx) < universe_size
                    and polarity in {"ON", "OFF"}
                    and subtype in {"Sustained", "Transient"}
                ):
                    flat_to_subtype[flat_idx] = f"{polarity}-{subtype}"

    stage4_shape = tuple(np.load(condition_dir / "Stage4_Visualization" / "X_processed.npy").shape)
    kept_flat_mask = np.zeros(universe_size, dtype=bool)
    if flat_to_kept:
        for fi in flat_to_kept:
            fi = int(fi)
            if 0 <= fi < universe_size:
                kept_flat_mask[fi] = True
    else:
        kept_flat_mask = valid_flat_mask.astype(bool).copy()

    return ConditionData(
        name=condition_name,
        raw_flat=raw_flat,
        flat_to_fourier=flat_to_fourier,
        flat_to_subtype=flat_to_subtype,
        subtype_available=subtype_available,
        stage4_shape=stage4_shape,
        valid_flat_mask=valid_flat_mask,
        valid_flat_indices=valid_flat_indices,
        universe_size=universe_size,
        flat_to_kept=flat_to_kept,
        kept_flat_mask=kept_flat_mask,
    )


def build_sequence_segments(segments_s, n_sequences, condition_data, bin_ms):
    all_sequence_segments = []
    condition_counters = {cond: 0 for cond in condition_data}

    for sequence_index in range(n_sequences):
        seq_segments = []
        for segment_index, (ta, tb, contrast) in enumerate(segments_s):
            start_bin = int(round((ta * 1000.0) / bin_ms))
            end_bin = int(round((tb * 1000.0) / bin_ms))
            condition = condition_name_from_contrast(contrast)
            occurrence_index = condition_counters[condition]
            condition_counters[condition] += 1
            if occurrence_index >= condition_data[condition].raw_flat.shape[0]:
                raise ValueError(
                    f"Segment mapping overflow for {condition}: need occurrence {occurrence_index}, "
                    f"but only {condition_data[condition].raw_flat.shape[0]} repetitions exist."
                )
            seq_segments.append(
                SegmentInfo(
                    sequence_index=sequence_index,
                    segment_index=segment_index,
                    start_bin=start_bin,
                    end_bin=end_bin,
                    condition=condition,
                    contrast=float(contrast),
                    occurrence_index=occurrence_index,
                )
            )
        all_sequence_segments.append(seq_segments)
    return all_sequence_segments


def compute_mode_stats(z_reps, sequence_segments, condition_data, active_threshold, silent_threshold, global_valid_mask, universe_size):
    modes = sorted(int(m) for m in np.unique(z_reps))
    stats = {
        mode: {
            "total_bins": 0,
            "active_counts": np.zeros(int(universe_size), dtype=np.uint32),
        }
        for mode in modes
    }

    for sequence_index, seq_segments in enumerate(sequence_segments):
        seq_modes = np.asarray(z_reps[sequence_index]).astype(int)
        for seg in seq_segments:
            seg_modes = seq_modes[seg.start_bin:seg.end_bin]
            seg_raw = condition_data[seg.condition].raw_flat[seg.occurrence_index]
            for mode in np.unique(seg_modes):
                mask = seg_modes == mode
                stats[int(mode)]["total_bins"] += int(mask.sum())
                stats[int(mode)]["active_counts"] += (seg_raw[mask] > 0).sum(axis=0).astype(np.uint32)

    finalized = {}
    for mode, raw_stats in stats.items():
        total_bins = int(raw_stats["total_bins"])
        active_counts = raw_stats["active_counts"]
        if total_bins == 0:
            active_mask = np.zeros(int(universe_size), dtype=bool)
            silent_mask = np.zeros(int(universe_size), dtype=bool)
        else:
            active_ratio = active_counts.astype(float) / float(total_bins)
            active_mask = global_valid_mask & (active_ratio >= float(active_threshold))
            silent_mask = global_valid_mask & (active_counts == 0)
        finalized[mode] = ModeStats(
            total_bins=total_bins,
            active_counts=active_counts,
            active_mask=active_mask,
            silent_mask=silent_mask,
            valid_cell_count=int(global_valid_mask.sum()),
        )
    return finalized


def count_fourier_labels(flat_indices, cond_data):
    counts = {label: 0 for label in FOURIER_LABELS}
    for flat_idx in flat_indices:
        label = cond_data.flat_to_fourier.get(int(flat_idx))
        if label in counts:
            counts[label] += 1
    return counts


def _condition_lookup_order(condition_data: Dict[str, ConditionData], preferred: str) -> List[str]:
    names = sorted(
        condition_data.keys(),
        key=lambda name: (int(name[1:]) if name[1:].isdigit() else 9999, name),
    )
    if preferred in names:
        return [preferred] + [n for n in names if n != preferred]
    return names


def count_fourier_labels_multi(flat_indices, condition_data: Dict[str, ConditionData], preferred_condition: str):
    """Stage5 Fourier label per flat index; prefer the row's condition when the same flat exists in multiple maps."""
    counts = {label: 0 for label in FOURIER_LABELS}
    order = _condition_lookup_order(condition_data, preferred_condition)
    for flat_idx in flat_indices:
        fi = int(flat_idx)
        label = None
        for name in order:
            lbl = condition_data[name].flat_to_fourier.get(fi)
            if lbl in counts:
                label = lbl
                break
        if label is not None:
            counts[label] += 1
    return counts


def format_cell_id_lines(
    flat_ids: np.ndarray,
    condition_data: Dict[str, ConditionData],
    preferred_condition: str,
) -> List[str]:
    """One line per cell using the pipeline cell index from the label map only."""
    order = _condition_lookup_order(condition_data, preferred_condition)
    lines: List[str] = []
    for fi in sorted(int(x) for x in flat_ids):
        for name in order:
            k = condition_data[name].flat_to_kept.get(fi)
            if k is not None:
                lines.append(f"cell {k}")
                break
    return lines


def count_subtype_labels(flat_indices, cond_data):
    counts = {label: 0 for label in SUBTYPE_LABELS}
    for flat_idx in flat_indices:
        label = cond_data.flat_to_subtype.get(int(flat_idx))
        if label in counts:
            counts[label] += 1
    return counts


def smooth_trace(trace, sigma):
    trace = np.asarray(trace, dtype=float)
    if trace.size == 0 or sigma <= 0:
        return trace
    return gaussian_filter1d(trace, sigma=float(sigma), mode="nearest")


def active_trace_for_trial(trial_raw, active_mask, smoothing_sigma, bin_ms: float) -> np.ndarray:
    """Mean firing rate (Hz/cell) for local active cells; spikes/bin × (1000/bin_ms), then smoothed."""
    if not np.any(active_mask):
        return np.zeros(trial_raw.shape[0], dtype=float)
    scale = 1000.0 / float(bin_ms)
    mean_hz = trial_raw[:, active_mask].mean(axis=1) * scale
    return smooth_trace(mean_hz, smoothing_sigma)


def local_activity_masks(trial_raw, local_start, local_end, valid_mask):
    seg_slice = trial_raw[local_start:local_end]
    segment_active = np.any(seg_slice > 0, axis=0)
    local_active_mask = valid_mask & segment_active
    local_silent_mask = valid_mask & (~segment_active)
    return local_active_mask, local_silent_mask


def every_bin_spike_kept_mask(
    trial_raw: np.ndarray,
    local_trial_start: int,
    local_trial_end: int,
    kept_flat_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kept / label-mapped cells with a spike in every bin of this Viterbi window (same rule as segment consensus, one appearance)."""
    a, b = int(local_trial_start), int(local_trial_end)
    sl = trial_raw[a:b]
    if sl.shape[0] == 0:
        empty = np.zeros_like(kept_flat_mask, dtype=bool)
        return empty, np.array([], dtype=np.int64)
    every_bin = np.all(sl > 0, axis=0)
    m = every_bin & kept_flat_mask.astype(bool)
    return m, np.flatnonzero(m)


def compute_recording_global_active_consensus(
    z_reps: np.ndarray,
    sequence_segments: List[List[SegmentInfo]],
    condition_data: Dict[str, ConditionData],
    occurrence_fraction_threshold: float,
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """
    Per state: cells that spike in every time bin of that state on at least
    occurrence_fraction_threshold of all contiguous state runs (across all reps),
    restricted to label-map (kept) flats only. Uses the same bin boundaries as pair/full-rep views.
    """
    rep_len_bins = int(z_reps.shape[1])
    universe_size = int(next(iter(condition_data.values())).universe_size)
    all_modes = sorted(int(m) for m in np.unique(np.asarray(z_reps, dtype=int)))
    counts = {m: np.zeros(universe_size, dtype=np.uint32) for m in all_modes}
    n_occ = {m: 0 for m in all_modes}

    for rep_idx in range(int(z_reps.shape[0])):
        seq_segs = sequence_segments[rep_idx]
        pair_start, pair_end = 0, rep_len_bins
        boundary_bins = [seg.end_bin - pair_start for seg in seq_segs[:-1] if pair_start < seg.end_bin <= pair_end]
        pair_modes = np.asarray(z_reps[rep_idx][pair_start:pair_end]).astype(int)
        for local_start, local_end, mode in segments_from_states(pair_modes, force_splits=boundary_bins):
            global_start = pair_start + local_start
            owning_segment = find_segment_for_bin(seq_segs, int(global_start))
            cd = condition_data[owning_segment.condition]
            trial_raw = cd.raw_flat[owning_segment.occurrence_index]
            local_trial_start = global_start - owning_segment.start_bin
            local_trial_end = global_start + local_end - owning_segment.start_bin
            _, ga_flat = every_bin_spike_kept_mask(
                trial_raw=trial_raw,
                local_trial_start=local_trial_start,
                local_trial_end=local_trial_end,
                kept_flat_mask=cd.kept_flat_mask,
            )
            mi = int(mode)
            n_occ[mi] += 1
            counts[mi][ga_flat] += 1

    thr = float(occurrence_fraction_threshold)
    out_flat: Dict[int, np.ndarray] = {}
    for m in all_modes:
        n = int(n_occ[m])
        if n == 0:
            out_flat[m] = np.array([], dtype=np.int64)
            continue
        need = thr * float(n)
        sel = counts[m].astype(float) >= need
        out_flat[m] = np.flatnonzero(sel)
    return out_flat, n_occ


def find_segment_for_bin(seq_segments, bin_index):
    for seg in seq_segments:
        if seg.start_bin <= bin_index < seg.end_bin:
            return seg
    raise ValueError(f"Could not map bin {bin_index} to a sequence segment.")


def format_fourier_counts(counts):
    return f"ON={counts['ON']} OFF={counts['OFF']} Bip={counts['Biphasic']}"


def format_subtype_counts(counts):
    return (
        f"OS={counts['ON-Sustained']} OT={counts['ON-Transient']} "
        f"OFS={counts['OFF-Sustained']} OFT={counts['OFF-Transient']}"
    )


def subtype_text_for_row(row, activity_label):
    if not row["subtype_available"]:
        return f"{activity_label} subtype counts: NA (Stage6 unavailable)"
    if activity_label == "Active":
        counts = {
            "ON-Sustained": row["active_ON_Sustained"],
            "ON-Transient": row["active_ON_Transient"],
            "OFF-Sustained": row["active_OFF_Sustained"],
            "OFF-Transient": row["active_OFF_Transient"],
        }
    else:
        counts = {
            "ON-Sustained": row["silent_ON_Sustained"],
            "ON-Transient": row["silent_ON_Transient"],
            "OFF-Sustained": row["silent_OFF_Sustained"],
            "OFF-Transient": row["silent_OFF_Transient"],
        }
    return f"{activity_label} subtype counts: {format_subtype_counts(counts)}"


def fourier_text_for_row(row, activity_label):
    if activity_label == "Active":
        return (
            f"{activity_label} ON/OFF/Biphasic: ON={row['active_fourier_ON']} OFF={row['active_fourier_OFF']} "
            f"Bip={row['active_fourier_Biphasic']}"
        )
    return (
        f"{activity_label} ON/OFF/Biphasic: ON={row['silent_fourier_ON']} OFF={row['silent_fourier_OFF']} "
        f"Bip={row['silent_fourier_Biphasic']}"
    )


def global_fourier_text(row, activity_label):
    if activity_label == "Global active (this segment, every bin)":
        return (
            f"{activity_label} ON/OFF/Biphasic: ON={row['global_segment_every_bin_fourier_ON']} "
            f"OFF={row['global_segment_every_bin_fourier_OFF']} Bip={row['global_segment_every_bin_fourier_Biphasic']}"
        )
    return (
        f"{activity_label} ON/OFF/Biphasic: ON={row['global_silent_fourier_ON']} "
        f"OFF={row['global_silent_fourier_OFF']} Bip={row['global_silent_fourier_Biphasic']}"
    )


def local_pair_segments(sequence_segments):
    return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def draw_pair_sine(ax, pair_segments, pair_len_bins, bin_ms, freq_hz):
    x_end = bins_to_seconds(pair_len_bins, bin_ms)
    t_s = np.linspace(0.0, x_end, 1200)
    sine = np.zeros_like(t_s)

    for seg in pair_segments:
        local_ta = bins_to_seconds(seg.start_bin - pair_segments[0].start_bin, bin_ms)
        local_tb = bins_to_seconds(seg.end_bin - pair_segments[0].start_bin, bin_ms)
        mask = (t_s >= local_ta) & (t_s <= local_tb)
        if seg.contrast > 0:
            sine[mask] = (seg.contrast / 100.0) * np.sin(2 * np.pi * freq_hz * t_s[mask])
        ax.axvline(local_ta, color="#222222", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.text(
            (local_ta + local_tb) / 2.0,
            1.05,
            seg.condition,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )
    ax.axvline(x_end, color="#222222", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.plot(t_s, sine, color="black", linewidth=2.2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlim(0, x_end)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel(f"{freq_hz:.0f}Hz", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def draw_pair_mode_band(ax, pair_modes, boundary_bins, bin_ms, state_to_color):
    segs = segments_from_states(pair_modes, force_splits=boundary_bins)
    band_h = 0.84
    for start_bin, end_bin, mode in segs:
        xa = bins_to_seconds(start_bin, bin_ms)
        xb = bins_to_seconds(end_bin, bin_ms)
        color = state_to_color.get(int(mode), "#BBBBBB")
        ax.add_patch(Rectangle((xa, 0.0), xb - xa, band_h, linewidth=0, facecolor=color))
        ax.text(
            (xa + xb) / 2.0,
            band_h / 2.0,
            f"S{int(mode)}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#111111",
        )
    for boundary in boundary_bins:
        ax.axvline(bins_to_seconds(boundary, bin_ms), color="#222222", linestyle="--", linewidth=0.8, alpha=0.9)
    ax.set_xlim(0, bins_to_seconds(len(pair_modes), bin_ms))
    ax.set_ylim(-0.05, 1.25)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_instance_rows(
    sequence_index,
    z_reps,
    sequence_segments,
    condition_data,
    mode_stats,
    recording_global_by_mode: Dict[int, np.ndarray],
    recording_global_n_occ: Dict[int, int],
    smoothing_sigma,
    bin_ms,
    *,
    pair_segments=None,
    pair_window: Optional[Tuple[int, int]] = None,
):
    if pair_segments is not None:
        pair_start = pair_segments[0].start_bin
        pair_end = pair_segments[-1].end_bin
        boundary_bins = [pair_segments[0].end_bin - pair_start]
    elif pair_window is not None:
        pair_start, pair_end = int(pair_window[0]), int(pair_window[1])
        seq_segs = sequence_segments[sequence_index]
        boundary_bins = [seg.end_bin - pair_start for seg in seq_segs[:-1] if pair_start < seg.end_bin <= pair_end]
    else:
        raise ValueError("build_instance_rows requires pair_segments or pair_window")
    pair_modes = np.asarray(z_reps[sequence_index][pair_start:pair_end]).astype(int)
    pair_state_segments = segments_from_states(pair_modes, force_splits=boundary_bins)

    rows = []
    for local_start, local_end, mode in pair_state_segments:
        global_start = pair_start + local_start
        global_end = pair_start + local_end
        owning_segment = find_segment_for_bin(sequence_segments[sequence_index], global_start)
        cond_data = condition_data[owning_segment.condition]
        trial_raw = cond_data.raw_flat[owning_segment.occurrence_index]
        local_trial_start = global_start - owning_segment.start_bin
        local_trial_end = global_end - owning_segment.start_bin

        stats = mode_stats[int(mode)]
        contrast_pct = float(owning_segment.contrast)
        global_trial_start_bin = int(owning_segment.start_bin)
        mode_i = int(mode)
        rg_flat = recording_global_by_mode.get(mode_i, np.array([], dtype=np.int64))
        if isinstance(rg_flat, np.ndarray):
            rg_flat = np.asarray(rg_flat, dtype=np.int64)
        else:
            rg_flat = np.array(rg_flat, dtype=np.int64)
        global_active_indices = rg_flat
        global_silent_indices = np.flatnonzero(stats.silent_mask)
        local_active_mask, local_silent_mask = local_activity_masks(
            trial_raw=trial_raw,
            local_start=local_trial_start,
            local_end=local_trial_end,
            valid_mask=cond_data.valid_flat_mask,
        )
        local_active_indices = np.flatnonzero(local_active_mask)
        local_silent_indices = np.flatnonzero(local_silent_mask)
        active_trace = active_trace_for_trial(trial_raw, local_active_mask, smoothing_sigma, bin_ms)

        active_fourier = count_fourier_labels(local_active_indices, cond_data)
        silent_fourier = count_fourier_labels(local_silent_indices, cond_data)
        global_active_fourier = count_fourier_labels_multi(global_active_indices, condition_data, cond_data.name)
        global_silent_fourier = count_fourier_labels(global_silent_indices, cond_data)
        active_subtypes = count_subtype_labels(local_active_indices, cond_data)
        silent_subtypes = count_subtype_labels(local_silent_indices, cond_data)

        ga_every_mask, ga_every_flat = every_bin_spike_kept_mask(
            trial_raw=trial_raw,
            local_trial_start=local_trial_start,
            local_trial_end=local_trial_end,
            kept_flat_mask=cond_data.kept_flat_mask,
        )
        global_segment_every_bin_fourier = count_fourier_labels(ga_every_flat, cond_data)
        seg_cell_lines = format_cell_id_lines(ga_every_flat, condition_data, cond_data.name)
        rec_cell_lines = format_cell_id_lines(rg_flat, condition_data, cond_data.name)

        win_slice = trial_raw[local_trial_start:local_trial_end]
        if win_slice.shape[0] > 0:
            global_seg_silent_mask = cond_data.kept_flat_mask & np.all(win_slice == 0, axis=0)
        else:
            global_seg_silent_mask = np.zeros(cond_data.universe_size, dtype=bool)
        global_seg_silent_flat = np.flatnonzero(global_seg_silent_mask)
        global_seg_silent_fourier = count_fourier_labels(global_seg_silent_flat, cond_data)

        rows.append(
            {
                "sequence_index": sequence_index + 1,
                "mode": int(mode),
                "condition": owning_segment.condition,
                "condition_rep": owning_segment.occurrence_index + 1,
                "subtype_available": bool(cond_data.subtype_available),
                "pair_local_start_bin": int(local_start),
                "pair_local_end_bin": int(local_end),
                "trial_local_start_bin": int(local_trial_start),
                "trial_local_end_bin": int(local_trial_end),
                "contrast": contrast_pct,
                "global_trial_start_bin": global_trial_start_bin,
                "mode_total_bins": int(stats.total_bins),
                "condition_total_cells": int(cond_data.valid_flat_mask.sum()),
                "local_active_cells": int(local_active_mask.sum()),
                "local_silent_cells": int(local_silent_mask.sum()),
                "global_active_cells": int(rg_flat.size),
                "global_silent_cells": int(stats.silent_mask.sum()),
                "recording_global_n_occurrences": int(recording_global_n_occ.get(mode_i, 0)),
                "global_active_recording_flat": [int(x) for x in rg_flat],
                "global_active_segment_every_bin_count": int(ga_every_mask.sum()),
                "global_active_segment_every_bin_flat": [int(x) for x in ga_every_flat],
                "global_active_recording_cell_lines": rec_cell_lines,
                "global_active_segment_every_bin_cell_lines": seg_cell_lines,
                "global_active_fourier_ON": global_active_fourier["ON"],
                "global_active_fourier_OFF": global_active_fourier["OFF"],
                "global_active_fourier_Biphasic": global_active_fourier["Biphasic"],
                "global_segment_every_bin_fourier_ON": global_segment_every_bin_fourier["ON"],
                "global_segment_every_bin_fourier_OFF": global_segment_every_bin_fourier["OFF"],
                "global_segment_every_bin_fourier_Biphasic": global_segment_every_bin_fourier["Biphasic"],
                "global_silent_fourier_ON": global_silent_fourier["ON"],
                "global_silent_fourier_OFF": global_silent_fourier["OFF"],
                "global_silent_fourier_Biphasic": global_silent_fourier["Biphasic"],
                "active_fourier_ON": active_fourier["ON"],
                "active_fourier_OFF": active_fourier["OFF"],
                "active_fourier_Biphasic": active_fourier["Biphasic"],
                "silent_fourier_ON": silent_fourier["ON"],
                "silent_fourier_OFF": silent_fourier["OFF"],
                "silent_fourier_Biphasic": silent_fourier["Biphasic"],
                "global_segment_silent_count": int(global_seg_silent_mask.sum()),
                "global_segment_silent_fourier_ON": global_seg_silent_fourier["ON"],
                "global_segment_silent_fourier_OFF": global_seg_silent_fourier["OFF"],
                "global_segment_silent_fourier_Biphasic": global_seg_silent_fourier["Biphasic"],
                "local_active_flat": [int(x) for x in local_active_indices],
                "active_ON_Sustained": active_subtypes["ON-Sustained"],
                "active_ON_Transient": active_subtypes["ON-Transient"],
                "active_OFF_Sustained": active_subtypes["OFF-Sustained"],
                "active_OFF_Transient": active_subtypes["OFF-Transient"],
                "silent_ON_Sustained": silent_subtypes["ON-Sustained"],
                "silent_ON_Transient": silent_subtypes["ON-Transient"],
                "silent_OFF_Sustained": silent_subtypes["OFF-Sustained"],
                "silent_OFF_Transient": silent_subtypes["OFF-Transient"],
                "active_trace": active_trace,
            }
        )
    return rows


def _popup_segment_from_row(row, seg_index: int, state_to_color: dict, bin_ms: float, freq_hz: float) -> dict:
    contrast_pct = float(row["contrast"])
    global_trial_start_bin = int(row["global_trial_start_bin"])
    return {
        "index": seg_index,
        "mode": int(row["mode"]),
        "condition": row["condition"],
        "conditionRep": int(row["condition_rep"]),
        "localActiveCells": int(row["local_active_cells"]),
        "localSilentCells": int(row["local_silent_cells"]),
        "globalActiveCells": int(row["global_active_cells"]),
        "globalSilentCells": int(row["global_silent_cells"]),
        "recordingGlobalOccurrences": int(row["recording_global_n_occurrences"]),
        "globalActiveRecordingCellLines": list(row["global_active_recording_cell_lines"]),
        "globalActiveSegmentEveryBinCellLines": list(row["global_active_segment_every_bin_cell_lines"]),
        "globalActiveSegmentEveryBinCount": int(row["global_active_segment_every_bin_count"]),
        "globalActiveSegmentEveryBinFlat": [int(x) for x in row["global_active_segment_every_bin_flat"]],
        "pairStartBin": int(row["pair_local_start_bin"]),
        "pairEndBin": int(row["pair_local_end_bin"]),
        "trialStartBin": int(row["trial_local_start_bin"]),
        "trialEndBin": int(row["trial_local_end_bin"]),
        "color": state_to_color.get(int(row["mode"]), "#888888"),
        "contrastPct": contrast_pct,
        "globalTrialStartBin": global_trial_start_bin,
        "stimAmp": contrast_pct / 100.0,
        "trace": [round(float(v), 5) for v in np.asarray(row["active_trace"]).tolist()],
        "modeTotalBins": int(row["mode_total_bins"]),
        "conditionTotalCells": int(row["condition_total_cells"]),
        "subtypeAvailable": bool(row["subtype_available"]),
        "globalSegmentSilentCount": int(row["global_segment_silent_count"]),
        "localActiveFlatIndices": [int(x) for x in row["local_active_flat"]],
        "fourierGroups": [
            {
                "label": "Active",
                "kind": "active",
                "items": [
                    {"name": "ON", "value": int(row["active_fourier_ON"])},
                    {"name": "OFF", "value": int(row["active_fourier_OFF"])},
                    {"name": "Biphasic", "value": int(row["active_fourier_Biphasic"])},
                ],
            },
            {
                "label": "Silent",
                "kind": "silent",
                "items": [
                    {"name": "ON", "value": int(row["silent_fourier_ON"])},
                    {"name": "OFF", "value": int(row["silent_fourier_OFF"])},
                    {"name": "Biphasic", "value": int(row["silent_fourier_Biphasic"])},
                ],
            },
            {
                "label": "Always active (every bin)",
                "kind": "global-segment",
                "items": [
                    {"name": "ON", "value": int(row["global_segment_every_bin_fourier_ON"])},
                    {"name": "OFF", "value": int(row["global_segment_every_bin_fourier_OFF"])},
                    {"name": "Biphasic", "value": int(row["global_segment_every_bin_fourier_Biphasic"])},
                ],
            },
        ],
        "subtypeGroups": [
            {
                "label": "Active subtype counts",
                "kind": "active",
                "available": bool(row["subtype_available"]),
                "items": [
                    {"name": "OS", "value": int(row["active_ON_Sustained"])},
                    {"name": "OT", "value": int(row["active_ON_Transient"])},
                    {"name": "OFS", "value": int(row["active_OFF_Sustained"])},
                    {"name": "OFT", "value": int(row["active_OFF_Transient"])},
                ],
            },
            {
                "label": "Silent subtype counts",
                "kind": "silent",
                "available": bool(row["subtype_available"]),
                "items": [
                    {"name": "OS", "value": int(row["silent_ON_Sustained"])},
                    {"name": "OT", "value": int(row["silent_ON_Transient"])},
                    {"name": "OFS", "value": int(row["silent_OFF_Sustained"])},
                    {"name": "OFT", "value": int(row["silent_OFF_Transient"])},
                ],
            },
        ],
    }


def _segment_popup_metrics_help(active_threshold_pct: int) -> str:
    return (
        f"Threshold used: {active_threshold_pct}%\n"
        f"Active cells: Fired at least once in this segment.\n"
        "Silent cells: No spikes at all in this segment.\n"
        "Global silent: Never fired in this state across the entire recording.\n"
        "Global active (this segment, every bin): Label-mapped cells that fired in every time bin of this Viterbi window only.\n"
        "Fourier ON/OFF/Biphasic for that group uses Stage5 labels for this condition."
    )


def build_interactive_html_data(
    title,
    pair_label,
    pair_segments,
    pair_modes,
    instance_rows,
    state_to_color,
    bin_ms,
    freq_hz,
    active_threshold: float,
    sequence_folder_key: str,
    pair_stem: str,
    flat_to_fourier: Optional[Dict[int, str]] = None,
):
    pair_start = pair_segments[0].start_bin
    data_segments = []
    for idx, row in enumerate(instance_rows):
        data_segments.append(_popup_segment_from_row(row, idx, state_to_color, bin_ms, freq_hz))

    overview_segments = []
    for seg in pair_segments:
        overview_segments.append(
            {
                "condition": seg.condition,
                "contrast": float(seg.contrast),
                "startBin": int(seg.start_bin - pair_start),
                "endBin": int(seg.end_bin - pair_start),
            }
        )

    pct = int(round(float(active_threshold) * 100.0))
    return {
        "title": title,
        "pairLabel": pair_label.replace("_", " vs "),
        "sequenceFolderKey": sequence_folder_key,
        "pairStem": pair_stem,
        "activeThresholdPct": pct,
        "metricsHelpText": _segment_popup_metrics_help(pct),
        "binMs": float(bin_ms),
        "freqHz": float(freq_hz),
        "pairLenBins": int(len(pair_modes)),
        "trialLenBins": int(SEG_LEN_BINS),
        "overviewSegments": overview_segments,
        "segments": data_segments,
        "flatToFourier": {str(k): v for k, v in flat_to_fourier.items()} if flat_to_fourier else {},
    }


def save_pair_interactive_html(
    out_path,
    title,
    pair_label,
    pair_segments,
    pair_modes,
    instance_rows,
    state_to_color,
    bin_ms,
    freq_hz,
    active_threshold: float,
    sequence_folder_key: str,
    pair_stem: str,
    flat_to_fourier: Optional[Dict[int, str]] = None,
):
    data = build_interactive_html_data(
        title=title,
        pair_label=pair_label,
        pair_segments=pair_segments,
        pair_modes=pair_modes,
        instance_rows=instance_rows,
        state_to_color=state_to_color,
        bin_ms=bin_ms,
        freq_hz=freq_hz,
        active_threshold=active_threshold,
        sequence_folder_key=sequence_folder_key,
        pair_stem=pair_stem,
        flat_to_fourier=flat_to_fourier,
    )
    data_json = json.dumps(data)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} | {pair_label}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f7f7f7; color: #222; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 18px; }}
    h1 {{ font-size: 24px; margin: 0 0 6px; }}
    .sub {{ color: #555; margin: 0 0 16px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    svg {{ width: 100%; display: block; }}
    .plot-sine {{ height: 190px; border: 1px solid #e4e4e4; border-radius: 8px; background: #fff; }}
    .plot-trace {{ height: 240px; border: 1px solid #e4e4e4; border-radius: 8px; background: #fff; }}
    .hint {{ font-size: 14px; color: #555; margin-top: 8px; }}
    .state-rect {{ cursor: pointer; }}
    .state-rect:hover {{ stroke: #111; stroke-width: 2; }}
    dialog {{ width: min(1100px, 95vw); border: none; border-radius: 12px; padding: 0; box-shadow: 0 20px 50px rgba(0,0,0,0.35); }}
    .modal {{ padding: 18px; }}
    .modal-top {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    .modal-nav {{ display: flex; gap: 8px; align-items: center; }}
    .modal h2 {{ margin: 0; font-size: 22px; }}
    .close-btn {{ border: none; background: #222; color: white; border-radius: 8px; padding: 8px 12px; cursor: pointer; }}
    .stats {{ margin-top: 12px; display: grid; gap: 10px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 10px; margin-bottom: 12px; }}
    .metric-card {{ border: 1px solid #d8d8d8; border-radius: 10px; padding: 10px 12px; background: #fff; }}
    .metric-label {{ font-size: 13px; color: #555; margin-bottom: 4px; }}
    .metric-value {{ font-size: 28px; font-weight: 700; line-height: 1; }}
    .metric-note {{ font-size: 12px; color: #666; margin-top: 5px; }}
    .metric-active {{ border-left: 6px solid #63b8ff; }}
    .metric-silent {{ border-left: 6px solid #b8b8b8; }}
    .metric-global-active {{ border-left: 6px solid #35b36b; }}
    .metric-global-silent {{ border-left: 6px solid #d18e2f; }}
    .metric-global-segment {{ border-left: 6px solid #6a4c93; }}
    .metrics-help {{ font-size: 12px; color: #555; line-height: 1.45; margin-top: 10px; }}
    .cell-list-dialog {{ width: min(640px, 92vw); border: none; border-radius: 12px; padding: 0; box-shadow: 0 20px 50px rgba(0,0,0,0.35); }}
    .cell-list-pre {{ max-height: 52vh; overflow: auto; font-family: ui-monospace, monospace; font-size: 12px; white-space: pre-wrap; word-break: break-all; margin: 0; padding: 10px; background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; }}
    .section-title {{ font-weight: bold; margin-bottom: 6px; }}
    .segment-title {{ font-size: 18px; font-weight: bold; margin: 0 0 10px; }}
    .nav-btn {{ border: none; background: #444; color: white; border-radius: 8px; padding: 8px 12px; cursor: pointer; font-size: 18px; line-height: 1; }}
    .nav-btn:disabled {{ opacity: 0.35; cursor: default; }}
    .groups-wrap {{ display: grid; gap: 12px; }}
    .count-group {{ border: 1px solid #e4e4e4; border-radius: 10px; padding: 12px; background: #fff; }}
    .group-header {{ font-size: 18px; font-weight: 700; margin-bottom: 10px; }}
    .group-active {{ border-left: 6px solid #63b8ff; }}
    .group-silent {{ border-left: 6px solid #b8b8b8; }}
    .group-global-active {{ border-left: 6px solid #35b36b; }}
    .group-global-segment {{ border-left: 6px solid #6a4c93; }}
    .group-global-silent {{ border-left: 6px solid #d18e2f; }}
    .group-global-segment-silent {{ border-left: 6px solid #c0392b; }}
    .metric-global-segment-silent {{ border-left: 6px solid #c0392b; }}
    .chip-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .chip-grid.subtype {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .count-chip {{ border-radius: 10px; padding: 10px 8px; text-align: center; border: 1px solid #ddd; }}
    .chip-name {{ font-size: 13px; font-weight: 700; margin-bottom: 4px; }}
    .chip-value {{ font-size: 24px; font-weight: 800; line-height: 1; }}
    .chip-on {{ background: #fff2db; border-color: #efc57c; }}
    .chip-off {{ background: #eaf4ff; border-color: #94c4ff; }}
    .chip-biphasic {{ background: #f3ebff; border-color: #c7a7ff; }}
    .chip-os {{ background: #fff3de; border-color: #efc57c; }}
    .chip-ot {{ background: #fff9e6; border-color: #edd57a; }}
    .chip-ofs {{ background: #ebf5ff; border-color: #94c4ff; }}
    .chip-oft {{ background: #eefcff; border-color: #8fd5e6; }}
    .chip-oi {{ background: #f8efe4; border-color: #d9b48f; }}
    .chip-ofi {{ background: #f2f2f2; border-color: #c9c9c9; }}
    .group-note {{ font-size: 14px; color: #666; }}
    .meta-strip {{ border: 1px solid #e4e4e4; border-radius: 10px; background: #fff; padding: 10px 12px; font-size: 16px; font-weight: 600; }}
    .axis-hint {{ font-size: 13px; color: #555; margin: 0 0 12px; line-height: 1.35; }}
    .dl-btn {{ border: none; background: #1565c0; color: #fff; border-radius: 8px; padding: 8px 14px; cursor: pointer; font-size: 14px; }}
    .dl-btn:hover {{ background: #0d47a1; }}
    .details-btn {{ border: none; background: #7b4fa6; color: #fff; border-radius: 8px; padding: 8px 14px; cursor: pointer; font-size: 14px; }}
    .details-btn:hover {{ background: #5e3585; }}
    .details-dialog {{ width: min(860px, 97vw); border: none; border-radius: 14px; padding: 0; box-shadow: 0 24px 60px rgba(0,0,0,0.40); overflow: hidden; }}
    .di-wrap {{ display: flex; flex-direction: column; max-height: 90vh; }}
    .di-topbar {{ display: flex; align-items: center; gap: 10px; padding: 14px 18px; background: #1e1e2e; color: #fff; }}
    .di-mode-label {{ flex: 1; text-align: center; font-size: 14px; font-weight: 600; color: #e0e0ff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .di-nav-btn {{ border: none; background: #3d3d5c; color: #fff; border-radius: 8px; padding: 7px 14px; cursor: pointer; font-size: 14px; font-weight: 600; }}
    .di-nav-btn:hover:not(:disabled) {{ background: #5555aa; }}
    .di-nav-btn:disabled {{ opacity: 0.35; cursor: default; }}
    .di-close-btn {{ border: none; background: #c0392b; color: #fff; border-radius: 8px; padding: 7px 12px; cursor: pointer; font-size: 16px; margin-left: 6px; }}
    .di-body {{ padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 18px; }}
    .di-section {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 12px; overflow: hidden; }}
    .di-section-title {{ padding: 10px 16px; background: #f0f0f8; font-size: 13px; font-weight: 700; color: #333; letter-spacing: 0.5px; border-bottom: 1px solid #e0e0e0; }}
    .di-total {{ font-weight: 400; color: #666; font-size: 12px; }}
    .di-cell-row {{ display: flex; align-items: center; gap: 14px; padding: 14px 16px; border-bottom: 1px solid #f0f0f0; flex-wrap: wrap; }}
    .di-cell-row:last-child {{ border-bottom: none; }}
    .di-row-active {{ background: #f0f8ff; }}
    .di-row-silent {{ background: #f8f8f8; }}
    .di-row-ga {{ background: #f5f0ff; }}
    .di-row-label {{ display: flex; align-items: baseline; gap: 8px; min-width: 180px; }}
    .di-big-num {{ font-size: 32px; font-weight: 800; line-height: 1; color: #222; }}
    .di-row-name {{ font-size: 14px; font-weight: 700; color: #444; }}
    .di-row-pct {{ font-size: 13px; color: #888; }}
    .di-chip {{ border-radius: 20px; padding: 6px 12px; font-size: 13px; font-weight: 700; display: inline-flex; align-items: center; gap: 4px; }}
    .di-on {{ background: #fff2db; color: #b07800; border: 1px solid #efc57c; }}
    .di-off {{ background: #eaf4ff; color: #1a5a9a; border: 1px solid #94c4ff; }}
    .di-bip {{ background: #f3ebff; color: #6b2fa0; border: 1px solid #c7a7ff; }}
    .di-pct {{ font-weight: 400; font-size: 11px; opacity: 0.8; }}
    .di-no-stim {{ font-size: 12px; color: #bbb; font-style: italic; }}
    .di-changes {{ padding: 6px 16px 12px; display: flex; flex-direction: column; gap: 6px; }}
    .di-change-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 8px; padding: 7px 10px; border-radius: 8px; background: #fafafa; border: 1px solid #eee; }}
    .di-mini-chips {{ display: flex; gap: 5px; flex-wrap: wrap; margin-left: 4px; }}
    .di-mini-chips .di-chip {{ font-size: 11px; padding: 2px 7px; border-radius: 10px; font-weight: 700; color: white; }}
    .di-mini-chips .di-on {{ background: #d48c00; }}
    .di-mini-chips .di-off {{ background: #1460a8; }}
    .di-mini-chips .di-bip {{ background: #7020b0; }}
    .di-change-new {{ background: #f0fff4; border-color: #b8e6c4; }}
    .di-change-lost {{ background: #fff5f5; border-color: #f5c2c2; }}
    .di-change-kept {{ background: #f5f5ff; border-color: #c8c8f5; }}
    .di-change-label {{ min-width: 130px; font-size: 13px; font-weight: 600; color: #444; }}
    .di-change-val {{ font-size: 20px; font-weight: 800; min-width: 70px; color: #222; }}
    .di-change-delta {{ font-size: 13px; font-weight: 600; margin-left: auto; }}
    .delta-pos {{ color: #22863a; font-weight: 700; }}
    .delta-neg {{ color: #c0392b; font-weight: 700; }}
    .delta-zero {{ color: #888; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js" crossorigin="anonymous"></script>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{title}</h1>
      <p class="sub">{pair_label.replace("_", " vs ")}. Tap a Viterbi segment to open the details popup.</p>
      <svg id="overviewSine" class="plot-sine" viewBox="0 0 1200 170" preserveAspectRatio="none"></svg>
      <svg id="overviewStates" class="plot-trace" viewBox="0 0 1200 120" preserveAspectRatio="none"></svg>
      <div class="hint">Interactive view: click/tap a colored state block below to see the segment title, sine reference, response trace, and all counts for that segment.</div>
    </div>
  </div>

  <dialog id="segmentDialog">
    <div class="modal">
      <div class="modal-top">
        <h2>Segment details</h2>
        <div class="modal-nav">
          <button id="prevBtn" class="nav-btn" type="button" aria-label="Previous segment">&#8592;</button>
          <button id="nextBtn" class="nav-btn" type="button" aria-label="Next segment">&#8594;</button>
          <button type="button" id="detailsBtn" class="details-btn">Details</button>
          <button type="button" id="downloadPopupPng" class="dl-btn">Download PNG</button>
          <button class="close-btn" onclick="document.getElementById('segmentDialog').close()">Close</button>
        </div>
      </div>
      <div id="popupCaptureBody">
        <div class="card" style="margin-top:14px;">
          <div id="detailTitle" class="segment-title"></div>
          <p class="axis-hint">
            <b>Same time axis (trial time):</b> stimulus reference is drawn in the <i>top</i> panel only;
            mean firing rate of local active cells is in the <i>bottom</i> panel. Align vertically to compare timing (not overlaid on one plot).
          </p>
          <div class="section-title">Stimulus reference (sine, scaled by contrast)</div>
          <svg id="detailSine" class="plot-sine" viewBox="0 0 1100 170" preserveAspectRatio="none"></svg>
          <div class="section-title">Mean firing rate (local active cells, Hz)</div>
          <svg id="detailTrace" class="plot-trace" viewBox="0 0 1100 240" preserveAspectRatio="none"></svg>
        </div>
        <div id="summaryGrid" class="summary-grid"></div>
        <div id="metaStrip" class="meta-strip"></div>
        <div id="metricsHelp" class="metrics-help"></div>
        <div class="groups-wrap" id="fourierGroups"></div>
        <div class="groups-wrap" id="subtypeGroups"></div>
      </div>
    </div>
  </dialog>

  <dialog id="detailsDialog" class="details-dialog">
    <div id="detailsContent"></div>
  </dialog>

  <dialog id="cellListDialog" class="cell-list-dialog">
    <div class="modal">
      <div class="modal-top">
        <h2 id="cellListTitle">Cell list</h2>
        <button type="button" class="close-btn" onclick="document.getElementById('cellListDialog').close()">Close</button>
      </div>
      <p class="axis-hint">Lines show the pipeline cell index from the label map and the flat index in the padded stack when both are known.</p>
      <pre id="cellListPre" class="cell-list-pre"></pre>
    </div>
  </dialog>

  <script>
    const data = {data_json};
    let currentSegmentIndex = 0;

    function svgEl(name, attrs = {{}}) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      return el;
    }}

    function linePath(values, width, height, padY = 16) {{
      if (!values.length) return "";
      const yMin = Math.min(...values);
      const yMax = Math.max(...values);
      const span = (yMax - yMin) || 1;
      return values.map((v, i) => {{
        const x = (i / (values.length - 1 || 1)) * width;
        const y = height - padY - ((v - yMin) / span) * (height - padY * 2);
        return `${{i === 0 ? 'M' : 'L'}}${{x.toFixed(2)}},${{y.toFixed(2)}}`;
      }}).join(" ");
    }}

    function drawOverview() {{
      const sineSvg = document.getElementById("overviewSine");
      const stateSvg = document.getElementById("overviewStates");
      const width = 1200;
      const sineHeight = 170;
      const bandHeight = 120;
      const pairLen = data.pairLenBins;

      sineSvg.innerHTML = "";
      stateSvg.innerHTML = "";

      sineSvg.appendChild(svgEl("rect", {{x: 0, y: 0, width, height: sineHeight, fill: "white"}}));
      stateSvg.appendChild(svgEl("rect", {{x: 0, y: 0, width, height: bandHeight, fill: "white"}}));

      const xs = [];
      const ys = [];
      for (let i = 0; i < 1200; i++) {{
        const bin = (i / 1199) * pairLen;
        let amp = 0;
        for (const seg of data.overviewSegments) {{
          if (bin >= seg.startBin && bin <= seg.endBin) {{
            amp = seg.contrast / 100.0;
            break;
          }}
        }}
        const t = (bin * data.binMs) / 1000.0;
        xs.push((i / 1199) * width);
        ys.push(amp * Math.sin(2 * Math.PI * data.freqHz * t));
      }}
      const path = xs.map((x, i) => {{
        const y = 85 - ys[i] * 60;
        return `${{i === 0 ? 'M' : 'L'}}${{x.toFixed(2)}},${{y.toFixed(2)}}`;
      }}).join(" ");
      sineSvg.appendChild(svgEl("path", {{d: path, fill: "none", stroke: "#111", "stroke-width": 2.8}}));
      sineSvg.appendChild(svgEl("line", {{x1: 0, y1: 85, x2: width, y2: 85, stroke: "#999", "stroke-dasharray": "4 4"}}));

      for (const seg of data.overviewSegments) {{
        const x0 = (seg.startBin / pairLen) * width;
        const x1 = (seg.endBin / pairLen) * width;
        sineSvg.appendChild(svgEl("line", {{x1: x0, y1: 0, x2: x0, y2: sineHeight, stroke: "#222", "stroke-dasharray": "4 4"}}));
        sineSvg.appendChild(svgEl("text", {{x: (x0 + x1) / 2, y: 20, "text-anchor": "middle", "font-size": 15, "font-weight": "bold", fill: "#333"}})).textContent = seg.condition;
      }}
      sineSvg.appendChild(svgEl("line", {{x1: width, y1: 0, x2: width, y2: sineHeight, stroke: "#222", "stroke-dasharray": "4 4"}}));

      for (const seg of data.segments) {{
        const x0 = (seg.pairStartBin / pairLen) * width;
        const x1 = (seg.pairEndBin / pairLen) * width;
        const rect = svgEl("rect", {{
          x: x0, y: 20, width: Math.max(1, x1 - x0), height: 60,
          fill: seg.color, class: "state-rect", rx: 4, ry: 4
        }});
        rect.addEventListener("click", () => showSegment(seg));
        stateSvg.appendChild(rect);
        stateSvg.appendChild(svgEl("text", {{
          x: (x0 + x1) / 2, y: 56, "text-anchor": "middle",
          "font-size": 14, "font-weight": "bold", fill: "#111"
        }})).textContent = `S${{seg.mode}}`;
      }}
    }}

    const PLOT_LEFT = 72;
    const PLOT_RIGHT = 18;

    function plotInnerWidth(fullW) {{
      return fullW - PLOT_LEFT - PLOT_RIGHT;
    }}

    function xAtBinIndex(i, fullW) {{
      const pw = plotInnerWidth(fullW);
      const n = data.trialLenBins;
      return PLOT_LEFT + (i / Math.max(n - 1, 1)) * pw;
    }}

    function xAtBinBoundary(b, fullW) {{
      const pw = plotInnerWidth(fullW);
      return PLOT_LEFT + (b / data.trialLenBins) * pw;
    }}

    function drawDetailSine(seg) {{
      const svg = document.getElementById("detailSine");
      const width = 1100;
      const height = 170;
      const midY = 85;
      const amp = 60;
      const pw = plotInnerWidth(width);
      svg.innerHTML = "";
      svg.appendChild(svgEl("rect", {{x: 0, y: 0, width, height, fill: "white"}}));
      svg.appendChild(svgEl("line", {{x1: PLOT_LEFT, y1: midY, x2: width - PLOT_RIGHT, y2: midY, stroke: "#999", "stroke-dasharray": "4 4"}}));

      const condLabel = seg.contrastPct > 0
        ? `${{seg.condition}} contrast reference (c=${{seg.contrastPct}}%)`
        : `${{seg.condition}} no stimulus (c=0%)`;
      svg.appendChild(svgEl("text", {{x: PLOT_LEFT, y: 18, "font-size": 14, "font-weight": "bold", fill: "#222"}})).textContent = condLabel;

      const xs = [];
      const ys = [];
      for (let i = 0; i < data.trialLenBins; i++) {{
        const t = (i * data.binMs) / 1000.0;
        xs.push(xAtBinIndex(i, width));
        ys.push(seg.stimAmp * Math.sin(2 * Math.PI * data.freqHz * t));
      }}

      if (seg.contrastPct > 0) {{
        const path = xs.map((x, i) => {{
          const y = midY - ys[i] * amp;
          return `${{i === 0 ? 'M' : 'L'}}${{x.toFixed(2)}},${{y.toFixed(2)}}`;
        }}).join(" ");
        svg.appendChild(svgEl("path", {{d: path, fill: "none", stroke: "#111", "stroke-width": 2.8}}));
      }} else {{
        svg.appendChild(svgEl("line", {{x1: PLOT_LEFT, y1: midY, x2: width - PLOT_RIGHT, y2: midY, stroke: "#444", "stroke-width": 2.0}}));
        svg.appendChild(svgEl("text", {{
          x: PLOT_LEFT + pw / 2, y: midY - 14,
          "text-anchor": "middle", "font-size": 14, fill: "#888", "font-style": "italic"
        }})).textContent = "No sinusoidal stimulus, baseline (contrast = 0%)";
      }}

      const x0 = xAtBinBoundary(seg.trialStartBin, width);
      const x1 = xAtBinBoundary(seg.trialEndBin, width);
      svg.appendChild(svgEl("rect", {{x: x0, y: 0, width: Math.max(1, x1 - x0), height, fill: seg.color, opacity: 0.13}}));
      svg.appendChild(svgEl("line", {{x1: x0, y1: 0, x2: x0, y2: height, stroke: "#333", "stroke-dasharray": "5 5"}}));
      svg.appendChild(svgEl("line", {{x1: x1, y1: 0, x2: x1, y2: height, stroke: "#333", "stroke-dasharray": "5 5"}}));

      const xMaxMs = (data.trialLenBins * data.binMs);
      [0, 0.25, 0.5, 0.75, 1.0].forEach((f) => {{
        const ms = f * xMaxMs;
        const bx = PLOT_LEFT + f * pw;
        svg.appendChild(svgEl("line", {{x1: bx, y1: height - 2, x2: bx, y2: height, stroke: "#666"}}));
        svg.appendChild(svgEl("text", {{
          x: bx, y: height - 6, "text-anchor": "middle", "font-size": 11, fill: "#666"
        }})).textContent = String(Math.round(ms));
      }});
    }}

    function drawDetailTrace(seg) {{
      const svg = document.getElementById("detailTrace");
      const width = 1100;
      const height = 240;
      const top = 14;
      const bottom = 36;
      const plotW = plotInnerWidth(width);
      const plotH = height - top - bottom;
      svg.innerHTML = "";
      svg.appendChild(svgEl("rect", {{x: 0, y: 0, width, height, fill: "white"}}));
      const vals = seg.trace || [];
      const yLabel = "Mean FR (Hz/cell)";
      let yMin = vals.length ? Math.min(...vals) : 0;
      let yMax = vals.length ? Math.max(...vals) : 1;
      if (Math.abs(yMax - yMin) < 1e-12) {{
        const pad = yMax === 0 ? 1.0 : Math.max(0.1, Math.abs(yMax) * 0.1);
        yMin -= pad;
        yMax += pad;
      }}
      const yOf = (v) => top + ((yMax - v) / (yMax - yMin)) * plotH;
      const xOf = (i) => xAtBinIndex(i, width);

      svg.appendChild(svgEl("line", {{x1: PLOT_LEFT, y1: top, x2: PLOT_LEFT, y2: top + plotH, stroke: "#666"}}));
      svg.appendChild(svgEl("line", {{x1: PLOT_LEFT, y1: top + plotH, x2: PLOT_LEFT + plotW, y2: top + plotH, stroke: "#666"}}));
      svg.appendChild(svgEl("text", {{
        x: 20, y: top + plotH / 2, "text-anchor": "middle", "font-size": 13, fill: "#333",
        transform: `rotate(-90 20 ${{(top + plotH / 2).toFixed(2)}})`
      }})).textContent = yLabel;

      for (const tv of [yMax, (yMin + yMax) / 2, yMin]) {{
        const ty = yOf(tv);
        svg.appendChild(svgEl("line", {{x1: PLOT_LEFT - 4, y1: ty, x2: PLOT_LEFT + plotW, y2: ty, stroke: "#ddd"}}));
        svg.appendChild(svgEl("text", {{x: PLOT_LEFT - 8, y: ty + 4, "text-anchor": "end", "font-size": 12, fill: "#444"}})).textContent = tv.toFixed(1);
      }}

      if (vals.length) {{
        const path = vals.map((v, i) => `${{i === 0 ? 'M' : 'L'}}${{xOf(i).toFixed(2)}},${{yOf(v).toFixed(2)}}`).join(" ");
        svg.appendChild(svgEl("path", {{d: path, fill: "none", stroke: seg.color, "stroke-width": 3.2}}));
      }}

      const x0b = xAtBinBoundary(seg.trialStartBin, width);
      const x1b = xAtBinBoundary(seg.trialEndBin, width);
      svg.appendChild(svgEl("rect", {{x: x0b, y: top, width: Math.max(1, x1b - x0b), height: plotH, fill: seg.color, opacity: 0.16}}));
      svg.appendChild(svgEl("line", {{x1: x0b, y1: top, x2: x0b, y2: top + plotH, stroke: "#333", "stroke-dasharray": "5 5"}}));
      svg.appendChild(svgEl("line", {{x1: x1b, y1: top, x2: x1b, y2: top + plotH, stroke: "#333", "stroke-dasharray": "5 5"}}));

      const xMaxMs = (data.trialLenBins * data.binMs);
      [0, 0.25, 0.5, 0.75, 1.0].forEach((f) => {{
        const ms = f * xMaxMs;
        const bx = PLOT_LEFT + f * plotW;
        svg.appendChild(svgEl("line", {{x1: bx, y1: top + plotH, x2: bx, y2: top + plotH + 5, stroke: "#666"}}));
        svg.appendChild(svgEl("text", {{
          x: bx, y: top + plotH + 22, "text-anchor": "middle", "font-size": 12, fill: "#333"
        }})).textContent = String(Math.round(ms));
      }});
      svg.appendChild(svgEl("text", {{
        x: PLOT_LEFT + plotW / 2, y: height - 6, "text-anchor": "middle", "font-size": 13, fill: "#444", "font-weight": "600"
      }})).textContent = "Time in trial (ms)";
    }}

    function safeFilePart(s) {{
      return String(s).replace(/[^a-zA-Z0-9._-]+/g, "_");
    }}

    function buildPopupPngName(seg) {{
      return `${{safeFilePart(data.sequenceFolderKey)}}_${{safeFilePart(data.pairStem)}}_S${{seg.mode}}_${{safeFilePart(seg.condition)}}_rep${{seg.conditionRep}}_bins${{seg.trialStartBin}}-${{seg.trialEndBin}}_meanFR_perCell_Hz.png`;
    }}

    async function downloadPopupAsPng() {{
      if (typeof html2canvas === "undefined") {{
        alert("html2canvas failed to load — cannot export PNG.");
        return;
      }}
      const seg = data.segments[currentSegmentIndex];
      const el = document.getElementById("popupCaptureBody");
      const canvas = await html2canvas(el, {{ scale: 2, backgroundColor: "#ffffff", logging: false }});
      const name = buildPopupPngName(seg);
      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png");
      a.download = name;
      a.click();
    }}

    function chipClass(name) {{
      const key = String(name).toLowerCase();
      const map = {{
        on: "count-chip chip-on",
        off: "count-chip chip-off",
        biphasic: "count-chip chip-biphasic",
        os: "count-chip chip-os",
        ot: "count-chip chip-ot",
        ofs: "count-chip chip-ofs",
        oft: "count-chip chip-oft",
      }};
      return map[key] || "count-chip";
    }}

    function groupClass(kind) {{
      if (kind === "active") return "count-group group-active";
      if (kind === "silent") return "count-group group-silent";
      if (kind === "global-active") return "count-group group-global-active";
      if (kind === "global-segment") return "count-group group-global-segment";
      if (kind === "global-silent") return "count-group group-global-silent";
      if (kind === "global-segment-silent") return "count-group group-global-segment-silent";
      return "count-group";
    }}

    function openCellList(title, lines) {{
      const t = document.getElementById("cellListTitle");
      if (t) t.textContent = title || "Cell list";
      const pre = document.getElementById("cellListPre");
      const arr = Array.isArray(lines) ? lines : [];
      pre.textContent = arr.length ? arr.join(String.fromCharCode(10)) : "(none)";
      document.getElementById("cellListDialog").showModal();
    }}

    function findPrevSegment(seg) {{
      const sameRep = data.segments.filter(s => s.repIndex === seg.repIndex && s.pairStartBin < seg.pairStartBin);
      if (!sameRep.length) return null;
      sameRep.sort((a, b) => b.pairStartBin - a.pairStartBin);
      return sameRep[0];
    }}

    function setOps(prevArr, currArr) {{
      const prevSet = new Set(prevArr);
      const currSet = new Set(currArr);
      return {{
        added: currArr.filter(x => !prevSet.has(x)),
        lost: prevArr.filter(x => !currSet.has(x)),
        common: currArr.filter(x => prevSet.has(x)),
      }};
    }}

    function fourierChipsOf(indices) {{
      if (!data.flatToFourier || !indices || indices.length === 0) return "";
      let on = 0, off = 0, bip = 0;
      for (const fi of indices) {{
        const lbl = data.flatToFourier[String(fi)];
        if (lbl === "ON") on++;
        else if (lbl === "OFF") off++;
        else if (lbl === "Biphasic") bip++;
      }}
      const n = indices.length;
      const fmt = c => n > 0 ? `${{c}} (${{(c/n*100).toFixed(0)}}%)` : "—";
      return `<span class="di-mini-chips">` +
        `<span class="di-chip di-on">ON ${{fmt(on)}}</span>` +
        `<span class="di-chip di-off">OFF ${{fmt(off)}}</span>` +
        `<span class="di-chip di-bip">Bip ${{fmt(bip)}}</span>` +
        `</span>`;
    }}

    function deltaClass(n) {{
      if (n > 0) return "delta-pos";
      if (n < 0) return "delta-neg";
      return "delta-zero";
    }}

    function getFourierGroup(seg, kind) {{
      return (seg.fourierGroups || []).find(g => g.kind === kind) || {{ items: [] }};
    }}

    function fourierInlineHtml(group, total) {{
      const items = group.items || [];
      if (!items.length) return "";
      const pct = n => total > 0 ? ((n/total)*100).toFixed(0)+"%" : "—";
      return items.map(it => {{
        const cls = it.name === "ON"  ? "di-on"
                  : it.name === "OFF" ? "di-off" : "di-bip";
        return `<span class="di-chip ${{cls}}">${{it.name}} ${{it.value}} <span class="di-pct">(${{pct(it.value)}})</span></span>`;
      }}).join("");
    }}

    function navigateDetails(delta) {{
      const newIdx = currentSegmentIndex + delta;
      if (newIdx < 0 || newIdx >= data.segments.length) return;
      currentSegmentIndex = newIdx;
      const nextSeg = data.segments[currentSegmentIndex];
      // Silently refresh main popup content
      const dt = document.getElementById("detailTitle");
      if (dt) dt.textContent = nextSeg.repLabel
        ? `${{data.title}} | ${{nextSeg.repLabel}} | ${{nextSeg.condition}} rep ${{nextSeg.conditionRep}} | State ${{nextSeg.mode}} | bins ${{nextSeg.trialStartBin}}-${{nextSeg.trialEndBin}}`
        : `${{data.title}} | ${{nextSeg.condition}} rep ${{nextSeg.conditionRep}} | State ${{nextSeg.mode}} | bins ${{nextSeg.trialStartBin}}-${{nextSeg.trialEndBin}}`;
      try {{ drawDetailSine(nextSeg); drawDetailTrace(nextSeg); }} catch(e) {{}}
      renderSummary(nextSeg);
      updateNavButtons();
      const isC0 = Math.abs(nextSeg.contrastPct) < 0.5;
      if (isC0) {{
        document.getElementById("fourierGroups").innerHTML = `<p style="color:#999;font-size:13px;padding:8px 0">No Fourier/subtype breakdown for c0 (no stimulus present).</p>`;
        document.getElementById("subtypeGroups").innerHTML = "";
      }} else {{
        renderGroups("fourierGroups", nextSeg.fourierGroups, false);
        renderGroups("subtypeGroups", nextSeg.subtypeGroups, true);
      }}
      showDetails(nextSeg);
    }}

    function showDetails(seg) {{
      const total = seg.conditionTotalCells || 1;
      const pctOf = n => ((n / total) * 100).toFixed(1) + "%";
      const isC0 = Math.abs(seg.contrastPct) < 0.5;

      // Fourier groups from payload
      const actGroup  = getFourierGroup(seg, "active");
      const silGroup  = getFourierGroup(seg, "silent");
      const gaGroup   = getFourierGroup(seg, "global-segment");

      const prevSeg = findPrevSegment(seg);
      const nextSeg = data.segments[currentSegmentIndex + 1] || null;
      const hasPrev = currentSegmentIndex > 0;
      const hasNext = currentSegmentIndex < data.segments.length - 1;

      const modeLabel = `S${{seg.mode}} | ${{seg.condition}} rep ${{seg.conditionRep}} | c${{seg.contrastPct}}% | bins ${{seg.trialStartBin}}–${{seg.trialEndBin}}`;

      let html = `
      <div class="di-wrap">
        <div class="di-topbar">
          <button class="di-nav-btn" ${{hasPrev ? "" : "disabled"}} onclick="navigateDetails(-1)">&#8592; Prev</button>
          <span class="di-mode-label">${{modeLabel}}</span>
          <button class="di-nav-btn" ${{hasNext ? "" : "disabled"}} onclick="navigateDetails(1)">Next &#8594;</button>
          <button class="di-close-btn" onclick="document.getElementById('detailsDialog').close()">&#10005;</button>
        </div>

        <div class="di-body">
          <div class="di-section">
            <div class="di-section-title">&#9632; CELL POPULATIONS &nbsp;<span class="di-total">total kept: ${{total}}</span></div>
            <div class="di-cell-row di-row-active">
              <div class="di-row-label">
                <span class="di-big-num">${{seg.localActiveCells}}</span>
                <span class="di-row-name">Active</span>
                <span class="di-row-pct">${{pctOf(seg.localActiveCells)}}</span>
              </div>
              ${{isC0 ? '<span class="di-no-stim">no stimulus — Fourier n/a</span>' : fourierInlineHtml(actGroup, seg.localActiveCells)}}
            </div>
            <div class="di-cell-row di-row-silent">
              <div class="di-row-label">
                <span class="di-big-num">${{seg.globalSegmentSilentCount}}</span>
                <span class="di-row-name">Silent</span>
                <span class="di-row-pct">${{pctOf(seg.globalSegmentSilentCount)}}</span>
              </div>
              ${{isC0 ? '<span class="di-no-stim">no stimulus — Fourier n/a</span>' : fourierInlineHtml(silGroup, seg.globalSegmentSilentCount)}}
            </div>
            <div class="di-cell-row di-row-ga">
              <div class="di-row-label">
                <span class="di-big-num">${{seg.globalActiveSegmentEveryBinCount}}</span>
                <span class="di-row-name">Always active</span>
                <span class="di-row-pct">${{pctOf(seg.globalActiveSegmentEveryBinCount)}}<br><span style="font-size:11px;color:#888">spike every bin</span></span>
              </div>
              ${{isC0 ? '<span class="di-no-stim">no stimulus — Fourier n/a</span>' : fourierInlineHtml(gaGroup, seg.globalActiveSegmentEveryBinCount)}}
            </div>
          </div>`;

      if (prevSeg) {{
        const prevCount = prevSeg.localActiveCells || 0;
        const currCount = seg.localActiveCells;
        const rawDelta  = currCount - prevCount;
        const sign      = rawDelta >= 0 ? "+" : "";
        const pctDelta  = prevCount > 0 ? `${{sign}}${{((rawDelta/prevCount)*100).toFixed(1)}}%` : "N/A";
        const arrowIcon = rawDelta > 0 ? "▲" : rawDelta < 0 ? "▼" : "■";
        const arrowCls  = rawDelta > 0 ? "delta-pos" : rawDelta < 0 ? "delta-neg" : "delta-zero";

        let changesHtml = `
          <div class="di-change-row">
            <span class="di-change-label">Active count</span>
            <span class="di-change-val">${{prevCount}} → ${{currCount}}</span>
            <span class="di-change-delta ${{arrowCls}}">${{arrowIcon}} ${{sign}}${{rawDelta}} (${{pctDelta}})</span>
          </div>`;

        if (seg.localActiveFlatIndices && prevSeg.localActiveFlatIndices) {{
          const ops = setOps(prevSeg.localActiveFlatIndices, seg.localActiveFlatIndices);
          const pOf = (n,d) => d > 0 ? ((n/d)*100).toFixed(1)+"%" : "—";
          changesHtml += `
          <div class="di-change-row di-change-new">
            <span class="di-change-label">&#10133; New cells</span>
            <span class="di-change-val">${{ops.added.length}}</span>
            <span class="di-change-delta delta-pos">${{pOf(ops.added.length, prevCount)}} of prev</span>
            ${{!isC0 ? fourierChipsOf(ops.added) : ''}}
          </div>
          <div class="di-change-row di-change-lost">
            <span class="di-change-label">&#10134; Lost cells</span>
            <span class="di-change-val">${{ops.lost.length}}</span>
            <span class="di-change-delta delta-neg">−${{pOf(ops.lost.length, prevCount)}} of prev</span>
            ${{!isC0 ? fourierChipsOf(ops.lost) : ''}}
          </div>
          <div class="di-change-row di-change-kept">
            <span class="di-change-label">&#8635; Carried over</span>
            <span class="di-change-val">${{ops.common.length}}</span>
            <span class="di-change-delta">${{pOf(ops.common.length, prevCount)}} of prev</span>
            ${{!isC0 ? fourierChipsOf(ops.common) : ''}}
          </div>`;
        }}

        html += `<div class="di-section">
          <div class="di-section-title">&#9632; VS PREVIOUS MODE &nbsp;<span class="di-total">S${{prevSeg.mode}}, ${{prevSeg.condition}} rep ${{prevSeg.conditionRep}}, bins ${{prevSeg.trialStartBin}}–${{prevSeg.trialEndBin}}</span></div>
          <div class="di-changes">${{changesHtml}}</div>
        </div>`;
      }} else {{
        html += `<div class="di-section"><div class="di-section-title">&#9632; VS PREVIOUS MODE</div>
          <p style="color:#999;font-size:14px;padding:8px 0">No previous segment in this repetition.</p></div>`;
      }}

      html += `</div></div>`;
      document.getElementById("detailsContent").innerHTML = html;
      document.getElementById("detailsDialog").showModal();
    }}

    function renderSummary(seg) {{
      const summary = document.getElementById("summaryGrid");
      summary.innerHTML = "";
      const nBins = seg.trialEndBin - seg.trialStartBin;
      const cards = [
        {{
          cls: "metric-card metric-active",
          label: "Active",
          value: String(seg.localActiveCells),
          note: "≥1 spike in this window",
          list: false,
        }},
        {{
          cls: "metric-card metric-global-segment-silent",
          label: "Silent",
          value: String(seg.globalSegmentSilentCount),
          note: "0 spikes in ALL bins of this window",
          list: false,
        }},
        {{
          cls: "metric-card metric-global-segment",
          label: "Always active (every bin)",
          value: String(seg.globalActiveSegmentEveryBinCount),
          note: `spike in every bin (${{nBins}} bins)`,
          list: "segment",
        }},
      ];
      cards.forEach(card => {{
        const div = document.createElement("div");
        div.className = card.cls;
        div.innerHTML = `<div class="metric-label">${{card.label}}</div><div class="metric-value">${{card.value}}</div><div class="metric-note">${{card.note}}</div>`;
        if (card.list === "segment" && seg.globalActiveSegmentEveryBinCount > 0) {{
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "dl-btn";
          btn.style.marginTop = "8px";
          btn.textContent = "Show cells";
          btn.addEventListener("click", () => openCellList("Always active (every bin)", seg.globalActiveSegmentEveryBinCellLines || []));
          div.appendChild(btn);
        }}
        summary.appendChild(div);
      }});
    }}

    function renderGroups(targetId, groups, subtype=false) {{
      const target = document.getElementById(targetId);
      target.innerHTML = "";
      groups.forEach(group => {{
        const wrapper = document.createElement("div");
        wrapper.className = groupClass(group.kind);
        const header = document.createElement("div");
        header.className = "group-header";
        header.textContent = group.label;
        wrapper.appendChild(header);
        if (group.available === false) {{
          const note = document.createElement("div");
          note.className = "group-note";
          note.textContent = "Subtype labels are not available for this condition.";
          wrapper.appendChild(note);
        }} else {{
          const grid = document.createElement("div");
          grid.className = subtype ? "chip-grid subtype" : "chip-grid";
          group.items.forEach(item => {{
            const chip = document.createElement("div");
            chip.className = chipClass(item.name);
            chip.innerHTML = `<div class="chip-name">${{item.name}}</div><div class="chip-value">${{item.value}}</div>`;
            grid.appendChild(chip);
          }});
          wrapper.appendChild(grid);
        }}
        target.appendChild(wrapper);
      }});
    }}

    function updateNavButtons() {{
      document.getElementById("prevBtn").disabled = currentSegmentIndex <= 0;
      document.getElementById("nextBtn").disabled = currentSegmentIndex >= data.segments.length - 1;
    }}

    function showSegment(segOrIndex) {{
      const seg = typeof segOrIndex === "number" ? data.segments[segOrIndex] : segOrIndex;
      currentSegmentIndex = seg.index;
      document.getElementById("detailTitle").textContent =
        `${{data.title}} | ${{seg.condition}} rep ${{seg.conditionRep}} | State ${{seg.mode}} | bins ${{seg.trialStartBin}}-${{seg.trialEndBin}}`;
      drawDetailSine(seg);
      drawDetailTrace(seg);
      renderSummary(seg);
      document.getElementById("metaStrip").textContent =
        `Bins labeled with this state (dataset): ${{seg.modeTotalBins}}`;
      const mh = document.getElementById("metricsHelp");
      if (mh) mh.textContent = data.metricsHelpText || "";
      const isC0 = Math.abs(seg.contrastPct) < 0.5;
      if (isC0) {{
        document.getElementById("fourierGroups").innerHTML = `<p style="color:#999;font-size:13px;padding:8px 0">No Fourier/subtype breakdown for c0 (no stimulus present).</p>`;
        document.getElementById("subtypeGroups").innerHTML = "";
      }} else {{
        renderGroups("fourierGroups", seg.fourierGroups, false);
        renderGroups("subtypeGroups", seg.subtypeGroups, true);
      }}
      updateNavButtons();
      document.getElementById("segmentDialog").showModal();
    }}

    document.getElementById("prevBtn").addEventListener("click", () => {{
      if (currentSegmentIndex > 0) showSegment(currentSegmentIndex - 1);
    }});
    document.getElementById("nextBtn").addEventListener("click", () => {{
      if (currentSegmentIndex < data.segments.length - 1) showSegment(currentSegmentIndex + 1);
    }});
    document.addEventListener("keydown", (evt) => {{
      const dialog = document.getElementById("segmentDialog");
      if (!dialog.open) return;
      if (evt.key === "ArrowLeft" && currentSegmentIndex > 0) {{
        showSegment(currentSegmentIndex - 1);
      }}
      if (evt.key === "ArrowRight" && currentSegmentIndex < data.segments.length - 1) {{
        showSegment(currentSegmentIndex + 1);
      }}
    }});

    document.getElementById("downloadPopupPng").addEventListener("click", () => downloadPopupAsPng());
    document.getElementById("detailsBtn").addEventListener("click", () => {{
      const seg = data.segments[currentSegmentIndex];
      if (seg) showDetails(seg);
    }});

    drawOverview();
  </script>
</body>
</html>
"""
    Path(out_path).write_text(html, encoding="utf-8")


def build_all_reps_interactive_payload(
    title: str,
    z_reps: np.ndarray,
    rep_labels: List[str],
    segments_s: List[Tuple[float, float, float]],
    sequence_segments: List[List[SegmentInfo]],
    condition_data: Dict[str, ConditionData],
    mode_stats: Dict[int, ModeStats],
    recording_global_by_mode: Dict[int, np.ndarray],
    recording_global_n_occ: Dict[int, int],
    state_to_color: dict,
    bin_ms: float,
    freq_hz: float,
    active_threshold: float,
    smoothing_sigma: float,
) -> dict:
    rep_len_bins = int(z_reps.shape[1])
    overview_segments = []
    for ta, tb, c in segments_s:
        overview_segments.append(
            {
                "condition": condition_name_from_contrast(c),
                "contrast": float(c),
                "startBin": int(round((ta * 1000.0) / bin_ms)),
                "endBin": int(round((tb * 1000.0) / bin_ms)),
            }
        )
    pct = int(round(float(active_threshold) * 100.0))
    flat_to_fourier_map: Dict[int, str] = {}
    for cd in condition_data.values():
        flat_to_fourier_map.update(cd.flat_to_fourier)
    data_segments: List[dict] = []
    gidx = 0
    for rep_idx in range(int(z_reps.shape[0])):
        instance_rows = build_instance_rows(
            sequence_index=rep_idx,
            z_reps=z_reps,
            sequence_segments=sequence_segments,
            condition_data=condition_data,
            mode_stats=mode_stats,
            recording_global_by_mode=recording_global_by_mode,
            recording_global_n_occ=recording_global_n_occ,
            smoothing_sigma=smoothing_sigma,
            bin_ms=bin_ms,
            pair_window=(0, rep_len_bins),
        )
        for row in instance_rows:
            seg = _popup_segment_from_row(row, gidx, state_to_color, bin_ms, freq_hz)
            seg["repIndex"] = rep_idx
            seg["repLabel"] = rep_labels[rep_idx]
            data_segments.append(seg)
            gidx += 1
    return {
        "pageKind": "all_reps",
        "title": title,
        "sequenceFolderKey": "all_reps",
        "pairStem": "",
        "activeThresholdPct": pct,
        "metricsHelpText": _segment_popup_metrics_help(pct),
        "binMs": float(bin_ms),
        "freqHz": float(freq_hz),
        "pairLenBins": rep_len_bins,
        "trialLenBins": int(SEG_LEN_BINS),
        "nReps": int(z_reps.shape[0]),
        "repLabels": rep_labels,
        "stateLegend": [
            {"state": int(s), "color": c}
            for s, c in sorted(((int(k), v) for k, v in state_to_color.items()), key=lambda kv: kv[0])
        ],
        "overviewSegments": overview_segments,
        "segments": data_segments,
        "flatToFourier": {str(k): v for k, v in flat_to_fourier_map.items()},
    }


def save_all_reps_interactive_html(
    viterbi_dir: Path,
    page_title: str,
    payload: dict,
) -> Path:
    viterbi_dir = Path(viterbi_dir)
    safe_title = html.escape(page_title)
    data_json = json.dumps(payload)
    out_path = viterbi_dir / "all_reps_interactive.html"
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title} | All reps</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f7f7f7; color: #222; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 18px; }}
    h1 {{ font-size: 24px; margin: 0 0 6px; }}
    .sub {{ color: #555; margin: 0 0 16px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    svg {{ width: 100%; display: block; }}
    .plot-sine {{ height: 190px; border: 1px solid #e4e4e4; border-radius: 8px; background: #fff; }}
    .plot-reps {{ min-height: 120px; border: 1px solid #e4e4e4; border-radius: 8px; background: #fff; }}
    .hint {{ font-size: 14px; color: #555; margin-top: 8px; }}
    .state-rect {{ cursor: pointer; }}
    .state-rect:hover {{ stroke: #111; stroke-width: 2; }}
    dialog {{ width: min(1100px, 95vw); border: none; border-radius: 12px; padding: 0; box-shadow: 0 20px 50px rgba(0,0,0,0.35); }}
    .modal {{ padding: 18px; }}
    .modal-top {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    .modal-nav {{ display: flex; gap: 8px; align-items: center; }}
    .modal h2 {{ margin: 0; font-size: 22px; }}
    .close-btn {{ border: none; background: #222; color: white; border-radius: 8px; padding: 8px 12px; cursor: pointer; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 10px; margin-bottom: 12px; }}
    .metric-card {{ border: 1px solid #d8d8d8; border-radius: 10px; padding: 10px 12px; background: #fff; }}
    .metric-label {{ font-size: 13px; color: #555; margin-bottom: 4px; }}
    .metric-value {{ font-size: 28px; font-weight: 700; line-height: 1; }}
    .metric-note {{ font-size: 12px; color: #666; margin-top: 5px; }}
    .metric-active {{ border-left: 6px solid #63b8ff; }}
    .metric-silent {{ border-left: 6px solid #b8b8b8; }}
    .metric-global-active {{ border-left: 6px solid #35b36b; }}
    .metric-global-silent {{ border-left: 6px solid #d18e2f; }}
    .metric-global-segment {{ border-left: 6px solid #6a4c93; }}
    .metrics-help {{ font-size: 12px; color: #555; line-height: 1.45; margin-top: 10px; }}
    .cell-list-dialog {{ width: min(640px, 92vw); border: none; border-radius: 12px; padding: 0; box-shadow: 0 20px 50px rgba(0,0,0,0.35); }}
    .cell-list-pre {{ max-height: 52vh; overflow: auto; font-family: ui-monospace, monospace; font-size: 12px; white-space: pre-wrap; word-break: break-all; margin: 0; padding: 10px; background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; }}
    .section-title {{ font-weight: bold; margin-bottom: 6px; }}
    .segment-title {{ font-size: 18px; font-weight: bold; margin: 0 0 10px; }}
    .nav-btn {{ border: none; background: #444; color: white; border-radius: 8px; padding: 8px 12px; cursor: pointer; font-size: 18px; line-height: 1; }}
    .nav-btn:disabled {{ opacity: 0.35; cursor: default; }}
    .groups-wrap {{ display: grid; gap: 12px; }}
    .count-group {{ border: 1px solid #e4e4e4; border-radius: 10px; padding: 12px; background: #fff; }}
    .group-header {{ font-size: 18px; font-weight: 700; margin-bottom: 10px; }}
    .group-active {{ border-left: 6px solid #63b8ff; }}
    .group-silent {{ border-left: 6px solid #b8b8b8; }}
    .group-global-active {{ border-left: 6px solid #35b36b; }}
    .group-global-segment {{ border-left: 6px solid #6a4c93; }}
    .group-global-silent {{ border-left: 6px solid #d18e2f; }}
    .group-global-segment-silent {{ border-left: 6px solid #c0392b; }}
    .metric-global-segment-silent {{ border-left: 6px solid #c0392b; }}
    .chip-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .chip-grid.subtype {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .count-chip {{ border-radius: 10px; padding: 10px 8px; text-align: center; border: 1px solid #ddd; }}
    .chip-name {{ font-size: 13px; font-weight: 700; margin-bottom: 4px; }}
    .chip-value {{ font-size: 24px; font-weight: 800; line-height: 1; }}
    .chip-on {{ background: #fff2db; border-color: #efc57c; }}
    .chip-off {{ background: #eaf4ff; border-color: #94c4ff; }}
    .chip-biphasic {{ background: #f3ebff; border-color: #c7a7ff; }}
    .chip-os {{ background: #fff3de; border-color: #efc57c; }}
    .chip-ot {{ background: #fff9e6; border-color: #edd57a; }}
    .chip-ofs {{ background: #ebf5ff; border-color: #94c4ff; }}
    .chip-oft {{ background: #eefcff; border-color: #8fd5e6; }}
    .chip-oi {{ background: #f8efe4; border-color: #d9b48f; }}
    .chip-ofi {{ background: #f2f2f2; border-color: #c9c9c9; }}
    .group-note {{ font-size: 14px; color: #666; }}
    .meta-strip {{ border: 1px solid #e4e4e4; border-radius: 10px; background: #fff; padding: 10px 12px; font-size: 16px; font-weight: 600; }}
    .axis-hint {{ font-size: 13px; color: #555; margin: 0 0 12px; line-height: 1.35; }}
    .dl-btn {{ border: none; background: #1565c0; color: #fff; border-radius: 8px; padding: 8px 14px; cursor: pointer; font-size: 14px; }}
    .dl-btn:hover {{ background: #0d47a1; }}
    .details-btn {{ border: none; background: #7b4fa6; color: #fff; border-radius: 8px; padding: 8px 14px; cursor: pointer; font-size: 14px; }}
    .details-btn:hover {{ background: #5e3585; }}
    .details-dialog {{ width: min(860px, 97vw); border: none; border-radius: 14px; padding: 0; box-shadow: 0 24px 60px rgba(0,0,0,0.40); overflow: hidden; }}
    .di-wrap {{ display: flex; flex-direction: column; max-height: 90vh; }}
    .di-topbar {{ display: flex; align-items: center; gap: 10px; padding: 14px 18px; background: #1e1e2e; color: #fff; }}
    .di-mode-label {{ flex: 1; text-align: center; font-size: 14px; font-weight: 600; color: #e0e0ff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .di-nav-btn {{ border: none; background: #3d3d5c; color: #fff; border-radius: 8px; padding: 7px 14px; cursor: pointer; font-size: 14px; font-weight: 600; }}
    .di-nav-btn:hover:not(:disabled) {{ background: #5555aa; }}
    .di-nav-btn:disabled {{ opacity: 0.35; cursor: default; }}
    .di-close-btn {{ border: none; background: #c0392b; color: #fff; border-radius: 8px; padding: 7px 12px; cursor: pointer; font-size: 16px; margin-left: 6px; }}
    .di-body {{ padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 18px; }}
    .di-section {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 12px; overflow: hidden; }}
    .di-section-title {{ padding: 10px 16px; background: #f0f0f8; font-size: 13px; font-weight: 700; color: #333; letter-spacing: 0.5px; border-bottom: 1px solid #e0e0e0; }}
    .di-total {{ font-weight: 400; color: #666; font-size: 12px; }}
    .di-cell-row {{ display: flex; align-items: center; gap: 14px; padding: 14px 16px; border-bottom: 1px solid #f0f0f0; flex-wrap: wrap; }}
    .di-cell-row:last-child {{ border-bottom: none; }}
    .di-row-active {{ background: #f0f8ff; }}
    .di-row-silent {{ background: #f8f8f8; }}
    .di-row-ga {{ background: #f5f0ff; }}
    .di-row-label {{ display: flex; align-items: baseline; gap: 8px; min-width: 180px; }}
    .di-big-num {{ font-size: 32px; font-weight: 800; line-height: 1; color: #222; }}
    .di-row-name {{ font-size: 14px; font-weight: 700; color: #444; }}
    .di-row-pct {{ font-size: 13px; color: #888; }}
    .di-chip {{ border-radius: 20px; padding: 6px 12px; font-size: 13px; font-weight: 700; display: inline-flex; align-items: center; gap: 4px; }}
    .di-on {{ background: #fff2db; color: #b07800; border: 1px solid #efc57c; }}
    .di-off {{ background: #eaf4ff; color: #1a5a9a; border: 1px solid #94c4ff; }}
    .di-bip {{ background: #f3ebff; color: #6b2fa0; border: 1px solid #c7a7ff; }}
    .di-pct {{ font-weight: 400; font-size: 11px; opacity: 0.8; }}
    .di-no-stim {{ font-size: 12px; color: #bbb; font-style: italic; }}
    .di-changes {{ padding: 6px 16px 12px; display: flex; flex-direction: column; gap: 6px; }}
    .di-change-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 8px; padding: 7px 10px; border-radius: 8px; background: #fafafa; border: 1px solid #eee; }}
    .di-mini-chips {{ display: flex; gap: 5px; flex-wrap: wrap; margin-left: 4px; }}
    .di-mini-chips .di-chip {{ font-size: 11px; padding: 2px 7px; border-radius: 10px; font-weight: 700; color: white; }}
    .di-mini-chips .di-on {{ background: #d48c00; }}
    .di-mini-chips .di-off {{ background: #1460a8; }}
    .di-mini-chips .di-bip {{ background: #7020b0; }}
    .di-change-new {{ background: #f0fff4; border-color: #b8e6c4; }}
    .di-change-lost {{ background: #fff5f5; border-color: #f5c2c2; }}
    .di-change-kept {{ background: #f5f5ff; border-color: #c8c8f5; }}
    .di-change-label {{ min-width: 130px; font-size: 13px; font-weight: 600; color: #444; }}
    .di-change-val {{ font-size: 20px; font-weight: 800; min-width: 70px; color: #222; }}
    .di-change-delta {{ font-size: 13px; font-weight: 600; margin-left: auto; }}
    .delta-pos {{ color: #22863a; font-weight: 700; }}
    .delta-neg {{ color: #c0392b; font-weight: 700; }}
    .delta-zero {{ color: #888; }}
    .zoom-bar {{ display: flex; align-items: center; gap: 10px; margin: 10px 0 8px; flex-wrap: wrap; }}
    .zoom-label {{ font-size: 13px; font-weight: 600; color: #333; }}
    .zoom-slider {{ width: 220px; }}
    .zoom-value {{ font-size: 13px; color: #444; min-width: 42px; }}
    .overview-layout {{ display: grid; grid-template-columns: 1fr 220px; gap: 12px; align-items: start; }}
    .overview-scroll {{ overflow-x: auto; overflow-y: hidden; border: 1px solid #e4e4e4; border-radius: 8px; background: #fff; }}
    .legend-side {{ border: 1px solid #e4e4e4; border-radius: 8px; background: #fff; padding: 10px; max-height: 420px; overflow: auto; }}
    .legend-side h3 {{ margin: 0 0 8px; font-size: 14px; }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; margin: 5px 0; font-size: 13px; }}
    .legend-swatch {{ width: 16px; height: 12px; border-radius: 2px; border: 1px solid #999; flex: 0 0 auto; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js" crossorigin="anonymous"></script>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>{safe_title} all repetitions</h1>
      <p class="sub">Stimulus reference (top), Viterbi state bands for every repetition (below), shared time axis. Tap any colored segment for the same detail popup used in the contrast pair reports.</p>
      <div class="zoom-bar">
        <span class="zoom-label">Zoom</span>
        <input id="zoomRange" class="zoom-slider" type="range" min="1" max="8" step="0.5" value="1.5" />
        <span id="zoomValue" class="zoom-value">1.5x</span>
      </div>
      <div class="overview-layout">
        <div class="overview-scroll">
          <svg id="overviewSine" class="plot-sine" viewBox="0 0 1200 170" preserveAspectRatio="none"></svg>
          <svg id="overviewStates" class="plot-reps" viewBox="0 0 1200 120" preserveAspectRatio="xMidYMin meet"></svg>
        </div>
        <aside class="legend-side">
          <h3>State legend</h3>
          <div id="stateLegend"></div>
        </aside>
      </div>
      <div class="hint">Click or tap a state block in any row. Prev and next in the dialog move through segments in time order across all repetitions.</div>
    </div>
  </div>

  <dialog id="segmentDialog">
    <div class="modal">
      <div class="modal-top">
        <h2>Segment details</h2>
        <div class="modal-nav">
          <button id="prevBtn" class="nav-btn" type="button" aria-label="Previous segment">&#8592;</button>
          <button id="nextBtn" class="nav-btn" type="button" aria-label="Next segment">&#8594;</button>
          <button type="button" id="detailsBtn" class="details-btn">Details</button>
          <button type="button" id="downloadPopupPng" class="dl-btn">Download PNG</button>
          <button class="close-btn" onclick="document.getElementById('segmentDialog').close()">Close</button>
        </div>
      </div>
      <div id="popupCaptureBody">
        <div class="card" style="margin-top:14px;">
          <div id="detailTitle" class="segment-title"></div>
          <p class="axis-hint">
            <b>Same time axis (trial time):</b> stimulus reference is drawn in the <i>top</i> panel only;
            mean firing rate of local active cells is in the <i>bottom</i> panel. Align vertically to compare timing (not overlaid on one plot).
          </p>
          <div class="section-title">Stimulus reference (sine, scaled by contrast)</div>
          <svg id="detailSine" class="plot-sine" viewBox="0 0 1100 170" preserveAspectRatio="none"></svg>
          <div class="section-title">Mean firing rate (local active cells, Hz)</div>
          <svg id="detailTrace" class="plot-sine" viewBox="0 0 1100 240" preserveAspectRatio="none" style="height:240px"></svg>
        </div>
        <div id="summaryGrid" class="summary-grid"></div>
        <div id="metaStrip" class="meta-strip"></div>
        <div id="metricsHelp" class="metrics-help"></div>
        <div class="groups-wrap" id="fourierGroups"></div>
        <div class="groups-wrap" id="subtypeGroups"></div>
      </div>
    </div>
  </dialog>

  <dialog id="detailsDialog" class="details-dialog">
    <div id="detailsContent"></div>
  </dialog>

  <dialog id="cellListDialog" class="cell-list-dialog">
    <div class="modal">
      <div class="modal-top">
        <h2 id="cellListTitle">Cell list</h2>
        <button type="button" class="close-btn" onclick="document.getElementById('cellListDialog').close()">Close</button>
      </div>
      <p class="axis-hint">Lines show the pipeline cell index from the label map and the flat index in the padded stack when both are known.</p>
      <pre id="cellListPre" class="cell-list-pre"></pre>
    </div>
  </dialog>

  <script>
    const data = {data_json};
    let currentSegmentIndex = 0;
    const BASE_W = 1200;
    let zoomX = 1.5;

    function svgEl(name, attrs = {{}}) {{
      const el = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      return el;
    }}

    function applyZoom() {{
      const w = BASE_W * zoomX;
      const sineSvg = document.getElementById("overviewSine");
      const stateSvg = document.getElementById("overviewStates");
      sineSvg.style.width = `${{w}}px`;
      stateSvg.style.width = `${{w}}px`;
      const zv = document.getElementById("zoomValue");
      if (zv) zv.textContent = `${{zoomX.toFixed(1)}}x`;
    }}

    function drawOverview() {{
      const sineSvg = document.getElementById("overviewSine");
      const stateSvg = document.getElementById("overviewStates");
      const width = BASE_W;
      const sineHeight = 170;
      const pairLen = data.pairLenBins;
      const nReps = data.nReps;
      const bandH = 20;
      const gap = 6;
      const labelW = 72;
      const plotW = width - labelW;
      const statesH = Math.max(80, 10 + nReps * (bandH + gap));

      sineSvg.innerHTML = "";
      stateSvg.innerHTML = "";
      stateSvg.setAttribute("viewBox", `0 0 ${{width}} ${{statesH}}`);

      sineSvg.appendChild(svgEl("rect", {{x: 0, y: 0, width, height: sineHeight, fill: "white"}}));
      stateSvg.appendChild(svgEl("rect", {{x: 0, y: 0, width, height: statesH, fill: "white"}}));

      const sx0 = labelW;
      const sw = width - labelW;
      let sinePath = "";
      for (let i = 0; i < 1200; i++) {{
        const bin = (i / 1199) * pairLen;
        let amp = 0;
        for (const seg of data.overviewSegments) {{
          if (bin >= seg.startBin && bin <= seg.endBin) {{
            amp = seg.contrast / 100.0;
            break;
          }}
        }}
        const t = (bin * data.binMs) / 1000.0;
        const yv = amp * Math.sin(2 * Math.PI * data.freqHz * t);
        const xf = sx0 + (i / 1199) * sw;
        const y = 85 - yv * 60;
        sinePath += (i === 0 ? "M" : "L") + `${{xf.toFixed(2)}},${{y.toFixed(2)}}`;
      }}
      sineSvg.appendChild(svgEl("path", {{d: sinePath, fill: "none", stroke: "#111", "stroke-width": 2.8}}));
      sineSvg.appendChild(svgEl("line", {{x1: sx0, y1: 85, x2: width, y2: 85, stroke: "#999", "stroke-dasharray": "4 4"}}));
      sineSvg.appendChild(svgEl("text", {{x: labelW / 2, y: 95, "text-anchor": "middle", "font-size": 15, "font-weight": "bold", fill: "#222"}})).textContent = `${{data.freqHz}}Hz`;

      for (const seg of data.overviewSegments) {{
        const x0 = sx0 + (seg.startBin / pairLen) * sw;
        const x1 = sx0 + (seg.endBin / pairLen) * sw;
        const mid = (x0 + x1) / 2;
        let lbl = seg.condition;
        if (seg.contrast > 0) lbl = `c=${{Math.round(seg.contrast)}}%`;
        sineSvg.appendChild(svgEl("text", {{
          x: mid, y: 20, "text-anchor": "middle", "font-size": 12, "font-weight": "bold", fill: "#333"
        }})).textContent = lbl;
      }}
      sineSvg.appendChild(svgEl("line", {{x1: width, y1: 0, x2: width, y2: sineHeight, stroke: "#222", "stroke-dasharray": "4 4"}}));

      const boundaries = new Set();
      for (const seg of data.overviewSegments) {{
        boundaries.add(seg.startBin);
        boundaries.add(seg.endBin);
      }}
      boundaries.forEach((b) => {{
        if (b <= 0 || b >= pairLen) return;
        const xv = labelW + (b / pairLen) * plotW;
        sineSvg.appendChild(svgEl("line", {{x1: xv, y1: 0, x2: xv, y2: sineHeight, stroke: "#222", "stroke-dasharray": "4 4", opacity: 0.85}}));
        stateSvg.appendChild(svgEl("line", {{x1: xv, y1: 0, x2: xv, y2: statesH, stroke: "#222", "stroke-dasharray": "4 4", opacity: 0.85}}));
      }});

      for (let r = 0; r < nReps; r++) {{
        const y0 = 6 + r * (bandH + gap);
        const lab = data.repLabels[r] || `rep ${{r + 1}}`;
        stateSvg.appendChild(svgEl("text", {{
          x: 4, y: y0 + bandH * 0.72, "font-size": 12, "font-weight": "600", fill: "#333"
        }})).textContent = lab;

        const segsHere = data.segments.filter((s) => s.repIndex === r);
        for (const seg of segsHere) {{
          const x0 = labelW + (seg.pairStartBin / pairLen) * plotW;
          const x1 = labelW + (seg.pairEndBin / pairLen) * plotW;
          const rect = svgEl("rect", {{
            x: x0, y: y0, width: Math.max(1, x1 - x0), height: bandH,
            fill: seg.color, class: "state-rect", rx: 3, ry: 3
          }});
          rect.addEventListener("click", () => showSegment(seg));
          stateSvg.appendChild(rect);
          if (x1 - x0 > 28) {{
            stateSvg.appendChild(svgEl("text", {{
              x: (x0 + x1) / 2, y: y0 + bandH * 0.72, "text-anchor": "middle",
              "font-size": 11, "font-weight": "bold", fill: "#111", "pointer-events": "none"
            }})).textContent = `S${{seg.mode}}`;
          }}
        }}
      }}
    }}

    const PLOT_LEFT = 72;
    const PLOT_RIGHT = 18;

    function plotInnerWidth(fullW) {{
      return fullW - PLOT_LEFT - PLOT_RIGHT;
    }}

    function xAtBinIndex(i, fullW) {{
      const pw = plotInnerWidth(fullW);
      const n = data.trialLenBins;
      return PLOT_LEFT + (i / Math.max(n - 1, 1)) * pw;
    }}

    function xAtBinBoundary(b, fullW) {{
      const pw = plotInnerWidth(fullW);
      return PLOT_LEFT + (b / data.trialLenBins) * pw;
    }}

    function drawDetailSine(seg) {{
      const svg = document.getElementById("detailSine");
      const width = 1100;
      const height = 170;
      const midY = 85;
      const amp = 60;
      const pw = plotInnerWidth(width);
      svg.innerHTML = "";
      svg.appendChild(svgEl("rect", {{x: 0, y: 0, width, height, fill: "white"}}));
      svg.appendChild(svgEl("line", {{x1: PLOT_LEFT, y1: midY, x2: width - PLOT_RIGHT, y2: midY, stroke: "#999", "stroke-dasharray": "4 4"}}));

      const condLabel = seg.contrastPct > 0
        ? `${{seg.condition}} contrast reference (c=${{seg.contrastPct}}%)`
        : `${{seg.condition}} — no stimulus (c=0%)`;
      svg.appendChild(svgEl("text", {{x: PLOT_LEFT, y: 18, "font-size": 14, "font-weight": "bold", fill: "#222"}})).textContent = condLabel;

      const xs = [];
      const ys = [];
      for (let i = 0; i < data.trialLenBins; i++) {{
        const t = (i * data.binMs) / 1000.0;
        xs.push(xAtBinIndex(i, width));
        ys.push(seg.stimAmp * Math.sin(2 * Math.PI * data.freqHz * t));
      }}

      if (seg.contrastPct > 0) {{
        const path = xs.map((x, i) => {{
          const y = midY - ys[i] * amp;
          return `${{i === 0 ? 'M' : 'L'}}${{x.toFixed(2)}},${{y.toFixed(2)}}`;
        }}).join(" ");
        svg.appendChild(svgEl("path", {{d: path, fill: "none", stroke: "#111", "stroke-width": 2.8}}));
      }} else {{
        svg.appendChild(svgEl("line", {{x1: PLOT_LEFT, y1: midY, x2: width - PLOT_RIGHT, y2: midY, stroke: "#444", "stroke-width": 2.0}}));
        svg.appendChild(svgEl("text", {{
          x: PLOT_LEFT + pw / 2, y: midY - 14,
          "text-anchor": "middle", "font-size": 14, fill: "#888", "font-style": "italic"
        }})).textContent = "No sinusoidal stimulus — baseline (contrast = 0%)";
      }}

      const x0 = xAtBinBoundary(seg.trialStartBin, width);
      const x1 = xAtBinBoundary(seg.trialEndBin, width);
      svg.appendChild(svgEl("rect", {{x: x0, y: 0, width: Math.max(1, x1 - x0), height, fill: seg.color, opacity: 0.13}}));
      svg.appendChild(svgEl("line", {{x1: x0, y1: 0, x2: x0, y2: height, stroke: "#333", "stroke-dasharray": "5 5"}}));
      svg.appendChild(svgEl("line", {{x1: x1, y1: 0, x2: x1, y2: height, stroke: "#333", "stroke-dasharray": "5 5"}}));

      const xMaxMs = (data.trialLenBins * data.binMs);
      [0, 0.25, 0.5, 0.75, 1.0].forEach((f) => {{
        const ms = f * xMaxMs;
        const bx = PLOT_LEFT + f * pw;
        svg.appendChild(svgEl("line", {{x1: bx, y1: height - 2, x2: bx, y2: height, stroke: "#666"}}));
        svg.appendChild(svgEl("text", {{
          x: bx, y: height - 6, "text-anchor": "middle", "font-size": 11, fill: "#666"
        }})).textContent = String(Math.round(ms));
      }});
    }}

    function drawDetailTrace(seg) {{
      const svg = document.getElementById("detailTrace");
      const width = 1100;
      const height = 240;
      const top = 14;
      const bottom = 36;
      const plotW = plotInnerWidth(width);
      const plotH = height - top - bottom;
      svg.innerHTML = "";
      svg.appendChild(svgEl("rect", {{x: 0, y: 0, width, height, fill: "white"}}));
      const vals = seg.trace || [];
      const yLabel = "Mean FR (Hz/cell)";
      let yMin = vals.length ? Math.min(...vals) : 0;
      let yMax = vals.length ? Math.max(...vals) : 1;
      if (Math.abs(yMax - yMin) < 1e-12) {{
        const pad = yMax === 0 ? 1.0 : Math.max(0.1, Math.abs(yMax) * 0.1);
        yMin -= pad;
        yMax += pad;
      }}
      const yOf = (v) => top + ((yMax - v) / (yMax - yMin)) * plotH;
      const xOf = (i) => xAtBinIndex(i, width);

      svg.appendChild(svgEl("line", {{x1: PLOT_LEFT, y1: top, x2: PLOT_LEFT, y2: top + plotH, stroke: "#666"}}));
      svg.appendChild(svgEl("line", {{x1: PLOT_LEFT, y1: top + plotH, x2: PLOT_LEFT + plotW, y2: top + plotH, stroke: "#666"}}));
      svg.appendChild(svgEl("text", {{
        x: 20, y: top + plotH / 2, "text-anchor": "middle", "font-size": 13, fill: "#333",
        transform: `rotate(-90 20 ${{(top + plotH / 2).toFixed(2)}})`
      }})).textContent = yLabel;

      for (const tv of [yMax, (yMin + yMax) / 2, yMin]) {{
        const ty = yOf(tv);
        svg.appendChild(svgEl("line", {{x1: PLOT_LEFT - 4, y1: ty, x2: PLOT_LEFT + plotW, y2: ty, stroke: "#ddd"}}));
        svg.appendChild(svgEl("text", {{x: PLOT_LEFT - 8, y: ty + 4, "text-anchor": "end", "font-size": 12, fill: "#444"}})).textContent = tv.toFixed(1);
      }}

      if (vals.length) {{
        const path = vals.map((v, i) => `${{i === 0 ? 'M' : 'L'}}${{xOf(i).toFixed(2)}},${{yOf(v).toFixed(2)}}`).join(" ");
        svg.appendChild(svgEl("path", {{d: path, fill: "none", stroke: seg.color, "stroke-width": 3.2}}));
      }}

      const x0b = xAtBinBoundary(seg.trialStartBin, width);
      const x1b = xAtBinBoundary(seg.trialEndBin, width);
      svg.appendChild(svgEl("rect", {{x: x0b, y: top, width: Math.max(1, x1b - x0b), height: plotH, fill: seg.color, opacity: 0.16}}));
      svg.appendChild(svgEl("line", {{x1: x0b, y1: top, x2: x0b, y2: top + plotH, stroke: "#333", "stroke-dasharray": "5 5"}}));
      svg.appendChild(svgEl("line", {{x1: x1b, y1: top, x2: x1b, y2: top + plotH, stroke: "#333", "stroke-dasharray": "5 5"}}));

      const xMaxMs = (data.trialLenBins * data.binMs);
      [0, 0.25, 0.5, 0.75, 1.0].forEach((f) => {{
        const ms = f * xMaxMs;
        const bx = PLOT_LEFT + f * plotW;
        svg.appendChild(svgEl("line", {{x1: bx, y1: top + plotH, x2: bx, y2: top + plotH + 5, stroke: "#666"}}));
        svg.appendChild(svgEl("text", {{
          x: bx, y: top + plotH + 22, "text-anchor": "middle", "font-size": 12, fill: "#333"
        }})).textContent = String(Math.round(ms));
      }});
      svg.appendChild(svgEl("text", {{
        x: PLOT_LEFT + plotW / 2, y: height - 6, "text-anchor": "middle", "font-size": 13, fill: "#444", "font-weight": "600"
      }})).textContent = "Time in trial (ms)";
    }}

    function safeFilePart(s) {{
      return String(s).replace(/[^a-zA-Z0-9._-]+/g, "_");
    }}

    function buildPopupPngName(seg) {{
      return `all_reps_${{safeFilePart(seg.repLabel)}}_S${{seg.mode}}_${{safeFilePart(seg.condition)}}_rep${{seg.conditionRep}}_bins${{seg.trialStartBin}}-${{seg.trialEndBin}}_meanFR_perCell_Hz.png`;
    }}

    async function downloadPopupAsPng() {{
      if (typeof html2canvas === "undefined") {{
        alert("html2canvas failed to load — cannot export PNG.");
        return;
      }}
      const seg = data.segments[currentSegmentIndex];
      const el = document.getElementById("popupCaptureBody");
      const canvas = await html2canvas(el, {{ scale: 2, backgroundColor: "#ffffff", logging: false }});
      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png");
      a.download = buildPopupPngName(seg);
      a.click();
    }}

    function chipClass(name) {{
      const key = String(name).toLowerCase();
      const map = {{
        on: "count-chip chip-on",
        off: "count-chip chip-off",
        biphasic: "count-chip chip-biphasic",
        os: "count-chip chip-os",
        ot: "count-chip chip-ot",
        ofs: "count-chip chip-ofs",
        oft: "count-chip chip-oft",
      }};
      return map[key] || "count-chip";
    }}

    function groupClass(kind) {{
      if (kind === "active") return "count-group group-active";
      if (kind === "silent") return "count-group group-silent";
      if (kind === "global-active") return "count-group group-global-active";
      if (kind === "global-segment") return "count-group group-global-segment";
      if (kind === "global-silent") return "count-group group-global-silent";
      if (kind === "global-segment-silent") return "count-group group-global-segment-silent";
      return "count-group";
    }}

    function openCellList(title, lines) {{
      const t = document.getElementById("cellListTitle");
      if (t) t.textContent = title || "Cell list";
      const pre = document.getElementById("cellListPre");
      const arr = Array.isArray(lines) ? lines : [];
      pre.textContent = arr.length ? arr.join(String.fromCharCode(10)) : "(none)";
      document.getElementById("cellListDialog").showModal();
    }}

    function findPrevSegment(seg) {{
      const sameRep = data.segments.filter(s => s.repIndex === seg.repIndex && s.pairStartBin < seg.pairStartBin);
      if (!sameRep.length) return null;
      sameRep.sort((a, b) => b.pairStartBin - a.pairStartBin);
      return sameRep[0];
    }}

    function setOps(prevArr, currArr) {{
      const prevSet = new Set(prevArr);
      const currSet = new Set(currArr);
      return {{
        added: currArr.filter(x => !prevSet.has(x)),
        lost: prevArr.filter(x => !currSet.has(x)),
        common: currArr.filter(x => prevSet.has(x)),
      }};
    }}

    function fourierChipsOf(indices) {{
      if (!data.flatToFourier || !indices || indices.length === 0) return "";
      let on = 0, off = 0, bip = 0;
      for (const fi of indices) {{
        const lbl = data.flatToFourier[String(fi)];
        if (lbl === "ON") on++;
        else if (lbl === "OFF") off++;
        else if (lbl === "Biphasic") bip++;
      }}
      const n = indices.length;
      const fmt = c => n > 0 ? `${{c}} (${{(c/n*100).toFixed(0)}}%)` : "—";
      return `<span class="di-mini-chips">` +
        `<span class="di-chip di-on">ON ${{fmt(on)}}</span>` +
        `<span class="di-chip di-off">OFF ${{fmt(off)}}</span>` +
        `<span class="di-chip di-bip">Bip ${{fmt(bip)}}</span>` +
        `</span>`;
    }}

    function deltaClass(n) {{
      if (n > 0) return "delta-pos";
      if (n < 0) return "delta-neg";
      return "delta-zero";
    }}

    function getFourierGroup(seg, kind) {{
      return (seg.fourierGroups || []).find(g => g.kind === kind) || {{ items: [] }};
    }}

    function fourierInlineHtml(group, total) {{
      const items = group.items || [];
      if (!items.length) return "";
      const pct = n => total > 0 ? ((n/total)*100).toFixed(0)+"%" : "—";
      return items.map(it => {{
        const cls = it.name === "ON"  ? "di-on"
                  : it.name === "OFF" ? "di-off" : "di-bip";
        return `<span class="di-chip ${{cls}}">${{it.name}} ${{it.value}} <span class="di-pct">(${{pct(it.value)}})</span></span>`;
      }}).join("");
    }}

    function navigateDetails(delta) {{
      const newIdx = currentSegmentIndex + delta;
      if (newIdx < 0 || newIdx >= data.segments.length) return;
      currentSegmentIndex = newIdx;
      const nextSeg = data.segments[currentSegmentIndex];
      // Silently refresh main popup content
      const dt = document.getElementById("detailTitle");
      if (dt) dt.textContent = nextSeg.repLabel
        ? `${{data.title}} | ${{nextSeg.repLabel}} | ${{nextSeg.condition}} rep ${{nextSeg.conditionRep}} | State ${{nextSeg.mode}} | bins ${{nextSeg.trialStartBin}}-${{nextSeg.trialEndBin}}`
        : `${{data.title}} | ${{nextSeg.condition}} rep ${{nextSeg.conditionRep}} | State ${{nextSeg.mode}} | bins ${{nextSeg.trialStartBin}}-${{nextSeg.trialEndBin}}`;
      try {{ drawDetailSine(nextSeg); drawDetailTrace(nextSeg); }} catch(e) {{}}
      renderSummary(nextSeg);
      updateNavButtons();
      const isC0 = Math.abs(nextSeg.contrastPct) < 0.5;
      if (isC0) {{
        document.getElementById("fourierGroups").innerHTML = `<p style="color:#999;font-size:13px;padding:8px 0">No Fourier/subtype breakdown for c0 (no stimulus present).</p>`;
        document.getElementById("subtypeGroups").innerHTML = "";
      }} else {{
        renderGroups("fourierGroups", nextSeg.fourierGroups, false);
        renderGroups("subtypeGroups", nextSeg.subtypeGroups, true);
      }}
      showDetails(nextSeg);
    }}

    function showDetails(seg) {{
      const total = seg.conditionTotalCells || 1;
      const pctOf = n => ((n / total) * 100).toFixed(1) + "%";
      const isC0 = Math.abs(seg.contrastPct) < 0.5;

      // Fourier groups from payload
      const actGroup  = getFourierGroup(seg, "active");
      const silGroup  = getFourierGroup(seg, "silent");
      const gaGroup   = getFourierGroup(seg, "global-segment");

      const prevSeg = findPrevSegment(seg);
      const nextSeg = data.segments[currentSegmentIndex + 1] || null;
      const hasPrev = currentSegmentIndex > 0;
      const hasNext = currentSegmentIndex < data.segments.length - 1;

      const modeLabel = `S${{seg.mode}} | ${{seg.condition}} rep ${{seg.conditionRep}} | c${{seg.contrastPct}}% | bins ${{seg.trialStartBin}}–${{seg.trialEndBin}}`;

      let html = `
      <div class="di-wrap">
        <div class="di-topbar">
          <button class="di-nav-btn" ${{hasPrev ? "" : "disabled"}} onclick="navigateDetails(-1)">&#8592; Prev</button>
          <span class="di-mode-label">${{modeLabel}}</span>
          <button class="di-nav-btn" ${{hasNext ? "" : "disabled"}} onclick="navigateDetails(1)">Next &#8594;</button>
          <button class="di-close-btn" onclick="document.getElementById('detailsDialog').close()">&#10005;</button>
        </div>

        <div class="di-body">
          <div class="di-section">
            <div class="di-section-title">&#9632; CELL POPULATIONS &nbsp;<span class="di-total">total kept: ${{total}}</span></div>
            <div class="di-cell-row di-row-active">
              <div class="di-row-label">
                <span class="di-big-num">${{seg.localActiveCells}}</span>
                <span class="di-row-name">Active</span>
                <span class="di-row-pct">${{pctOf(seg.localActiveCells)}}</span>
              </div>
              ${{isC0 ? '<span class="di-no-stim">no stimulus — Fourier n/a</span>' : fourierInlineHtml(actGroup, seg.localActiveCells)}}
            </div>
            <div class="di-cell-row di-row-silent">
              <div class="di-row-label">
                <span class="di-big-num">${{seg.globalSegmentSilentCount}}</span>
                <span class="di-row-name">Silent</span>
                <span class="di-row-pct">${{pctOf(seg.globalSegmentSilentCount)}}</span>
              </div>
              ${{isC0 ? '<span class="di-no-stim">no stimulus — Fourier n/a</span>' : fourierInlineHtml(silGroup, seg.globalSegmentSilentCount)}}
            </div>
            <div class="di-cell-row di-row-ga">
              <div class="di-row-label">
                <span class="di-big-num">${{seg.globalActiveSegmentEveryBinCount}}</span>
                <span class="di-row-name">Always active</span>
                <span class="di-row-pct">${{pctOf(seg.globalActiveSegmentEveryBinCount)}}<br><span style="font-size:11px;color:#888">spike every bin</span></span>
              </div>
              ${{isC0 ? '<span class="di-no-stim">no stimulus — Fourier n/a</span>' : fourierInlineHtml(gaGroup, seg.globalActiveSegmentEveryBinCount)}}
            </div>
          </div>`;

      if (prevSeg) {{
        const prevCount = prevSeg.localActiveCells || 0;
        const currCount = seg.localActiveCells;
        const rawDelta  = currCount - prevCount;
        const sign      = rawDelta >= 0 ? "+" : "";
        const pctDelta  = prevCount > 0 ? `${{sign}}${{((rawDelta/prevCount)*100).toFixed(1)}}%` : "N/A";
        const arrowIcon = rawDelta > 0 ? "▲" : rawDelta < 0 ? "▼" : "■";
        const arrowCls  = rawDelta > 0 ? "delta-pos" : rawDelta < 0 ? "delta-neg" : "delta-zero";

        let changesHtml = `
          <div class="di-change-row">
            <span class="di-change-label">Active count</span>
            <span class="di-change-val">${{prevCount}} → ${{currCount}}</span>
            <span class="di-change-delta ${{arrowCls}}">${{arrowIcon}} ${{sign}}${{rawDelta}} (${{pctDelta}})</span>
          </div>`;

        if (seg.localActiveFlatIndices && prevSeg.localActiveFlatIndices) {{
          const ops = setOps(prevSeg.localActiveFlatIndices, seg.localActiveFlatIndices);
          const pOf = (n,d) => d > 0 ? ((n/d)*100).toFixed(1)+"%" : "—";
          changesHtml += `
          <div class="di-change-row di-change-new">
            <span class="di-change-label">&#10133; New cells</span>
            <span class="di-change-val">${{ops.added.length}}</span>
            <span class="di-change-delta delta-pos">${{pOf(ops.added.length, prevCount)}} of prev</span>
            ${{!isC0 ? fourierChipsOf(ops.added) : ''}}
          </div>
          <div class="di-change-row di-change-lost">
            <span class="di-change-label">&#10134; Lost cells</span>
            <span class="di-change-val">${{ops.lost.length}}</span>
            <span class="di-change-delta delta-neg">−${{pOf(ops.lost.length, prevCount)}} of prev</span>
            ${{!isC0 ? fourierChipsOf(ops.lost) : ''}}
          </div>
          <div class="di-change-row di-change-kept">
            <span class="di-change-label">&#8635; Carried over</span>
            <span class="di-change-val">${{ops.common.length}}</span>
            <span class="di-change-delta">${{pOf(ops.common.length, prevCount)}} of prev</span>
            ${{!isC0 ? fourierChipsOf(ops.common) : ''}}
          </div>`;
        }}

        html += `<div class="di-section">
          <div class="di-section-title">&#9632; VS PREVIOUS MODE &nbsp;<span class="di-total">S${{prevSeg.mode}}, ${{prevSeg.condition}} rep ${{prevSeg.conditionRep}}, bins ${{prevSeg.trialStartBin}}–${{prevSeg.trialEndBin}}</span></div>
          <div class="di-changes">${{changesHtml}}</div>
        </div>`;
      }} else {{
        html += `<div class="di-section"><div class="di-section-title">&#9632; VS PREVIOUS MODE</div>
          <p style="color:#999;font-size:14px;padding:8px 0">No previous segment in this repetition.</p></div>`;
      }}

      html += `</div></div>`;
      document.getElementById("detailsContent").innerHTML = html;
      document.getElementById("detailsDialog").showModal();
    }}

    function renderSummary(seg) {{
      const summary = document.getElementById("summaryGrid");
      summary.innerHTML = "";
      const nBins = seg.trialEndBin - seg.trialStartBin;
      const cards = [
        {{ cls: "metric-card metric-active", label: "Active", value: String(seg.localActiveCells), note: "≥1 spike in this window", list: false }},
        {{ cls: "metric-card metric-global-segment-silent", label: "Silent", value: String(seg.globalSegmentSilentCount), note: "0 spikes in ALL bins of this window", list: false }},
        {{ cls: "metric-card metric-global-segment", label: "Always active (every bin)", value: String(seg.globalActiveSegmentEveryBinCount), note: `spike in every bin (${{nBins}} bins)`, list: "segment" }},
      ];
      cards.forEach(card => {{
        const div = document.createElement("div");
        div.className = card.cls;
        div.innerHTML = `<div class="metric-label">${{card.label}}</div><div class="metric-value">${{card.value}}</div><div class="metric-note">${{card.note}}</div>`;
        if (card.list === "segment" && seg.globalActiveSegmentEveryBinCount > 0) {{
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "dl-btn";
          btn.style.marginTop = "8px";
          btn.textContent = "Show cells";
          btn.addEventListener("click", () => openCellList("Global active (this segment, every bin)", seg.globalActiveSegmentEveryBinCellLines || []));
          div.appendChild(btn);
        }}
        summary.appendChild(div);
      }});
    }}

    function renderGroups(targetId, groups, subtype=false) {{
      const target = document.getElementById(targetId);
      target.innerHTML = "";
      groups.forEach(group => {{
        const wrapper = document.createElement("div");
        wrapper.className = groupClass(group.kind);
        const header = document.createElement("div");
        header.className = "group-header";
        header.textContent = group.label;
        wrapper.appendChild(header);
        if (group.available === false) {{
          const note = document.createElement("div");
          note.className = "group-note";
          note.textContent = "Subtype labels are not available for this condition.";
          wrapper.appendChild(note);
        }} else {{
          const grid = document.createElement("div");
          grid.className = subtype ? "chip-grid subtype" : "chip-grid";
          group.items.forEach(item => {{
            const chip = document.createElement("div");
            chip.className = chipClass(item.name);
            chip.innerHTML = `<div class="chip-name">${{item.name}}</div><div class="chip-value">${{item.value}}</div>`;
            grid.appendChild(chip);
          }});
          wrapper.appendChild(grid);
        }}
        target.appendChild(wrapper);
      }});
    }}

    function updateNavButtons() {{
      document.getElementById("prevBtn").disabled = currentSegmentIndex <= 0;
      document.getElementById("nextBtn").disabled = currentSegmentIndex >= data.segments.length - 1;
    }}

    function showSegment(segOrIndex) {{
      const seg = typeof segOrIndex === "number" ? data.segments[segOrIndex] : segOrIndex;
      currentSegmentIndex = seg.index;
      document.getElementById("detailTitle").textContent =
        `${{data.title}} | ${{seg.repLabel}} | ${{seg.condition}} rep ${{seg.conditionRep}} | State ${{seg.mode}} | bins ${{seg.trialStartBin}}-${{seg.trialEndBin}}`;
      drawDetailSine(seg);
      drawDetailTrace(seg);
      renderSummary(seg);
      document.getElementById("metaStrip").textContent =
        `Bins labeled with this state (dataset): ${{seg.modeTotalBins}}`;
      const mh = document.getElementById("metricsHelp");
      if (mh) mh.textContent = data.metricsHelpText || "";
      const isC0 = Math.abs(seg.contrastPct) < 0.5;
      if (isC0) {{
        document.getElementById("fourierGroups").innerHTML = `<p style="color:#999;font-size:13px;padding:8px 0">No Fourier/subtype breakdown for c0 (no stimulus present).</p>`;
        document.getElementById("subtypeGroups").innerHTML = "";
      }} else {{
        renderGroups("fourierGroups", seg.fourierGroups, false);
        renderGroups("subtypeGroups", seg.subtypeGroups, true);
      }}
      updateNavButtons();
      document.getElementById("segmentDialog").showModal();
    }}

    function renderStateLegend() {{
      const el = document.getElementById("stateLegend");
      if (!el) return;
      el.innerHTML = "";
      (data.stateLegend || []).forEach((it) => {{
        const row = document.createElement("div");
        row.className = "legend-item";
        row.innerHTML = `<span class="legend-swatch" style="background:${{it.color}}"></span><span>State ${{it.state}}</span>`;
        el.appendChild(row);
      }});
    }}

    document.getElementById("prevBtn").addEventListener("click", () => {{
      if (currentSegmentIndex > 0) showSegment(currentSegmentIndex - 1);
    }});
    document.getElementById("nextBtn").addEventListener("click", () => {{
      if (currentSegmentIndex < data.segments.length - 1) showSegment(currentSegmentIndex + 1);
    }});
    document.addEventListener("keydown", (evt) => {{
      const dialog = document.getElementById("segmentDialog");
      if (!dialog.open) return;
      if (evt.key === "ArrowLeft" && currentSegmentIndex > 0) {{
        showSegment(currentSegmentIndex - 1);
      }}
      if (evt.key === "ArrowRight" && currentSegmentIndex < data.segments.length - 1) {{
        showSegment(currentSegmentIndex + 1);
      }}
    }});

    document.getElementById("downloadPopupPng").addEventListener("click", () => downloadPopupAsPng());
    document.getElementById("detailsBtn").addEventListener("click", () => {{
      const seg = data.segments[currentSegmentIndex];
      if (seg) showDetails(seg);
    }});

    const zr = document.getElementById("zoomRange");
    if (zr) {{
      zr.addEventListener("input", (evt) => {{
        zoomX = Number(evt.target.value || 1);
        applyZoom();
      }});
    }}
    drawOverview();
    applyZoom();
    renderStateLegend();
  </script>
</body>
</html>
"""
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


def plot_pair_report(
    out_path,
    html_out_path,
    sequence_index,
    title,
    pair_segments,
    z_reps,
    sequence_segments,
    condition_data,
    mode_stats,
    recording_global_by_mode: Dict[int, np.ndarray],
    recording_global_n_occ: Dict[int, int],
    state_to_color,
    bin_ms,
    freq_hz,
    smoothing_sigma,
    active_threshold: float,
    sequence_folder_key: str,
    pair_stem: str,
):
    pair_start = pair_segments[0].start_bin
    pair_end = pair_segments[-1].end_bin
    pair_modes = np.asarray(z_reps[sequence_index][pair_start:pair_end]).astype(int)
    pair_label = f"{pair_segments[0].condition}_{pair_segments[1].condition}"
    instance_rows = build_instance_rows(
        sequence_index=sequence_index,
        z_reps=z_reps,
        sequence_segments=sequence_segments,
        condition_data=condition_data,
        mode_stats=mode_stats,
        recording_global_by_mode=recording_global_by_mode,
        recording_global_n_occ=recording_global_n_occ,
        smoothing_sigma=smoothing_sigma,
        bin_ms=bin_ms,
        pair_segments=pair_segments,
    )
    pair_flat_to_fourier: Dict[int, str] = {}
    for cd in condition_data.values():
        pair_flat_to_fourier.update(cd.flat_to_fourier)
    active_thr_pct = int(round(float(active_threshold) * 100.0))

    n_cols = 2
    n_rows = max(1, math.ceil(len(instance_rows) / n_cols))
    fig = plt.figure(figsize=(22, 5.8 + 4.2 * n_rows))
    gs = fig.add_gridspec(
        2 + n_rows,
        n_cols,
        height_ratios=[1.2, 1.0] + [3.8] * n_rows,
        hspace=0.55,
        wspace=0.28,
    )

    ax_sine = fig.add_subplot(gs[0, :])
    draw_pair_sine(ax_sine, pair_segments, pair_end - pair_start, bin_ms, freq_hz)
    ax_sine.set_title(f"{title} | {pair_label.replace('_', ' vs ')}", fontsize=14, fontweight="bold", pad=8)

    ax_band = fig.add_subplot(gs[1, :])
    draw_pair_mode_band(
        ax=ax_band,
        pair_modes=pair_modes,
        boundary_bins=[pair_segments[0].end_bin - pair_start],
        bin_ms=bin_ms,
        state_to_color=state_to_color,
    )
    ax_band.set_title("Zoomed Viterbi states", fontsize=11, pad=6)

    time_ms = bins_to_milliseconds(np.arange(SEG_LEN_BINS), bin_ms)
    csv_rows = []
    for idx, row in enumerate(instance_rows):
        cell_gs = gs[2 + idx // n_cols, idx % n_cols].subgridspec(
            3, 1, height_ratios=[0.42, 1.0, 0.75], hspace=0.08
        )
        ax_ref = fig.add_subplot(cell_gs[0, 0])
        ax = fig.add_subplot(cell_gs[1, 0])
        ax_txt = fig.add_subplot(cell_gs[2, 0])

        trace = row["active_trace"]
        stim_amp = float(pair_segments[0].contrast if row["condition"] == pair_segments[0].condition else pair_segments[1].contrast)
        stim_trace = (stim_amp / 100.0) * np.sin(2 * np.pi * freq_hz * (time_ms / 1000.0))
        ax_ref.plot(time_ms, stim_trace, color="black", linewidth=1.6)
        ax_ref.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.5)
        ax_ref.axvline(row["trial_local_start_bin"] * bin_ms, color="#333333", linestyle="--", linewidth=1.0)
        ax_ref.axvline(row["trial_local_end_bin"] * bin_ms, color="#333333", linestyle="--", linewidth=1.0)
        ax_ref.set_xlim(0, SEG_LEN_BINS * bin_ms)
        ax_ref.set_ylim(-1.1, 1.1)
        ax_ref.set_yticks([-1, 0, 1])
        ax_ref.tick_params(axis="x", labelbottom=False)
        ax_ref.set_ylabel("Stim\n(sine)", fontsize=8)
        ax_ref.spines["top"].set_visible(False)
        ax_ref.spines["right"].set_visible(False)

        col = state_to_color.get(row["mode"], "#444444")
        ax.plot(time_ms, trace, color=col, linewidth=3.0)
        span_start = row["trial_local_start_bin"] * bin_ms
        span_end = row["trial_local_end_bin"] * bin_ms
        ax.axvspan(span_start, span_end, color=state_to_color.get(row["mode"], "#444444"), alpha=0.22)
        ax.axvline(span_start, color="#333333", linestyle="--", linewidth=1.0)
        ax.axvline(span_end, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xlim(0, SEG_LEN_BINS * bin_ms)
        ax.grid(True, alpha=0.18)
        ax.set_title(
            f"{row['condition']} rep {row['condition_rep']} | S{row['mode']} | bins {row['trial_local_start_bin']}-{row['trial_local_end_bin']}",
            fontsize=9,
            pad=5,
        )
        ax.set_ylabel("Mean FR (Hz/cell)\nlocal active", fontsize=8)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Time in condition trial (ms)")
        else:
            ax.tick_params(axis="x", labelbottom=False)

        text = "\n".join(
            [
                f"Local cells: active={row['local_active_cells']} silent={row['local_silent_cells']} total={row['condition_total_cells']}",
                (
                    f"Every-bin this segment (label-mapped): {row['global_active_segment_every_bin_count']} cells. "
                    f"Mode-wide silent (0 spikes in all state bins)={row['global_silent_cells']}; "
                    f"mode bins={row['mode_total_bins']}"
                ),
                fourier_text_for_row(row, "Active"),
                fourier_text_for_row(row, "Silent"),
                global_fourier_text(row, "Global active (this segment, every bin)"),
                global_fourier_text(row, "Global silent"),
                subtype_text_for_row(row, "Active"),
                subtype_text_for_row(row, "Silent"),
            ]
        )
        ax_txt.axis("off")
        ax_txt.text(
            0.01,
            0.98,
            text,
            transform=ax_txt.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white", edgecolor="#BBBBBB", alpha=0.96),
        )

        csv_row = {k: v for k, v in row.items() if k != "active_trace"}
        csv_rows.append(csv_row)

    total_slots = n_rows * n_cols
    for idx in range(len(instance_rows), total_slots):
        ax = fig.add_subplot(gs[2 + idx // n_cols, idx % n_cols])
        ax.axis("off")

    fig.text(
        0.5,
        0.01,
        "Local active/silent counts are computed only inside this specific mode occurrence. "
        f"'Global active (this segment, every bin)' counts label-mapped cells that spike in every bin of this window. "
        "Mode-wide silent means zero spikes across all bins labeled with that state. Top panel: stimulus sine; bottom: mean FR (Hz/cell) for local active cells; "
        f"shared time axis. Dashed vertical lines mark the mode window. Trace smoothing sigma={smoothing_sigma}.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=170)
    plt.close(fig)
    save_pair_interactive_html(
        out_path=html_out_path,
        title=title,
        pair_label=pair_label,
        pair_segments=pair_segments,
        pair_modes=pair_modes,
        instance_rows=instance_rows,
        state_to_color=state_to_color,
        bin_ms=bin_ms,
        freq_hz=freq_hz,
        active_threshold=active_threshold,
        sequence_folder_key=sequence_folder_key,
        pair_stem=pair_stem,
        flat_to_fourier=pair_flat_to_fourier,
    )

    return pd.DataFrame(csv_rows)


def build_table_rows(instance_df):
    table_rows = []
    for row in instance_df.itertuples(index=False):
        if bool(row.subtype_available):
            active_st = (
                f"OS={row.active_ON_Sustained} OT={row.active_ON_Transient} "
                f"OFS={row.active_OFF_Sustained} OFT={row.active_OFF_Transient}"
            )
            silent_st = (
                f"OS={row.silent_ON_Sustained} OT={row.silent_ON_Transient} "
                f"OFS={row.silent_OFF_Sustained} OFT={row.silent_OFF_Transient}"
            )
        else:
            active_st = "NA"
            silent_st = "NA"
        table_rows.append(
            [
                f"S{row.mode}",
                f"{row.condition} r{row.condition_rep}",
                f"{row.trial_local_start_bin}-{row.trial_local_end_bin}",
                f"{row.local_active_cells}/{row.local_silent_cells}",
                f"{row.active_fourier_ON}/{row.active_fourier_OFF}/{row.active_fourier_Biphasic}",
                f"{row.silent_fourier_ON}/{row.silent_fourier_OFF}/{row.silent_fourier_Biphasic}",
                active_st,
                silent_st,
            ]
        )
    return table_rows


def save_pair_table_png(out_path, title, pair_label, instance_df):
    columns = [
        "State",
        "Cond/Rep",
        "Mode bins",
        "Local active/silent",
        "Active ON/OFF/Biphasic",
        "Silent ON/OFF/Biphasic",
        "Active subtype",
        "Silent subtype",
    ]
    rows = build_table_rows(instance_df)
    n_rows = max(1, len(rows))
    fig_h = 1.8 + 0.36 * n_rows
    fig, ax = plt.subplots(figsize=(22, fig_h))
    ax.axis("off")
    ax.set_title(
        f"{title} | {pair_label.replace('_', ' vs ')} | summary table",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="left",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#EAEAEA")
        cell.set_edgecolor("#BBBBBB")
    fig.text(
        0.5,
        0.02,
        "Subtype table entries show NA when the condition has no usable Stage6 sustained/transient labels.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def load_bundle_analysis_context(
    base_dir: Path,
    z_reps: np.ndarray,
    segments_s: List[Tuple[float, float, float]],
    bin_ms: float,
    freq_hz: float,
    active_threshold: float,
    silent_threshold: float,
    legend_top_n: int,
) -> dict:
    base_dir = Path(base_dir)
    condition_names = [condition_name_from_contrast(seg[2]) for seg in segments_s]
    condition_names = sorted(set(condition_names), key=lambda name: (int(name[1:]) if name[1:].isdigit() else 9999, name))
    max_universe = 0
    for name in condition_names:
        cdir = resolve_condition_dir(base_dir, name, freq_hz)
        raw_path = next(cdir.glob("*_filtered_4d.npz"))
        raw = np.load(raw_path, allow_pickle=True)["array"]
        _, _, c, u = raw.shape
        max_universe = max(max_universe, int(c * u))

    condition_data = {name: load_condition_data(base_dir, name, freq_hz, target_universe_size=max_universe) for name in condition_names}
    global_valid_mask = np.any(
        np.vstack([condition_data[name].valid_flat_mask.astype(bool) for name in condition_names]),
        axis=0,
    )
    universe_size = int(global_valid_mask.shape[0])

    sequence_segments = build_sequence_segments(segments_s, z_reps.shape[0], condition_data, bin_ms)
    mode_stats = compute_mode_stats(
        z_reps=z_reps,
        sequence_segments=sequence_segments,
        condition_data=condition_data,
        active_threshold=active_threshold,
        silent_threshold=silent_threshold,
        global_valid_mask=global_valid_mask,
        universe_size=universe_size,
    )
    recording_global_by_mode, recording_global_n_occ = compute_recording_global_active_consensus(
        z_reps=z_reps,
        sequence_segments=sequence_segments,
        condition_data=condition_data,
        occurrence_fraction_threshold=active_threshold,
    )
    state_to_color, _, _ = build_state_color_map(z_reps, legend_top_n=legend_top_n)
    return {
        "condition_data": condition_data,
        "sequence_segments": sequence_segments,
        "mode_stats": mode_stats,
        "state_to_color": state_to_color,
        "recording_global_by_mode": recording_global_by_mode,
        "recording_global_n_occ": recording_global_n_occ,
    }


def save_global_mode_summary(out_dir, mode_stats):
    rows = []
    for mode, stats in sorted(mode_stats.items()):
        active_count = int(stats.active_mask.sum())
        silent_count = int(stats.silent_mask.sum())
        rows.append(
            {
                "mode": int(mode),
                "total_bins": int(stats.total_bins),
                "active_cells": active_count,
                "silent_cells": silent_count,
                "valid_cells": int(stats.valid_cell_count),
                "other_cells": int(stats.valid_cell_count - active_count - silent_count),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "global_mode_activity_summary.csv", index=False)


def generate_detailed_sequence_reports(
    bundle_path,
    z_reps,
    segments_s,
    bin_ms,
    freq_hz,
    title,
    active_threshold,
    silent_threshold,
    smoothing_sigma,
    legend_top_n,
    analysis_context: Optional[dict] = None,
):
    base_dir = Path(bundle_path).parent
    if analysis_context is None:
        analysis_context = load_bundle_analysis_context(
            base_dir,
            z_reps,
            segments_s,
            bin_ms,
            freq_hz,
            active_threshold,
            silent_threshold,
            legend_top_n,
        )
    condition_data = analysis_context["condition_data"]
    sequence_segments = analysis_context["sequence_segments"]
    mode_stats = analysis_context["mode_stats"]
    state_to_color = analysis_context["state_to_color"]

    out_dir = base_dir / "analysis" / "viterbi_bands" / "sequence_pair_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_global_mode_summary(out_dir, mode_stats)

    for sequence_index, seq_segments in enumerate(sequence_segments):
        tag = "train" if sequence_index < 10 else "test"
        seq_dir = out_dir / f"sequence_{sequence_index + 1:02d}_{tag}"
        seq_dir.mkdir(exist_ok=True)
        pair_ranges = local_pair_segments(seq_segments)
        for pair_index, (left_idx, right_idx) in enumerate(pair_ranges, start=1):
            pair_segments = [seq_segments[left_idx], seq_segments[right_idx]]
            left_name = pair_segments[0].condition
            right_name = pair_segments[1].condition
            png_path = seq_dir / f"{pair_index:02d}_{left_name}_{right_name}.png"
            html_path = seq_dir / f"{pair_index:02d}_{left_name}_{right_name}_interactive.html"
            csv_path = seq_dir / f"{pair_index:02d}_{left_name}_{right_name}_instances.csv"
            table_png_path = seq_dir / f"{pair_index:02d}_{left_name}_{right_name}_table.png"
            pair_stem = f"{pair_index:02d}_{left_name}_{right_name}"
            seq_folder_key = f"sequence_{sequence_index + 1:02d}_{tag}"
            instance_df = plot_pair_report(
                out_path=png_path,
                html_out_path=html_path,
                sequence_index=sequence_index,
                title=f"{title} | Sequence {sequence_index + 1:02d} ({tag})",
                pair_segments=pair_segments,
                z_reps=z_reps,
                sequence_segments=sequence_segments,
                condition_data=condition_data,
                mode_stats=mode_stats,
                recording_global_by_mode=analysis_context["recording_global_by_mode"],
                recording_global_n_occ=analysis_context["recording_global_n_occ"],
                state_to_color=state_to_color,
                bin_ms=bin_ms,
                freq_hz=freq_hz,
                smoothing_sigma=smoothing_sigma,
                active_threshold=active_threshold,
                sequence_folder_key=seq_folder_key,
                pair_stem=pair_stem,
            )
            instance_df.to_csv(csv_path, index=False)
            save_pair_table_png(
                out_path=table_png_path,
                title=f"{title} | Sequence {sequence_index + 1:02d} ({tag})",
                pair_label=f"{left_name}_{right_name}",
                instance_df=instance_df,
            )


def plot_viterbi_bands_reps(
    z_reps,
    bin_ms=None,
    freq_hz=2.0,
    segments_s=None,
    title="Dataset",
    rep_labels=None,
    figsize=None,
    show_legend=True,
    legend_top_n=None,
    out_path=None,
):
    Z = np.asarray(z_reps).astype(int)
    assert Z.ndim == 2, "z_reps must be 2D: (n_reps, T_rep)"
    n_reps, T = Z.shape

    if bin_ms is None:
        to_x   = lambda t: t
        xlabel = "time bin"
        x_end  = T
    else:
        to_x   = lambda t: t * (bin_ms / 1000.0)
        xlabel = "time (s)"
        x_end  = to_x(T)

    state_to_color, state_counts, sorted_states = build_state_color_map(Z, legend_top_n=legend_top_n)

    if rep_labels is None:
        rep_labels = [f"rep {i+1}" for i in range(n_reps)]

    band_h = 0.8
    gap    = 0.25

    if figsize is None:
        figsize = (18, max(3.5, 0.55 * n_reps) + 2.0)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        2, 1,
        height_ratios=[1.6, max(2.5, 0.55 * n_reps)],
        hspace=0.08,
        figure=fig,
    )
    ax_sine = fig.add_subplot(gs[0])
    ax      = fig.add_subplot(gs[1])

    if segments_s is None:
        segments_s = SEG_STRUCTURE_S
    n_pts = 4000
    t_s   = np.linspace(0, x_end, n_pts)
    sine  = np.zeros(n_pts)

    for (ta, tb, c) in segments_s:
        mask = (t_s >= ta) & (t_s <= tb)
        if c > 0:
            sine[mask] = (c / 100.0) * np.sin(2 * np.pi * freq_hz * t_s[mask])

    ax_sine.plot(t_s, sine, color='black', linewidth=2.5)
    ax_sine.set_ylim(-1.2, 1.2)
    ax_sine.set_yticks([-1, 0, 1])
    ax_sine.set_yticklabels(['-1', '0', '1'], fontsize=14, fontweight='bold')
    ax_sine.set_ylabel(f'{freq_hz:.0f}Hz', fontsize=16, fontweight='bold')
    ax_sine.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_sine.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_sine.set_xlim(0, x_end)
    ax_sine.spines['top'].set_visible(False)
    ax_sine.spines['right'].set_visible(False)
    ax_sine.spines['bottom'].set_visible(False)
    ax_sine.set_title(title, fontsize=13, fontweight='bold', pad=6)

    for (ta, tb, c) in segments_s:
        if c > 0:
            ax_sine.text((ta + tb) / 2, 1.05, f"c={int(c)}%",
                         ha='center', va='bottom', fontsize=9, fontweight='bold',
                         color='#333333')

    boundaries = sorted({t for (ta, tb, _) in segments_s for t in (ta, tb)
                         if 0 < t < x_end})
    for t in boundaries:
        ax_sine.axvline(t, color='#222222', linestyle='--', linewidth=0.8, alpha=0.9)
        ax.axvline(t,      color='#222222', linestyle='--', linewidth=0.8, alpha=0.9)

    for r in range(n_reps):
        z    = Z[r]
        segs = segments_from_states(z)
        y0   = (n_reps - 1 - r) * (band_h + gap)
        for (a, b, s) in segs:
            xa, xb = to_x(a), to_x(b)
            ax.add_patch(Rectangle(
                (xa, y0), xb - xa, band_h,
                linewidth=0, facecolor=state_to_color.get(int(s), "#BBBBBB"),
            ))

    ax.set_xlim(0, x_end)
    ax.set_ylim(-gap, n_reps * (band_h + gap))
    ax.set_xlabel(xlabel)

    yticks = [(n_reps - 1 - r) * (band_h + gap) + band_h / 2 for r in range(n_reps)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(rep_labels)
    ax.grid(True, axis='x', alpha=0.3)

    if show_legend:
        handles  = [Patch(facecolor=state_to_color[s], label=f"State {s}")
                    for s in sorted_states]
        max_rows = 16
        ncol     = max(1, math.ceil(len(handles) / max_rows))
        fig.subplots_adjust(right=0.78)
        ax.legend(
            handles=handles, ncol=ncol,
            bbox_to_anchor=(1.02, 1), loc='upper left',
            borderaxespad=0., columnspacing=1.0,
            handlelength=1.2, handletextpad=0.4,
            fontsize=9, frameon=True,
        )

    if out_path is None:
        out_path = "viterbi_bands_all_reps.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="Path to *_bundle (1).npz")
    parser.add_argument("--legend_top_n", type=int, default=30)
    parser.add_argument("--active_threshold", type=float, default=0.35)
    parser.add_argument("--silent_threshold", type=float, default=0.05)
    parser.add_argument("--trace_sigma", type=float, default=1.0)
    parser.add_argument("--skip_detailed_reports", action="store_true")
    parser.add_argument(
        "--all_reps_only",
        action="store_true",
        help="Only write all_reps_interactive.html (skip summary PNG and pair/sequence reports).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Override output directory (default: <bundle_parent>/analysis/viterbi_bands).",
    )
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Override data directory for Filtered_Data/clustering (default: bundle parent).",
    )
    args = parser.parse_args()

    bundle_path = args.bundle
    p = Path(bundle_path)

    low = str(p).lower()
    if "4hz" in low or "f4hz" in low:
        freq_hz = 4.0
    else:
        freq_hz = FREQ_HZ_DEFAULT

    bin_ms = BIN_MS_DEFAULT
    rep_len_bins = REP_LEN_BINS

    data = np.load(bundle_path, allow_pickle=True)
    modes_train = np.asarray(data["modes_train"])
    modes_test = np.asarray(data["modes_test"])

    if modes_train.shape[0] % rep_len_bins != 0 or modes_test.shape[0] % rep_len_bins != 0:
        raise ValueError(
            f"Expected modes_{'train'}/{'test'} lengths divisible by {rep_len_bins}, "
            f"got {modes_train.shape[0]} and {modes_test.shape[0]}"
        )

    n_train = modes_train.shape[0] // rep_len_bins
    n_test = modes_test.shape[0] // rep_len_bins

    z_reps_train = modes_train.reshape(n_train, rep_len_bins)
    z_reps_test = modes_test.reshape(n_test, rep_len_bins)
    z_reps = np.concatenate((z_reps_train, z_reps_test), axis=0)

    rep_labels = [f"rep {i+1}" for i in range(len(z_reps))]
    title = parse_title_from_bundle(bundle_path)

    if "contrast" in data and np.asarray(data["contrast"]).shape[0] == modes_train.shape[0]:
        contrast = np.asarray(data["contrast"])
        segments_s = infer_segments_from_contrast(contrast, rep_len_bins, bin_ms)
    else:
        segments_s = SEG_STRUCTURE_S

    out_dir = Path(args.out_dir) if args.out_dir else (p.parent / "analysis" / "viterbi_bands")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "viterbi_bands_all_reps.png"

    if not args.all_reps_only:
        plot_viterbi_bands_reps(
            z_reps,
            bin_ms=bin_ms,
            freq_hz=freq_hz,
            segments_s=segments_s,
            title=title,
            rep_labels=rep_labels,
            legend_top_n=args.legend_top_n,
            out_path=str(out_path),
        )
    data_dir = Path(args.base_dir) if args.base_dir else p.parent
    analysis_ctx = load_bundle_analysis_context(
        data_dir,
        z_reps,
        segments_s,
        bin_ms,
        freq_hz,
        args.active_threshold,
        args.silent_threshold,
        args.legend_top_n,
    )
    all_reps_payload = build_all_reps_interactive_payload(
        title=title,
        z_reps=z_reps,
        rep_labels=rep_labels,
        segments_s=segments_s,
        sequence_segments=analysis_ctx["sequence_segments"],
        condition_data=analysis_ctx["condition_data"],
        mode_stats=analysis_ctx["mode_stats"],
        recording_global_by_mode=analysis_ctx["recording_global_by_mode"],
        recording_global_n_occ=analysis_ctx["recording_global_n_occ"],
        state_to_color=analysis_ctx["state_to_color"],
        bin_ms=bin_ms,
        freq_hz=freq_hz,
        active_threshold=args.active_threshold,
        smoothing_sigma=args.trace_sigma,
    )
    save_all_reps_interactive_html(out_dir, title, all_reps_payload)

    if not args.skip_detailed_reports and not args.all_reps_only:
        generate_detailed_sequence_reports(
            bundle_path=bundle_path,
            z_reps=z_reps,
            segments_s=segments_s,
            bin_ms=bin_ms,
            freq_hz=freq_hz,
            title=title,
            active_threshold=args.active_threshold,
            silent_threshold=args.silent_threshold,
            smoothing_sigma=args.trace_sigma,
            legend_top_n=args.legend_top_n,
            analysis_context=analysis_ctx,
        )


if __name__ == "__main__":
    main()
