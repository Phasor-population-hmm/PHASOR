#!/usr/bin/env python3
"""Classify Viterbi modes and write one workbook into PHASOR_OUTPUT_ROOT."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import os as _os
_OUT_ROOT = Path(_os.environ.get("PHASOR_OUTPUT_ROOT",
                 str(Path(__file__).resolve().parent.parent / "phasor_output")))
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


PIPE_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PIPE_ROOT / "05_viterbi" / "generate_contrast_progression_png.py"
VB_PATH = PIPE_ROOT / "05_viterbi" / "viterbi_bands_wt22.py"

BIN_MS = 10
REP_LEN_BINS = 3000
FREQ_HZ = 2.0
CONTRASTS = [50, 60, 70, 80, 90]
DOMINANCE_MARGIN = 0.05
CLASSES = ["ON-dominated", "OFF-dominated", "Biphasic-dominated", "Balanced"]

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vb = load_module(VB_PATH, "phasor_viterbi_bands")
registry = load_module(REGISTRY_PATH, "phasor_dataset_registry")


def load_z_reps(bundle_path: Path) -> np.ndarray:
    data = np.load(bundle_path, allow_pickle=True)
    if "modes_train" in data and "modes_test" in data:
        flat = np.concatenate(
            [
                np.asarray(data["modes_train"], dtype=int).reshape(-1),
                np.asarray(data["modes_test"], dtype=int).reshape(-1),
            ]
        )
    elif "modes" in data:
        flat = np.asarray(data["modes"], dtype=int).reshape(-1)
    else:
        raise KeyError(f"{bundle_path} has no modes_train/modes_test or modes")
    if flat.size % REP_LEN_BINS != 0:
        raise ValueError(
            f"{bundle_path}: {flat.size} mode bins are not divisible by {REP_LEN_BINS}"
        )
    return flat.reshape(-1, REP_LEN_BINS)


def segment_bounds() -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for start_s, end_s, contrast in vb.SEG_STRUCTURE_S:
        c = int(round(float(contrast)))
        out.setdefault(c, []).append(
            (
                int(round(start_s * 1000 / BIN_MS)),
                int(round(end_s * 1000 / BIN_MS)),
            )
        )
    return out


SEGMENTS = segment_bounds()


def background_mode(z_reps: np.ndarray) -> tuple[int, dict[int, int]]:
    counts: dict[int, int] = {}
    for rep in z_reps:
        for start, end in SEGMENTS[0]:
            states, state_counts = np.unique(rep[start:end], return_counts=True)
            for state, count in zip(states, state_counts):
                counts[int(state)] = counts.get(int(state), 0) + int(count)
    if not counts:
        raise ValueError("No c0 mode bins found")
    mode = sorted(counts, key=lambda state: (-counts[state], state))[0]
    return int(mode), counts


def classify_mode(on_p: float, off_p: float, bip_p: float) -> str:
    """Highest subtype wins if it beats both others by ≥ DOMINANCE_MARGIN."""
    values = {
        "ON-dominated": on_p,
        "OFF-dominated": off_p,
        "Biphasic-dominated": bip_p,
    }
    top_label = max(values, key=values.get)
    top_value = values[top_label]
    others = [v for key, v in values.items() if key != top_label]
    if all(top_value - other >= DOMINANCE_MARGIN - 1e-12 for other in others):
        return top_label
    return "Balanced"


def active_cells_for_mode(
    z_reps: np.ndarray,
    condition_data: Any,
    contrast: int,
    mode: int,
) -> tuple[np.ndarray, int, int]:
    segments = SEGMENTS[contrast]
    if len(segments) != 1:
        raise ValueError(f"Expected one c{contrast} segment per repetition")
    start, end = segments[0]
    n_reps = z_reps.shape[0]
    if condition_data.raw_flat.shape[0] < n_reps:
        raise ValueError(
            f"c{contrast}: {condition_data.raw_flat.shape[0]} raw occurrences "
            f"for {n_reps} Viterbi repetitions"
        )

    active = np.zeros(condition_data.universe_size, dtype=bool)
    total_bins = 0
    occurrence_count = 0
    for rep_idx in range(n_reps):
        local_mode_mask = z_reps[rep_idx, start:end] == mode
        n_bins = int(local_mode_mask.sum())
        if n_bins == 0:
            continue
        total_bins += n_bins
        occurrence_count += 1
        raw = condition_data.raw_flat[rep_idx]
        if raw.shape[0] != end - start:
            raise ValueError(
                f"c{contrast}: raw segment has {raw.shape[0]} bins, expected {end - start}"
            )
        active |= np.any(raw[local_mode_mask] > 0, axis=0)
    active &= condition_data.kept_flat_mask
    return active, total_bins, occurrence_count


def analyze_dataset(ds: dict[str, Any], cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = ds["name"]
    display = registry.ds_display_name(name)
    bundle = Path(ds["bundle"])
    run_dir = registry.ds_run_dir(ds)
    if not bundle.exists():
        raise FileNotFoundError(f"Missing bundle: {bundle}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    z_reps = load_z_reps(bundle)
    bg_mode, bg_counts = background_mode(z_reps)
    details: list[dict[str, Any]] = []

    for contrast in CONTRASTS:
        condition_name = f"c{contrast}"
        cd = vb.load_condition_data(run_dir, condition_name, FREQ_HZ)
        start, end = SEGMENTS[contrast][0]
        present_modes = sorted(int(v) for v in np.unique(z_reps[:, start:end]))

        for mode in present_modes:
            active_mask, n_bins, n_occ = active_cells_for_mode(
                z_reps, cd, contrast, mode
            )
            active_indices = np.flatnonzero(active_mask)
            subtype_counts = {"ON": 0, "OFF": 0, "Biphasic": 0}
            for flat_idx in active_indices:
                label = cd.flat_to_fourier.get(int(flat_idx))
                if label in subtype_counts:
                    subtype_counts[label] += 1

            classified = int(sum(subtype_counts.values()))
            active_count = int(active_mask.sum())
            unclassified = max(0, active_count - classified)
            if classified > 0:
                on_p = subtype_counts["ON"] / classified
                off_p = subtype_counts["OFF"] / classified
                bip_p = subtype_counts["Biphasic"] / classified
                classification = classify_mode(on_p, off_p, bip_p)
            else:
                on_p = off_p = bip_p = np.nan
                classification = "No classified cells"

            is_background = mode == bg_mode
            included = bool(not is_background and classified > 0 and n_bins > 0)
            exclusion_reason = ""
            if is_background:
                exclusion_reason = "Most frequent state during c0"
            elif classified <= 0:
                exclusion_reason = "No classified active cells"
            elif n_bins <= 0:
                exclusion_reason = "Mode absent at contrast"

            details.append(
                {
                    "dataset": name,
                    "dataset_display": display,
                    "cohort": cohort,
                    "contrast": contrast,
                    "mode": mode,
                    "background_mode": bg_mode,
                    "background_c0_bins": bg_counts.get(mode, 0),
                    "is_background": is_background,
                    "included_in_summary": included,
                    "exclusion_reason": exclusion_reason,
                    "classification": classification,
                    "mode_time_bins": n_bins,
                    "mode_duration_ms": n_bins * BIN_MS,
                    "repetitions_with_mode": n_occ,
                    "active_kept_cells": active_count,
                    "ON_count": subtype_counts["ON"],
                    "OFF_count": subtype_counts["OFF"],
                    "Biphasic_count": subtype_counts["Biphasic"],
                    "Unclassified_active_count": unclassified,
                    "Total_classified_active": classified,
                    "ON_proportion": on_p,
                    "OFF_proportion": off_p,
                    "Biphasic_proportion": bip_p,
                    "label_coverage": (
                        classified / active_count if active_count > 0 else np.nan
                    ),
                    "bundle_path": str(bundle),
                    "run_dir": str(run_dir),
                }
            )

    detail_df = pd.DataFrame(details)
    summary_rows: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        subset = detail_df[
            (detail_df["contrast"] == contrast)
            & (detail_df["included_in_summary"])
        ]
        total = int(len(subset))
        row: dict[str, Any] = {
            "dataset": name,
            "dataset_display": display,
            "cohort": cohort,
            "contrast": contrast,
            "background_mode": bg_mode,
            "total_nonbackground_modes": total,
            "excluded_background_rows": int(
                (
                    (detail_df["contrast"] == contrast)
                    & detail_df["is_background"]
                ).sum()
            ),
            "excluded_unclassifiable_modes": int(
                (
                    (detail_df["contrast"] == contrast)
                    & (detail_df["classification"] == "No classified cells")
                    & (~detail_df["is_background"])
                ).sum()
            ),
        }
        for class_name in CLASSES:
            key = class_name.replace("-", "_").replace(" ", "_")
            count = int((subset["classification"] == class_name).sum())
            row[f"{key}_count"] = count
            row[f"{key}_proportion"] = count / total if total else np.nan
        summary_rows.append(row)
    return detail_df, pd.DataFrame(summary_rows)


METHODOLOGY_ROWS = [
    ("Purpose", "Classify each stimulus-responsive Viterbi mode by its unique active-cell Stage5 composition."),
    ("Contrasts", "c50, c60, c70, c80, and c90. c0 is used only to identify the background state."),
    ("Background exclusion", "For each dataset, the HMM state with the greatest total number of bins across all c0 segments and repetitions is the background mode. It remains visible in Mode_Details but is excluded from counts, proportions, and plots."),
    ("Active cell definition", "A kept cell is active for a contrast-mode pair when it fires at least one spike in any bin assigned to that mode across all repetitions of that contrast. Each cell is counted once."),
    ("Subtype labels", "ON, OFF, and Biphasic labels come from the contrast-specific Stage5 cell_fourier_results.csv and are joined through label_index_maps."),
    ("Subtype denominator", "ON/OFF/Biphasic proportions within a mode use Total_classified_active = ON + OFF + Biphasic. Unclassified active cells are reported but excluded from this denominator."),
    ("ON-dominated", "ON has the highest classified-cell count/proportion and exceeds both OFF and Biphasic by at least 5 percentage points."),
    ("OFF-dominated", "OFF has the highest classified-cell count/proportion and exceeds both ON and Biphasic by at least 5 percentage points."),
    ("Biphasic-dominated", "Biphasic has the highest classified-cell count/proportion and exceeds both ON and OFF by at least 5 percentage points."),
    ("Balanced", "The highest subtype does not exceed both others by at least 5 percentage points."),
    ("Unclassifiable", "A non-background mode with zero classified active cells is reported in Mode_Details but excluded from summary denominators."),
    ("Mode proportion", "For each dataset and contrast: class count divided by all classifiable non-background modes, including Balanced modes in the denominator."),
    ("Cross-dataset statistics", "Dataset recordings are the independent observations. Cohort lines show mean ± SEM of dataset-level mode proportions. Box plots show dataset-level distributions."),
    ("State comparability", "Raw HMM mode IDs are dataset-specific and are never averaged or equated across recordings."),
    ("WT cohort", "All registered WT baseline recordings except wt4_run1; bin-size variants and no-projection baseline are excluded."),
    ("AP4 cohort", "Only 2.5 µM AP4, n=2: wt8_ap4_run1 and wt10_ap4_run1 (wt9_ap4_run1 excluded)."),
    ("Washout cohort", "Matched washouts for included AP4 animals, n=2: wt8_wash_run1 and wt10_wash_run1 (wt9 wash and extra wt8 wash run excluded)."),
    ("RD1 cohort", "rd1_1, rd1_2, rd1_3, and rd1_1_post; included in Excel only (not in box plots)."),
]


def methodology_df() -> pd.DataFrame:
    return pd.DataFrame(METHODOLOGY_ROWS, columns=["Item", "Definition"])


def format_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for column_cells in ws.columns:
            letter = get_column_letter(column_cells[0].column)
            max_len = 0
            for cell in column_cells[:200]:
                value = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(value))
            ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 48)
        if ws.title == "Methodology":
            ws.column_dimensions["A"].width = 28
            ws.column_dimensions["B"].width = 110
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)


def write_excel(
    path: Path,
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    registry_df: pd.DataFrame | None = None,
    cohort_df: pd.DataFrame | None = None,
    errors_df: pd.DataFrame | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        methodology_df().to_excel(writer, sheet_name="Methodology", index=False)
        detail_df.to_excel(writer, sheet_name="Mode_Details", index=False)
        summary_df.to_excel(writer, sheet_name="Dataset_Contrast_Summary", index=False)
        if cohort_df is not None:
            cohort_df.to_excel(writer, sheet_name="Cohort_Summary", index=False)
        if registry_df is not None:
            registry_df.to_excel(writer, sheet_name="Dataset_Registry", index=False)
        if errors_df is not None and not errors_df.empty:
            errors_df.to_excel(writer, sheet_name="Errors", index=False)
    format_workbook(path)


def main() -> None:
    """Write one workbook into the analysis folder for the selected recording."""
    out = _OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    datasets = list(registry.DATASETS)
    if not datasets:
        raise SystemExit("No dataset was selected.")
    for ds in datasets:
        name = ds["name"]
        print(f"\n===== {name} =====", flush=True)
        detail_df, summary_df = analyze_dataset(ds, "")
        path = out / f"{name}_dominant_modes.xlsx"
        write_excel(path, detail_df, summary_df)
        print(f"saved {path.name}", flush=True)


if __name__ == "__main__":
    main()
