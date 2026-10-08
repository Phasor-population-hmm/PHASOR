#!/usr/bin/env python3
"""
Generate per-contrast label index map CSVs: txt_index, txt_channel, txt_unit,
txt_flat_index, npz_channel_idx, npz_unit_idx, flat_index, kept_index, cluster_label.

Uses run directory with metadata (channels, units), and per-contrast clustering
folders (e.g. clustering/filtered/c70_f2Hz) containing:
  - *_filtered_4d.npz (to compute kept mask)
  - Stage4_Visualization/cluster_labels.npy
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def generate_label_index_map_csv(
    run_dir: Path,
    contrast_folders: list[str],
    output_subdir: str = "label_index_maps",
) -> list[Path]:
    """
    run_dir: e.g. /path/to/wt8 (2Hz)/Run1
    contrast_folders: e.g. ["c0_f2Hz", "c50_f2Hz", "c60_f2Hz", "c70_f2Hz", "c80_f2Hz", "c90_f2Hz"]

    Output matches reference format: one row per *present* cell (any spike in main 4d run),
    ordered by flat_index. kept_index and cluster_label are set only for cells kept in that contrast.
    """
    run_dir = Path(run_dir)
    out_dir = run_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(run_dir.glob("*_metadata.npz"))
    if not meta_files:
        meta_files = sorted(run_dir.glob("Full_Recording/*_metadata.npz"))
    if not meta_files:
        meta_files = sorted(run_dir.glob("Filtered_Data/**/*_filtered_metadata.npz"))
    if not meta_files:
        meta_files = sorted(run_dir.glob("Filtered Data/**/*_filtered_metadata.npz"))
    if not meta_files:
        meta_files = sorted(run_dir.glob("clustering/filtered/**/*_filtered_metadata.npz"))

    main_4d_list = list(run_dir.glob("*_4d.npz"))
    if not main_4d_list:
        main_4d_list = list(run_dir.glob("Full_Recording/*_4d.npz"))
    if not main_4d_list:
        main_4d_list = list(run_dir.glob("**/*_4d.npz"))
    if not main_4d_list:
        raise FileNotFoundError(f"No *_4d.npz found under {run_dir} (need main run 4d for present cells)")
    main_array = np.load(main_4d_list[0])["array"]
    _, _, mc, mu = main_array.shape

    if meta_files:
        meta = np.load(meta_files[0], allow_pickle=True)
        channels = np.atleast_1d(meta["channels"]).tolist()
        units = np.atleast_1d(meta["units"]).tolist()
    else:
        channels = list(range(1, mc + 1))
        units = list(range(1, mu + 1))

    n_channels = len(channels)
    n_units = len(units)
    n_cells = n_channels * n_units

    if (mc, mu) != (n_channels, n_units):
        pass
    present_mask = (main_array.sum(axis=(0, 1)) > 0).reshape(-1)
    present_flat_indices = np.nonzero(present_mask)[0].tolist()

    saved = []
    for contrast in contrast_folders:
        cond_dir = run_dir / "clustering" / "filtered" / contrast
        if not cond_dir.exists():
            short = contrast.replace("_f2Hz", "").replace("_f4Hz", "")
            cond_num = "".join(ch for ch in short if ch.isdigit())
            variants = [short]
            if cond_num:
                variants.extend(
                    [f"C-{cond_num}", f"c-{cond_num}", f"C{cond_num}", f"c{cond_num}"]
                )
            roots = [
                run_dir / "Filtered_Data",
                run_dir / "Filtered Data",
                run_dir / "Filtered data",
            ]
            cond_dir = None
            for root in roots:
                if not root.exists():
                    continue
                for v in variants:
                    cand = root / v
                    if cand.exists():
                        cond_dir = cand
                        break
                if cond_dir is not None:
                    break
        if cond_dir is None or not cond_dir.exists():
            continue

        npz_list = list(cond_dir.glob("*_filtered_4d.npz"))
        if not npz_list:
            continue
        spike_array = np.load(npz_list[0])["array"]
        _, _, nc, nu = spike_array.shape
        if (nc, nu) != (n_channels, n_units):
            pass

        nonzero_mask_cu = spike_array.sum(axis=(0, 1)) > 0
        mask_flat = nonzero_mask_cu.reshape(n_cells)

        stage4 = cond_dir / "Stage4_Visualization"
        labels_path = stage4 / "cluster_labels.npy"
        if not labels_path.exists():
            continue
        labels = np.load(labels_path)
        n_kept = int(mask_flat.sum())
        if len(labels) != n_kept:
            pass

        rows = []
        kept_counter = 0
        for txt_index, flat_index in enumerate(present_flat_indices):
            npz_channel_idx = flat_index // n_units
            npz_unit_idx = flat_index % n_units
            txt_channel = int(channels[npz_channel_idx])
            txt_unit = int(units[npz_unit_idx])
            kept = bool(mask_flat[flat_index])
            if kept:
                kept_index = kept_counter
                cluster_label = int(labels[kept_counter])
                kept_counter += 1
            else:
                kept_index = ""
                cluster_label = ""
            rows.append({
                "txt_index": txt_index,
                "txt_channel": txt_channel,
                "txt_unit": txt_unit,
                "txt_flat_index": flat_index,
                "npz_channel_idx": npz_channel_idx,
                "npz_unit_idx": npz_unit_idx,
                "flat_index": flat_index,
                "kept_index": kept_index,
                "cluster_label": cluster_label,
            })

        df = pd.DataFrame(rows)
        contrast_base = contrast.replace("_f2Hz", "").replace("_f4Hz", "")
        out_file = out_dir / f"{contrast_base}_txtcells_label_index_map.csv"
        df.to_csv(out_file, index=False)
        saved.append(out_file)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate label index map CSVs for all contrasts")
    parser.add_argument("run_dir", type=str, help="Run directory (e.g. .../wt8 (2Hz)/Run1)")
    parser.add_argument(
        "--contrasts",
        nargs="+",
        default=["c0_f2Hz", "c50_f2Hz", "c60_f2Hz", "c70_f2Hz", "c80_f2Hz", "c90_f2Hz"],
        help="Contrast folder names under clustering/filtered or Filtered_Data",
    )
    parser.add_argument("--out-dir-name", default="label_index_maps", help="Output subdir name")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    saved = generate_label_index_map_csv(run_dir, args.contrasts, args.out_dir_name)


if __name__ == "__main__":
    main()
