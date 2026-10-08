#!/usr/bin/env python3
"""
Create filtered NPZ files for specific stimulus conditions


Usage:
python create_filtered_npz.py --contrast 0 --frequency 2 --output_dir Filtered_Data/
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class StimulusFilterProcessor:
    """Handles stimulus labeling, filtering, and averaging for NPZ creation"""

    def __init__(self, stims_json_path: str):
        """Initialize with stimulus configuration"""
        with open(stims_json_path, 'r') as f:
            self.stims_data = json.load(f)

        self.original_stimuli = self.stims_data['stimuli']
        self._create_stimulus_labels()

    def _create_stimulus_labels(self):
        """Create stimulus labels following the filtering logic from Processor.py"""
        self.stimulus_df_records = []
        filtered_index = 0

        for original_idx, stim in enumerate(self.original_stimuli):
            meta_comment = stim.get('meta', {}).get('comment', '').lower()
            direct_comment = stim.get('comment', '').lower()
            stim_type = stim.get('stimType', '').lower()

            if (meta_comment == 'rest' or direct_comment == 'rest' or
                meta_comment == 'final black' or direct_comment == 'final black' or
                meta_comment == 'initial black' or direct_comment == 'initial black'):
                continue

            record = {
                "filtered_index": filtered_index,
                "original_index": original_idx,
                "stimType": stim.get("stimType"),
                "contrast": stim.get("c"),
                "frequency": stim.get("hz"),
                "duration_ms": stim.get("durationMs"),
                "meta_comment": meta_comment,
                "direct_comment": direct_comment
            }

            self.stimulus_df_records.append(record)
            filtered_index += 1

        self.stimulus_df = pd.DataFrame(self.stimulus_df_records)

    def get_condition_indices(self, contrast: Optional[float] = None,
                            frequency: Optional[float] = None,
                            stim_type: Optional[str] = None) -> List[int]:
        """Get filtered indices for specific stimulus conditions"""

        mask = np.ones(len(self.stimulus_df), dtype=bool)

        if contrast is not None:
            mask &= (self.stimulus_df['contrast'] == contrast)

        if frequency is not None:
            mask &= (self.stimulus_df['frequency'] == frequency)

        if stim_type is not None:
            mask &= (self.stimulus_df['stimType'].str.lower() == stim_type.lower())

        selected_indices = self.stimulus_df[mask]['filtered_index'].tolist()


        if len(selected_indices) > 0:
            pass

        return selected_indices

    def filter_data(self, npz_file: str, selected_indices: List[int]) -> np.ndarray:
        """
        Load NPZ file and filter to selected stimuli( Not Averaging )
        """

        data = np.load(npz_file)
        spike_array = data['array']


        if len(selected_indices) == 0:
            raise ValueError("No stimuli selected!")

        max_idx = max(selected_indices)
        if max_idx >= spike_array.shape[0]:
            raise ValueError(f"Index {max_idx} exceeds data size {spike_array.shape[0]}")

        filtered_data = spike_array[selected_indices]

        return filtered_data


class FilteredNPZCreator:
    """Creates filtered NPZ files for specific stimulus conditions"""

    def __init__(self, data_dir: str, stims_json_path: str, output_dir: str):
        self.data_dir = data_dir
        self.stims_json_path = stims_json_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.filter_processor = StimulusFilterProcessor(stims_json_path)

    def write_contrast_folders(self, frequency: Optional[float] = None,
                               stim_type: Optional[str] = "FFSine") -> List[str]:
        """Write one folder per contrast. A missing contrast does not mix every condition together."""
        df = self.filter_processor.stimulus_df
        stim_mask = df["stimType"].str.lower() == str(stim_type).lower()
        scoped = df.loc[stim_mask]
        if scoped.empty:
            raise ValueError(f"No {stim_type} stimuli found in {self.stims_json_path}")

        if frequency is None:
            present = sorted({float(v) for v in scoped["frequency"].dropna().tolist()})
            frequencies = [2.0] if 2.0 in present else present
        else:
            frequencies = [float(frequency)]

        written = []
        single_frequency = len(frequencies) == 1
        for freq in frequencies:
            contrasts = sorted({
                float(v) for v in scoped.loc[scoped["frequency"] == freq, "contrast"].dropna().tolist()
            })
            if not contrasts:
                continue
            for contrast in contrasts:
                whole = float(contrast).is_integer()
                contrast_label = f"c{int(contrast)}" if whole else f"c{contrast}"
                freq_label = f"f{int(freq)}Hz" if float(freq).is_integer() else f"f{freq}Hz"
                folder = contrast_label if single_frequency else f"{contrast_label}_{freq_label}"
                sub_dir = os.path.join(self.output_dir, folder)
                os.makedirs(sub_dir, exist_ok=True)
                writer = FilteredNPZCreator(self.data_dir, self.stims_json_path, sub_dir)
                writer.create_filtered_npz_files(
                    contrast=contrast, frequency=freq, stim_type=stim_type
                )
                n_stim = len(writer.filter_processor.get_condition_indices(
                    contrast=contrast, frequency=freq, stim_type=stim_type
                ))
                print(f"Wrote {folder} ({n_stim} stimuli) -> {sub_dir}")
                written.append(sub_dir)
        if not written:
            raise ValueError("No stimuli matched the requested contrast and frequency")
        for path in Path(self.output_dir).glob("*"):
            if path.is_file() and ("cany" in path.name.lower() or "fany" in path.name.lower()):
                path.unlink()
                print(f"Removed combined file {path.name}")
        return written

    def create_filtered_npz_files(self, contrast: Optional[float] = None,
                                frequency: Optional[float] = None,
                                stim_type: Optional[str] = "FFSine"):
        """
        Create filtered NPZ files for the specified stimulus conditions

        Creates NPZ files with shape (1, time, channels, units) where the single stimulus
        represents the averaged response to the specified condition.
        """

        selected_indices = self.filter_processor.get_condition_indices(
            contrast=contrast, frequency=frequency, stim_type=stim_type
        )

        if len(selected_indices) == 0:
            raise ValueError("No stimuli match the specified criteria!")

        npz_files = list(Path(self.data_dir).glob("*_4d.npz"))

        for npz_file in npz_files:

            filtered_data = self.filter_processor.filter_data(
                str(npz_file), selected_indices
            )

            filtered_array = filtered_data

            metadata = {
                'shape': filtered_array.shape,
                'time_bin_ms': 10,
                'max_stimulus_duration_ms': 3000,
                'max_time_bins': filtered_array.shape[1],
                'channels': list(range(filtered_array.shape[2])),
                'units': list(range(filtered_array.shape[3])),
                'total_spikes': int(np.sum(filtered_array)),
                'stimulus_durations_ms': [3000] * filtered_array.shape[0],
                'original_stimuli_count': len(selected_indices),
                'filtered_stimuli_count': filtered_array.shape[0],
                'stimuli_removed': 0,
                'filter_criteria': {
                    'contrast': contrast,
                    'frequency': frequency,
                    'stim_type': stim_type,
                    'original_indices': selected_indices
                }
            }

            base_name = npz_file.stem.replace("_4d", "")
            contrast_str = f"c{int(contrast)}" if contrast is not None else "cAny"
            freq_str = f"f{int(frequency)}Hz" if frequency is not None else "fAnyHz"

            output_filename = f"{base_name}_{contrast_str}_{freq_str}_filtered_4d.npz"
            output_path = os.path.join(self.output_dir, output_filename)

            np.savez_compressed(output_path, array=filtered_array)

            metadata_filename = f"{base_name}_{contrast_str}_{freq_str}_filtered_metadata.npz"
            metadata_path = os.path.join(self.output_dir, metadata_filename)
            np.savez_compressed(metadata_path, **metadata)


def main():
    parser = argparse.ArgumentParser(description="Create filtered NPZ files for specific stimulus conditions")

    parser.add_argument("--data_dir", type=str, default="Data/",
                       help="Directory containing NPZ files")
    parser.add_argument("--stims_json", type=str, default="Gnat2.stims.json",
                       help="Path to stimulus configuration JSON file")
    parser.add_argument("--output_dir", type=str, default="Filtered_Data/",
                       help="Output directory for filtered NPZ files")

    parser.add_argument("--contrast", type=float, default=None,
                       help="Contrast value to filter (e.g., 0, 10, 30, 50, 70, 90)")
    parser.add_argument("--frequency", type=float, default=None,
                       help="Frequency value to filter (e.g., 2, 4, 6, 8, 10, 12)")
    parser.add_argument("--stim_type", type=str, default="FFSine",
                       help="Stimulus type to filter (default: FFSine)")

    args = parser.parse_args()


    creator = FilteredNPZCreator(args.data_dir, args.stims_json, args.output_dir)

    try:
        if args.contrast is None:
            creator.write_contrast_folders(frequency=args.frequency, stim_type=args.stim_type)
        else:
            creator.create_filtered_npz_files(
                contrast=args.contrast, frequency=args.frequency, stim_type=args.stim_type
            )
            print(f"Wrote filtered array to {args.output_dir}")
    except Exception as exc:
        print(f"Filter failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
