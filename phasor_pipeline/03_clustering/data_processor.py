#!/usr/bin/env python3
"""
Data loading + preprocessing for the clustering pipeline.
"""

import os
import csv
import warnings
import numpy as np
from typing import List, Tuple, Dict
from scipy.ndimage import gaussian_filter1d


class DataProcessor:
    """Load and preprocess 4D NPZ arrays for clustering."""

    def __init__(self, data_dir: str):
        """Create a processor for a folder containing `*_4d.npz` files."""
        self.data_dir = data_dir
        self.dataset_info = []
        self.genotype_labels = []
        self.sample_metadata = []

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")


    def infer_genotype_from_filename(self, filename: str) -> str:
        """Infer genotype label from a file path (best-effort)."""
        parts = [part for part in str(filename).replace("\\", "/").lower().split("/") if part]
        for part in reversed(parts):
            if part.startswith("wt"):
                return "wt"
            if part.startswith("rd"):
                return "rd"
            if "gnat" in part:
                return "gnat2"
        warnings.warn(f"Could not infer genotype from {filename!r}; using 'unknown'")
        return "unknown"

    def discover_datasets(self) -> List[Tuple[str, str]]:
        """Find all *_4d.npz files and infer genotypes

        Returns:
            List of (file_path, genotype) tuples
        """
        files = [f for f in os.listdir(self.data_dir) if f.endswith("_4d.npz")]

        if not files:
            raise RuntimeError(f"No *_4d.npz files found in {self.data_dir}")

        dataset_info = []
        for f in sorted(files):
            path = os.path.join(self.data_dir, f)
            genotype = self.infer_genotype_from_filename(f)
            dataset_info.append((path, genotype))


        return dataset_info

    def load_single_dataset(self, array_path: str) -> np.ndarray:
        """Load one 4D spike array from a `.npz` file."""
        try:
            arr = np.load(array_path)
            spike_array = arr['array'] if 'array' in arr.files else arr[list(arr.files)[0]]
            return spike_array
        except Exception as e:
            raise RuntimeError(f"Failed to load {array_path}: {e}")

    def process_single_dataset(self, spike_array: np.ndarray, dataset_name: str,
                              genotype: str, keep_timebins: int = 100) -> np.ndarray:
        """Convert a 4D array into per-(channel,unit) samples for clustering."""
        original_shape = spike_array.shape

        if len(original_shape) != 4:
            raise ValueError(f"Expected 4D array, got {len(original_shape)}D for {dataset_name}")

        S, T, C, U = original_shape

        k = min(keep_timebins, T) if keep_timebins is not None else T
        spike_array_trim = spike_array[:, :k, :, :]

        nonzero_mask_cu = np.any(np.abs(spike_array_trim) > 0, axis=(0, 1))
        kept_count = int(np.sum(nonzero_mask_cu))
        if kept_count == 0:
            raise ValueError(f"No non-zero (channel, unit) pairs in {dataset_name} after trimming to {k} timebins")
        removed_count = C * U - kept_count
        if removed_count > 0:
            pass

        avg_data = spike_array_trim.mean(axis=0)

        avg_data_flat = avg_data.reshape(k, C * U)
        mask_flat = nonzero_mask_cu.reshape(C * U)
        samples = avg_data_flat[:, mask_flat].T


        for c_idx in range(C):
            for u_idx in range(U):
                if nonzero_mask_cu[c_idx, u_idx]:
                    self.genotype_labels.append(genotype)
                    self.sample_metadata.append((dataset_name, c_idx + 1, u_idx + 1))

        self.dataset_info.append({
            'name': dataset_name,
            'genotype': genotype,
            'original_shape': original_shape,
            'num_samples': samples.shape[0],
            'num_timebins': samples.shape[1],
            'units': U,
            'channels': C
        })

        return samples.astype(np.float32)


    def combine_datasets(self, keep_timebins: int = 100) -> Tuple[np.ndarray, List[str], List[Tuple[str, int, int]]]:
        """Load and combine all datasets
        """
        dataset_info = self.discover_datasets()
        processed_datasets = []


        for i, (path, genotype) in enumerate(dataset_info):
            dataset_name = os.path.basename(path).replace('_4d.npz', '')
            spike_array = self.load_single_dataset(path)
            processed = self.process_single_dataset(spike_array, dataset_name, genotype, keep_timebins)
            processed_datasets.append(processed)

        combined = np.vstack(processed_datasets)

        self._print_combination_summary(combined)

        self._verify_metadata_alignment(combined)

        return combined, self.genotype_labels.copy(), self.sample_metadata.copy()

    def _print_combination_summary(self, combined: np.ndarray):
        """Print summary of combined datasets"""
        total_samples = combined.shape[0]

        genotype_counts = {}
        for info in self.dataset_info:
            genotype = info['genotype']
            genotype_counts[genotype] = genotype_counts.get(genotype, 0) + info['num_samples']

        for genotype, count in genotype_counts.items():
            percentage = count / total_samples * 100

    def _verify_metadata_alignment(self, combined: np.ndarray):
        """Verify that metadata arrays match data dimensions"""
        n_samples = combined.shape[0]

        if len(self.genotype_labels) != n_samples:
            raise RuntimeError(f"Label mismatch: {len(self.genotype_labels)} labels vs {n_samples} samples")

        if len(self.sample_metadata) != n_samples:
            raise RuntimeError(f"Metadata mismatch: {len(self.sample_metadata)} metadata vs {n_samples} samples")


    def apply_gaussian_smoothing(self, X: np.ndarray, sigma_bins: float = 3.0) -> np.ndarray:
        """Apply Gaussian smoothing
        """
        if sigma_bins <= 0:
            return X

        smoothed = gaussian_filter1d(X, sigma=sigma_bins, axis=1)
        return smoothed.astype(np.float32)

    def apply_zscore_normalization(self, X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Apply row-wise Z-score normalization

        Each sample (row) is normalized to mean=0, std=1
        """

        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)

        zero_std_count = np.sum(sd.flatten() < eps)
        if zero_std_count > 0:
            pass

        normalized = (X - mu) / (sd + eps)
        return normalized.astype(np.float32)


    def process_all_data(self, keep_timebins: int = 100,
                        sigma_bins: float = 3.0) -> Tuple[np.ndarray, List[str], List[Tuple[str, int, int]]]:
        """Complete Data Processing pipeline

        """
        X, genotype_labels, sample_metadata = self.combine_datasets(keep_timebins)

        X = self.apply_gaussian_smoothing(X, sigma_bins)

        X = self.apply_zscore_normalization(X)


        return X, genotype_labels, sample_metadata


    def save_metadata(self, output_dir: str, genotype_labels: List[str],
                     sample_metadata: List[Tuple[str, int, int]]) -> Tuple[str, str]:
        """Save sample metadata and genotype labels to files

        Returns:
            Tuple of (metadata_file_path, labels_file_path)
        """
        os.makedirs(output_dir, exist_ok=True)

        metadata_file = os.path.join(output_dir, "sample_metadata.csv")
        with open(metadata_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx", "dataset", "channel", "unit", "genotype"])
            for i, ((dataset, channel, unit), genotype) in enumerate(zip(sample_metadata, genotype_labels)):
                writer.writerow([i, dataset, channel, unit, genotype])

        labels_file = os.path.join(output_dir, "genotype_labels.txt")
        with open(labels_file, 'w') as f:
            for label in genotype_labels:
                f.write(f"{label}\n")


        return metadata_file, labels_file

    def get_dataset_summary(self) -> Dict:
        """Get summary of processed datasets

        Returns:
            Dictionary with dataset statistics
        """
        return {
            'datasets': self.dataset_info,
            'total_datasets': len(self.dataset_info),
            'total_samples': sum(info['num_samples'] for info in self.dataset_info),
            'genotype_labels': len(self.genotype_labels),
            'sample_metadata': len(self.sample_metadata)
        }

    def print_summary(self):
        """Print a summary of the processed data"""
        summary = self.get_dataset_summary()


        if self.dataset_info:
            for info in self.dataset_info:
                pass


def main():
    """Example usage of DataProcessor"""
    try:
        processor = DataProcessor("Data")

        X, genotype_labels, sample_metadata = processor.process_all_data(
            keep_timebins=100,
            sigma_bins=3.0
        )

        processor.save_metadata("test_output", genotype_labels, sample_metadata)

        processor.print_summary()

        for i in range(min(5, len(genotype_labels))):
            dataset, channel, unit = sample_metadata[i]
            genotype = genotype_labels[i]

    except Exception as e:
        raise


if __name__ == "__main__":
    main()
