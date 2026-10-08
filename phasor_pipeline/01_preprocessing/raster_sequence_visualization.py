#!/usr/bin/env python3
"""
Raster Plot Sequence Visualization
Creates separate raster plots for each sequence (trial) showing spikes across all channels

Usage: python raster_sequence_visualization.py <npz_file> <stims_json> <output_dir>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
import json
import sys


class RasterSequenceVisualizer:

    def __init__(self, npz_file, stims_json, output_dir, n_sequences=10, stimulus_frames_dir=None):
        self.npz_file = npz_file
        self.stims_json = stims_json
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_sequences = n_sequences
        if stimulus_frames_dir:
            self.stimulus_frames_dir = Path(stimulus_frames_dir).expanduser().resolve()
        else:
            self.stimulus_frames_dir = Path(__file__).resolve().parent / "stimulus_frames"
        if not self.stimulus_frames_dir.is_dir():
            self.stimulus_frames_dir = None

        self.load_spike_data()
        self.load_stims()

        self.detect_frequency()

    def detect_frequency(self):
        """Detect stimulus frequency from filename"""
        filename_lower = str(self.npz_file).lower()
        if '2hz' in filename_lower or 'f2hz' in filename_lower:
            self.freq_hz = 2
        elif '4hz' in filename_lower or 'f4hz' in filename_lower:
            self.freq_hz = 4
        else:
            self.freq_hz = 2

    def load_spike_data(self):
        """Load spike data from NPZ file"""
        data_dict = np.load(self.npz_file)
        data_key = list(data_dict.keys())[0]
        self.spike_data_full = data_dict[data_key]

        self.n_stimuli_full = self.spike_data_full.shape[0]
        self.n_timebins = self.spike_data_full.shape[1]
        self.n_channels = self.spike_data_full.shape[2]
        self.n_units_per_channel = self.spike_data_full.shape[3]


    def load_stims(self):
        """Load stimulus sequence from JSON file - SKIP ALL INTEGRITY FLASHES"""
        with open(self.stims_json, 'r') as f:
            stim_data = json.load(f)

        self.stimuli = stim_data.get('stimuli', [])

        integrity_count = sum(1 for s in self.stimuli if s.get('stimType', '') == 'Solid')

        self.stim_sequence = []

        self.filtered_stimuli = []
        for stim in self.stimuli:
            meta_comment = stim.get('meta', {}).get('comment', '').lower()
            direct_comment = stim.get('comment', '').lower()
            if not (meta_comment == 'rest' or direct_comment == 'rest' or
                    meta_comment == 'final black' or direct_comment == 'final black' or
                    meta_comment == 'initial black' or direct_comment == 'initial black'):
                self.filtered_stimuli.append(stim)

        self.ffsine_npz_indices = [i for i, s in enumerate(self.filtered_stimuli) if s.get('stimType', '') == 'FFSine']
        self.integrity_npz_indices = [i for i, s in enumerate(self.filtered_stimuli)
                                      if s.get('stimType', '') == 'Solid' and
                                      ('integrity flash' in str(s.get('meta', {}).get('comment', '')).lower() or
                                       'integrity flash' in str(s.get('comment', '')).lower())]
        if len(self.ffsine_npz_indices) == 0:
            raise ValueError("No FFSine stimuli found!")

        for stim in self.stimuli:
            if stim.get('stimType', '') == 'FFSine':
                contrast_value = stim.get('c', 0)
                contrast = f"c{contrast_value}"
                self.stim_sequence.append(contrast)


    def get_sequence_data(self, trial_idx):
        """Get spike data for a specific sequence (trial) - handles shuffled stimuli"""
        sequence_pattern = ['c0', 'c50', 'c0', 'c60', 'c0', 'c70', 'c0', 'c80', 'c0', 'c90']
        sequence_length = len(sequence_pattern)

        is_shuffled = self.stim_sequence[:10] != sequence_pattern

        if is_shuffled:
            contrast_occurrences = {}
            for idx, contrast in enumerate(self.stim_sequence):
                if contrast not in contrast_occurrences:
                    contrast_occurrences[contrast] = []
                contrast_occurrences[contrast].append(idx)

            pattern_occurrence_counts = {}
            for contrast in sequence_pattern:
                pattern_occurrence_counts[contrast] = pattern_occurrence_counts.get(contrast, 0) + 1

            npz_indices = []
            current_occurrence_counts = {}

            for contrast in sequence_pattern:
                current_occurrence_counts[contrast] = current_occurrence_counts.get(contrast, 0) + 1
                occurrence_num = current_occurrence_counts[contrast]

                occurrences_per_sequence = pattern_occurrence_counts[contrast]
                global_occurrence_num = (trial_idx) * occurrences_per_sequence + occurrence_num
                occurrence_idx = global_occurrence_num - 1

                if contrast in contrast_occurrences and occurrence_idx < len(contrast_occurrences[contrast]):
                    npz_indices.append(contrast_occurrences[contrast][occurrence_idx])
                else:
                    npz_indices.append(None)
        else:
            start_seq_idx = trial_idx * sequence_length
            end_seq_idx = start_seq_idx + sequence_length
            max_available = len(self.ffsine_npz_indices)
            if end_seq_idx > max_available:
                return None, None
            npz_indices = list(range(start_seq_idx, end_seq_idx))

        sequence_spikes = []
        for npz_idx in npz_indices:
            if npz_idx is None:
                continue
            if 0 <= npz_idx < len(self.ffsine_npz_indices):
                actual_npz_idx = self.ffsine_npz_indices[npz_idx]
                stim_data = self.spike_data_full[actual_npz_idx, :, :, :]
                sequence_spikes.append(stim_data)
            else:
                pass

        if len(sequence_spikes) == 0:
            return None, None

        if len(sequence_spikes) < sequence_length:
            pass

        concatenated = np.concatenate(sequence_spikes, axis=0)

        spikes_per_channel = np.sum(concatenated, axis=2)

        return spikes_per_channel, sequence_pattern

    def create_raster_plot(self, trial_idx, spikes_per_channel, sequence_pattern):
        """Create a single raster plot figure for one sequence"""

        actual_timebins = spikes_per_channel.shape[0]
        total_time_ms = actual_timebins * 10
        time_ms = np.arange(actual_timebins) * 10
        sequence_length = len(sequence_pattern)


        fig = plt.figure(figsize=(26, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[0.8, 1.5, 8], hspace=0.12)

        ax_sine = fig.add_subplot(gs[0, 0])
        n_pts = max(4000, int(self.freq_hz * (total_time_ms / 1000.0) * 40))
        time_continuous = np.linspace(0, total_time_ms, n_pts)
        t_s = time_continuous / 1000.0

        sine = np.zeros_like(t_s)
        actual_stim_count_for_sine = min(len(sequence_pattern), spikes_per_channel.shape[0] // self.n_timebins)
        for idx in range(actual_stim_count_for_sine):
            contrast = sequence_pattern[idx] if idx < len(sequence_pattern) else 'c0'
            start_time_ms = idx * self.n_timebins * 10
            end_time_ms = min((idx + 1) * self.n_timebins * 10, total_time_ms)
            start_time_s = start_time_ms / 1000.0
            end_time_s = end_time_ms / 1000.0

            mask = (t_s >= start_time_s) & (t_s <= end_time_s)
            digits = "".join(ch for ch in str(contrast) if ch.isdigit())
            amplitude = (float(digits) / 100.0) if digits else 0.0
            if amplitude > 0:
                sine[mask] = amplitude * np.sin(2 * np.pi * self.freq_hz * t_s[mask])

        ax_sine.plot(time_continuous, sine, color='black', linewidth=2.5)
        ax_sine.set_ylim(-1.2, 1.2)
        ax_sine.set_yticks([-1, 0, 1])
        ax_sine.set_yticklabels(['-1', '0', '1'], fontsize=14, fontweight='bold')
        ax_sine.set_ylabel(f'{self.freq_hz}Hz', fontsize=16, fontweight='bold')
        ax_sine.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_sine.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax_sine.set_xlim(0, total_time_ms)
        ax_sine.spines['top'].set_visible(False)
        ax_sine.spines['right'].set_visible(False)
        ax_sine.spines['bottom'].set_visible(False)

        ax_stim = fig.add_subplot(gs[1, 0])
        ax_stim.set_xlim(0, total_time_ms)
        ax_stim.set_ylim(0, 1)
        ax_stim.axis('off')

        stim_img_dir = self.stimulus_frames_dir

        actual_stim_count = min(len(sequence_pattern), spikes_per_channel.shape[0] // self.n_timebins)

        for idx in range(actual_stim_count):
            contrast = sequence_pattern[idx] if idx < len(sequence_pattern) else 'c0'
            start_time = idx * self.n_timebins * 10
            end_time = min(start_time + self.n_timebins * 10, total_time_ms)
            center_time = (start_time + end_time) / 2.0

            img_file = (stim_img_dir / f"stim_{contrast}.png") if stim_img_dir else None
            drew_image = False
            if img_file and img_file.exists():
                try:
                    img = plt.imread(img_file)
                    h = img.shape[0]
                    img_cropped = img[int(h*0.5):, :, :]
                    extent = [start_time, end_time, 0, 1]
                    ax_stim.imshow(img_cropped, extent=extent, aspect='auto', zorder=1)
                    drew_image = True
                except Exception:
                    drew_image = False
            if not drew_image:
                ax_stim.add_patch(plt.Rectangle(
                    (start_time, 0), end_time - start_time, 1,
                    facecolor="#39FF14", edgecolor="white", linewidth=2, zorder=1))

            contrast_val = contrast.replace('c', '')
            contrast_label = f"C-{contrast_val}"
            ax_stim.text(center_time, 0.5, contrast_label,
                        ha='center', va='center',
                        fontsize=50, fontweight='bold',
                        color='black', zorder=3)

            if idx > 0:
                ax_stim.axvline(start_time, color='white', linestyle='-', linewidth=2, zorder=2)

        ax_raster = fig.add_subplot(gs[2, 0])

        spike_times = []
        spike_channels = []

        for channel_idx in range(self.n_channels):
            for time_idx in range(spikes_per_channel.shape[0]):
                if spikes_per_channel[time_idx, channel_idx] > 0:
                    spike_times.append(time_idx * 10)
                    spike_channels.append(channel_idx + 1)

        if len(spike_times) > 0:
            ax_raster.scatter(spike_times, spike_channels, s=0.5, c='black', alpha=0.6, marker='|')

        ax_raster.set_xlim(0, total_time_ms)
        ax_raster.set_ylim(0.5, self.n_channels + 0.5)

        ax_raster.set_xlabel('Time (s)', fontsize=40, fontweight='bold', labelpad=15)
        x_ticks = ax_raster.get_xticks()
        ax_raster.set_xticklabels([f'{int(t/1000)}' for t in x_ticks], fontsize=32, fontweight='bold')

        ax_raster.set_ylabel('Channel #', fontsize=48, fontweight='bold', labelpad=30)
        y_tick_positions = [1, 60, 120]
        y_tick_labels = ['1', '60', '120']
        ax_raster.set_yticks(y_tick_positions)
        ax_raster.set_yticklabels(y_tick_labels, fontsize=32, fontweight='bold')

        ax_raster.tick_params(axis='both', labelsize=32, width=2, length=8)
        ax_raster.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='x')

        actual_stim_count = spikes_per_channel.shape[0] // self.n_timebins
        for idx in range(1, actual_stim_count):
            boundary_time = idx * self.n_timebins * 10
            if boundary_time < total_time_ms:
                ax_raster.axvline(boundary_time, color='gray', linestyle='--',
                                 linewidth=1.5, alpha=0.4)

        fig.text(0.5, 0.995, f'Raster Plot - Sequence {trial_idx + 1}',
                ha='center', va='top', fontsize=28, fontweight='bold')
        fig.text(0.5, 0.955, 'Stimulus Sequence Pattern',
                ha='center', va='top', fontsize=22, fontweight='bold', style='italic')

        output_file = self.output_dir / f"raster_sequence_{trial_idx + 1:02d}.png"
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(output_file, dpi=150)
        plt.close()

        return output_file

    def create_integrity_flash_plot(self, integrity_indices, title_suffix, total_sequences=None):
        """Create raster plot for integrity flashes"""

        filtered_stimuli = getattr(self, 'filtered_stimuli', None)
        if filtered_stimuli is None:
            with open(self.stims_json, 'r') as f:
                stim_data = json.load(f)
            all_stimuli = stim_data.get('stimuli', [])
            filtered_stimuli = []
            for stim in all_stimuli:
                meta_comment = stim.get('meta', {}).get('comment', '').lower()
                direct_comment = stim.get('comment', '').lower()
                if not (meta_comment == 'rest' or direct_comment == 'rest' or
                        meta_comment == 'final black' or direct_comment == 'final black' or
                        meta_comment == 'initial black' or direct_comment == 'initial black'):
                    filtered_stimuli.append(stim)

        sequence_spikes = []

        duration_per_flash_ms = 1040
        if len(integrity_indices) > 0 and integrity_indices[0] < len(filtered_stimuli):
            stim0 = filtered_stimuli[integrity_indices[0]]
            body_ms = int(stim0.get('bodyMs', 520) or 520)
            tail_ms = int(stim0.get('tailMs', 520) or 520)
            duration_per_flash_ms = int(body_ms + tail_ms)

        actual_timebins_per_flash = int(round(duration_per_flash_ms / 10.0))

        for npz_idx in integrity_indices:
            if 0 <= npz_idx < self.spike_data_full.shape[0]:
                stim_data = self.spike_data_full[npz_idx, :, :, :]
                stim_data_truncated = stim_data[:actual_timebins_per_flash, :, :]
                sequence_spikes.append(stim_data_truncated)

        if len(sequence_spikes) == 0:
            return None

        concatenated = np.concatenate(sequence_spikes, axis=0)
        spikes_per_channel = np.sum(concatenated, axis=2)

        duration_per_flash = duration_per_flash_ms
        total_timebins = actual_timebins_per_flash * len(integrity_indices)

        if spikes_per_channel.shape[0] != total_timebins:
            if spikes_per_channel.shape[0] > total_timebins:
                spikes_per_channel = spikes_per_channel[:total_timebins, :]

        n_flashes = len(integrity_indices)
        total_time_ms = duration_per_flash * n_flashes

        fig = plt.figure(figsize=(26, 14))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 9], hspace=0.05, left=0.05, right=0.98, top=0.94, bottom=0.05)

        ax_stim = fig.add_subplot(gs[0, 0])
        ax_stim.set_xlim(0, total_time_ms)
        ax_stim.set_ylim(0, 1)
        ax_stim.axis('off')

        def normalize_color(bg_color):
            bg_color_lower = str(bg_color).lower()
            if 'oklch' in bg_color_lower or 'gray' in bg_color_lower or '0.5' in bg_color_lower:
                return 'gray'
            elif 'red' in bg_color_lower:
                return 'red'
            elif 'green' in bg_color_lower:
                return 'green'
            elif 'blue' in bg_color_lower:
                return 'blue'
            return 'gray'

        n_flashes = len(integrity_indices)
        duration_per_flash = duration_per_flash_ms

        for idx, npz_idx in enumerate(integrity_indices):
            if npz_idx < len(filtered_stimuli):
                stim = filtered_stimuli[npz_idx]
                bg_color = stim.get('bgColor', 'unknown')
                color_name = normalize_color(bg_color)
                body_ms = int(stim.get('bodyMs', 520) or 520)
                tail_ms = int(stim.get('tailMs', 520) or 520)

                flash_start = idx * duration_per_flash
                body_start = flash_start
                body_end = body_start + body_ms
                tail_start = body_end
                tail_end = tail_start + tail_ms

                color_map = {
                    'gray': '#808080',
                    'red': '#FF0000',
                    'green': '#00FF00',
                    'blue': '#0000FF'
                }
                body_color = color_map.get(color_name, '#808080')
                ax_stim.add_patch(plt.Rectangle((body_start, 0), body_ms, 1,
                                              facecolor=body_color, edgecolor='white', linewidth=2))

                ax_stim.add_patch(plt.Rectangle((tail_start, 0), tail_ms, 1,
                                              facecolor='black', edgecolor='white', linewidth=2))

                if idx > 0:
                    ax_stim.axvline(flash_start, color='white', linestyle='-', linewidth=2, zorder=2)
            else:
                pass

        ax_raster = fig.add_subplot(gs[1, 0])

        spike_times = []
        spike_channels = []

        for channel_idx in range(self.n_channels):
            for time_idx in range(spikes_per_channel.shape[0]):
                n_spikes = int(spikes_per_channel[time_idx, channel_idx])
                if n_spikes > 0:
                    time_ms = time_idx * 10
                    for _ in range(n_spikes):
                        spike_times.append(time_ms)
                        spike_channels.append(channel_idx + 1)

        if len(spike_times) > 0:
            ax_raster.scatter(spike_times, spike_channels, s=8.0, c='black', alpha=1.0, marker='|', linewidths=2.0)
        else:
            pass

        ax_raster.set_xlim(0, total_time_ms)
        ax_raster.set_ylim(0.5, self.n_channels + 0.5)
        ax_raster.set_xlabel('Time (ms)', fontsize=20, fontweight='bold', labelpad=10)
        ax_raster.set_ylabel('Channel', fontsize=20, fontweight='bold', labelpad=10)
        ax_raster.tick_params(axis='both', labelsize=16, width=1.5, length=6)
        ax_raster.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='x')

        for idx, npz_idx in enumerate(integrity_indices):
            flash_start_bin = idx * actual_timebins_per_flash
            flash_end_bin = flash_start_bin + actual_timebins_per_flash
            flash_data = spikes_per_channel[flash_start_bin:flash_end_bin, :]
            total_spikes = flash_data.sum()
            active_channels = (flash_data.sum(axis=0) > 0).sum()
            if npz_idx < len(filtered_stimuli):
                stim = filtered_stimuli[npz_idx]
                bg_color = stim.get('bgColor', 'unknown')
                color_name = normalize_color(bg_color)
            else:
                pass

        for idx in range(n_flashes):
            flash_start = idx * duration_per_flash
            flash_end = (idx + 1) * duration_per_flash

            if idx > 0:
                ax_raster.axvline(flash_start, color='darkgray', linestyle='-', linewidth=2.5, alpha=0.8, zorder=1)

            body_end = flash_start + body_ms
            ax_raster.axvline(body_end, color='darkblue', linestyle='--', linewidth=2, alpha=0.7, zorder=1)

            if idx < n_flashes - 1:
                ax_raster.axvline(flash_end, color='darkgray', linestyle='-', linewidth=2.5, alpha=0.8, zorder=1)

        fig.text(0.5, 0.985, f'Raster Plot - {title_suffix} Integrity Flashes',
                ha='center', va='top', fontsize=26, fontweight='bold')

        if "First" in title_suffix:
            output_file = self.output_dir / f"raster_sequence_00_integrity_first.png"
        else:
            if total_sequences is not None:
                seq_num = total_sequences
            else:
                seq_num = self.n_sequences + 1
            output_file = self.output_dir / f"raster_sequence_{seq_num:02d}_integrity_last.png"
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
        plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        return output_file

    def create_all_visualizations(self):
        """Create raster plots for all sequences including integrity flashes"""

        with open(self.stims_json, 'r') as f:
            stim_data = json.load(f)

        all_stimuli = stim_data.get('stimuli', [])
        filtered_stimuli = []
        for stim in all_stimuli:
            meta_comment = stim.get('meta', {}).get('comment', '').lower()
            direct_comment = stim.get('comment', '').lower()
            if not (meta_comment == 'rest' or direct_comment == 'rest' or
                    meta_comment == 'final black' or direct_comment == 'final black' or
                    meta_comment == 'initial black' or direct_comment == 'initial black'):
                filtered_stimuli.append(stim)

        first_integrity_indices = []
        last_integrity_indices = []

        for i, stim in enumerate(filtered_stimuli):
            meta_comment = stim.get('meta', {}).get('comment', '').lower()
            direct_comment = stim.get('comment', '').lower()
            if 'integrity flash' in meta_comment or 'integrity flash' in direct_comment:
                if i < 10:
                    first_integrity_indices.append(i)
                else:
                    last_integrity_indices.append(i)

        output_files = []

        if len(first_integrity_indices) == 4:
            output_file = self.create_integrity_flash_plot(first_integrity_indices, "First 4")
            if output_file:
                output_files.append(output_file)

        sequence_length = 10
        n_available = len(getattr(self, "ffsine_npz_indices", [])) // sequence_length
        n_to_run = n_available if self.n_sequences == -1 else min(self.n_sequences, n_available)

        for trial_idx in range(n_to_run):
            try:
                spikes_per_channel, sequence_pattern = self.get_sequence_data(trial_idx)

                if spikes_per_channel is None:
                    continue

                output_file = self.create_raster_plot(trial_idx, spikes_per_channel, sequence_pattern)
                if output_file:
                    output_files.append(output_file)
            except (IndexError, ValueError) as e:
                continue

        if len(last_integrity_indices) == 4:
            total_sequences = n_available
            output_file = self.create_integrity_flash_plot(last_integrity_indices, "Last 4", total_sequences)
            if output_file:
                output_files.append(output_file)

        return output_files

    def create_selected_sequences(self, sequence_numbers_1based):
        """
        Create raster plots for only specific sequences (1-based numbers).
        This avoids confusion when you want, e.g., "sequence 5" exactly.
        Integrity plots are NOT generated in this mode.
        """
        output_files = []
        seqs = [int(x) for x in sequence_numbers_1based if int(x) >= 1]
        if not seqs:
            return output_files

        for seq in seqs:
            trial_idx = seq - 1
            try:
                spikes_per_channel, sequence_pattern = self.get_sequence_data(trial_idx)
                if spikes_per_channel is None:
                    continue
                output_file = self.create_raster_plot(trial_idx, spikes_per_channel, sequence_pattern)
                if output_file:
                    output_files.append(output_file)
            except (IndexError, ValueError) as e:
                continue

        return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Create raster plot visualizations for each sequence'
    )
    parser.add_argument('npz_file', type=str,
                       help='NPZ file with spike data')
    parser.add_argument('stims_json', type=str,
                       help='JSON file with stimulus sequence')
    parser.add_argument('output_dir', type=str,
                       help='Output directory for visualizations')
    parser.add_argument('--n_sequences', type=int, default=10,
                       help='Number of sequences to plot (default: 10). Use -1 to run ALL available sequences.')
    parser.add_argument('--sequence', type=int, default=0,
                       help='Plot exactly ONE sequence number (1-based). Overrides --n_sequences.')
    parser.add_argument('--stimulus_frames_dir', type=str, default=None,
                       help='Optional override. Default: stimulus_frames/ folder next to this script (code/Analysis/stimulus_frames).')

    args = parser.parse_args()

    try:
        visualizer = RasterSequenceVisualizer(
            args.npz_file,
            args.stims_json,
            args.output_dir,
            n_sequences=args.n_sequences,
            stimulus_frames_dir=args.stimulus_frames_dir
        )
        if int(args.sequence) > 0:
            output_files = visualizer.create_selected_sequences([int(args.sequence)])
        else:
            output_files = visualizer.create_all_visualizations()

        if output_files:
            for f in output_files:
                pass
        else:
            sys.exit(1)

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

