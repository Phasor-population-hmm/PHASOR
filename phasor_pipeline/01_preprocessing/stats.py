#!/usr/bin/env python3
"""
Clean Statistics Analysis with Channel Detection
Usage: python stats.py data.npz stimuli.json spike_data.txt [output_dir]
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import os
import textwrap
import pandas as pd
from collections import Counter
import seaborn as sns

class CleanStatsAnalysis:

    def __init__(self, npz_file, json_file, txt_file=None):
        self.npz_file = npz_file
        self.json_file = json_file
        self.txt_file = txt_file

        self.extract_metadata()

        self.load_npz_data()
        self.load_json_data()

        if txt_file and os.path.exists(txt_file):
            self.analyze_txt_channels()
        else:
            self.txt_channels = None

        self.calculate_statistics()

    def extract_metadata(self):
        """Extract key metadata from filenames"""
        npz_basename = os.path.basename(self.npz_file).lower()

        if 'rd' in npz_basename:
            self.genotype = 'RD'
        elif 'wt' in npz_basename or 'wildtype' in npz_basename:
            self.genotype = 'WT'
        elif 'gnat' in npz_basename:
            self.genotype = 'GNAT'
        elif 'ko' in npz_basename or 'knockout' in npz_basename:
            self.genotype = 'KO'
        else:
            self.genotype = 'Unknown'

        if 'slamdunk' in npz_basename or 'slam_dunk' in npz_basename:
            self.dataset_type = 'SLAM_DUNK'
        elif 'gnat' in npz_basename:
            self.dataset_type = 'GNAT'
        else:
            self.dataset_type = 'Unknown'

        self.filename = os.path.basename(self.npz_file)

    def load_npz_data(self):
        """Load and analyze NPZ data"""
        data_dict = np.load(self.npz_file)
        data_key = list(data_dict.keys())[0]
        self.spike_data = data_dict[data_key]

        self.n_stimuli = self.spike_data.shape[0]
        self.n_timebins = self.spike_data.shape[1]
        self.n_channels = self.spike_data.shape[2]
        self.n_units_per_channel = self.spike_data.shape[3]
        self.total_units = self.n_channels * self.n_units_per_channel

        self.file_size_mb = os.path.getsize(self.npz_file) / (1024*1024)
        self.data_type = str(self.spike_data.dtype)

    def load_json_data(self):
        """Load JSON stimuli data"""
        with open(self.json_file, 'r') as f:
            stim_data = json.load(f)

        stimuli = stim_data.get('stimuli', [])
        self.n_stimuli_json = len(stimuli)

        stim_types = [stim.get('stimType', 'Unknown') for stim in stimuli]
        self.stim_type_counts = Counter(stim_types)

        durations = [stim.get('durationMs', 0) for stim in stimuli]
        self.mean_duration_ms = np.mean(durations) if durations else 0

    def analyze_txt_channels(self):
        """Analyze txt file to identify actual channels present and missing"""

        try:
            with open(self.txt_file, 'r') as f:
                first_line = f.readline().strip()

            if ',' in first_line:
                separator = ','
            else:
                separator = '\t'

            if 'Channel' in first_line or 'channel' in first_line.lower():
                df = pd.read_csv(self.txt_file, sep=separator)
            else:
                df = pd.read_csv(self.txt_file, header=None, names=['Channel', 'Unit', 'Timestamp'], sep=separator)

            unique_channels = sorted(df['Channel'].unique())
            unique_units = sorted(df['Unit'].unique())

            expected_channels = set(range(1, 121))
            present_channels = set(unique_channels)
            missing_channels = sorted(expected_channels - present_channels)

            channel_spike_counts = df['Channel'].value_counts().sort_index()

            units_per_channel = df.groupby('Channel')['Unit'].nunique().sort_index()

            self.txt_channels = {
                'present_channels': unique_channels,
                'missing_channels': missing_channels,
                'n_present_channels': len(unique_channels),
                'n_missing_channels': len(missing_channels),
                'unique_units': unique_units,
                'channel_spike_counts': channel_spike_counts,
                'units_per_channel': units_per_channel,
                'total_txt_spikes': len(df)
            }

            if missing_channels:
                pass

        except Exception as e:
            self.txt_channels = None

    def calculate_statistics(self):
        """Calculate all key statistics"""
        total_spikes = np.sum(self.spike_data)
        firing_rates = self.spike_data / 0.01
        mean_firing_rate = np.mean(firing_rates)
        max_firing_rate = np.max(firing_rates)

        reshaped_data = self.spike_data.reshape(-1, self.total_units)
        unit_total_spikes = np.sum(reshaped_data, axis=0)
        active_units = np.sum(unit_total_spikes > 0)
        filler_units = self.total_units - active_units

        sparsity = np.sum(self.spike_data == 0) / self.spike_data.size * 100

        recording_time_s = self.n_timebins * 0.01

        self.stats = {
            'total_spikes': int(total_spikes),
            'mean_firing_rate': float(mean_firing_rate),
            'max_firing_rate': float(max_firing_rate),
            'active_units': int(active_units),
            'filler_units': int(filler_units),
            'sparsity': float(sparsity),
            'recording_time_s': float(recording_time_s)
        }

    def create_comprehensive_stats_plot(self, output_dir='clean_stats'):
        """Create one comprehensive statistics visualization"""
        os.makedirs(output_dir, exist_ok=True)

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#593E85', '#F4B942', '#8B4A8B']
        plt.style.use('default')

        fig = plt.figure(figsize=(22, 15))
        gs = fig.add_gridspec(3, 4, hspace=0.48, wspace=0.42,
                              left=0.05, right=0.98, top=0.92, bottom=0.06)

        fig.suptitle(f'Data Statistics Dashboard - Data Type: {self.genotype}',
                     fontsize=18, fontweight='bold', y=0.96)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')

        key_stats = f"""KEY STATISTICS

Total Spikes: {self.stats['total_spikes']:,}
Mean Rate: {self.stats['mean_firing_rate']:.2f} Hz
Max Rate: {self.stats['max_firing_rate']:.1f} Hz
Recording: {self.stats['recording_time_s']:.1f}s

DIMENSIONS
Stimuli: {self.n_stimuli:,}
Timebins: {self.n_timebins:,}
Channels: {self.n_channels:,}
Total Units: {self.total_units:,}

FILE INFO
Size: {self.file_size_mb:.1f} MB
Type: {self.data_type}
Sparsity: {self.stats['sparsity']:.1f}%"""

        ax1.text(0.05, 0.95, key_stats, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))

        ax2 = fig.add_subplot(gs[0, 1])
        sizes = [self.stats['active_units'], self.stats['filler_units']]
        labels = ['Active Units', 'Filler Units']
        colors_donut = [colors[0], colors[1]]

        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, colors=colors_donut,
            autopct='%1.1f%%', startangle=90, pctdistance=0.75, labeldistance=1.08,
            wedgeprops=dict(width=0.5),
            textprops={'fontweight': 'bold', 'fontsize': 12, 'color': 'black'})
        ax2.set_xlim(-1.7, 1.55)
        ax2.set_ylim(-1.45, 1.45)
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
            autotext.set_color('black')

        ax2.text(0, 0, f'{self.stats["active_units"]:,}\nActive\nUnits',
                ha='center', va='center', fontsize=11, fontweight='bold', color='black')
        ax2.set_title('Unit Activity Status', fontsize=12, fontweight='bold', pad=8)
        box = ax2.get_position()
        ax2.set_position([box.x0 - 0.018, box.y0, box.width, box.height])

        ax3 = fig.add_subplot(gs[0, 2])
        if self.txt_channels:
            present = self.txt_channels['n_present_channels']
            missing = self.txt_channels['n_missing_channels']

            bars = ax3.bar(['Present', 'Missing'], [present, missing],
                          color=[colors[2], colors[3]], alpha=0.8)
            ax3.set_ylabel('Number of Channels')
            ax3.set_title(f'Channel Status (Total: 120)', fontweight='bold', pad=10)
            ax3.set_ylim(0, 120)
            ax3.margins(y=0)

            for bar, value in zip(bars, [present, missing]):
                height = bar.get_height()
                x = bar.get_x() + bar.get_width() / 2
                if height >= 100:
                    ax3.text(x, height * 0.55, f'{value}', ha='center', va='center',
                            fontweight='bold', color='black', fontsize=12, clip_on=True)
                else:
                    ax3.text(x, height + 3, f'{value}', ha='center', va='bottom',
                            fontweight='bold', color='black', fontsize=12, clip_on=True)
        else:
            ax3.text(0.5, 0.5, 'No TXT file\nprovided', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Channel Analysis', fontweight='bold')

        ax4 = fig.add_subplot(gs[0, 3])
        stim_types = list(self.stim_type_counts.keys())
        stim_counts = list(self.stim_type_counts.values())

        bars = ax4.bar(range(len(stim_types)), stim_counts,
                      color=colors[:len(stim_types)], alpha=0.8)
        ax4.set_xticks(range(len(stim_types)))
        ax4.set_xticklabels(stim_types, rotation=30, ha='right')
        ax4.set_ylabel('Count')
        ax4.set_title('Stimulus Types', fontweight='bold', pad=10)
        top_count = max(stim_counts) if stim_counts else 1
        ax4.set_ylim(0, top_count * 1.22)

        for bar, count in zip(bars, stim_counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top_count * 0.03,
                    f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color='black', clip_on=True)

        ax5 = fig.add_subplot(gs[1, :2])
        temporal_pattern = np.mean(self.spike_data, axis=(0, 2, 3)) / 0.01
        time_ms = np.arange(len(temporal_pattern)) * 10

        ax5.plot(time_ms, temporal_pattern, color=colors[0], linewidth=2)
        ax5.fill_between(time_ms, temporal_pattern, alpha=0.3, color=colors[0])
        ax5.set_xlabel('Time (ms)')
        ax5.set_ylabel('Mean Firing Rate (Hz)')
        ax5.set_title('Average Temporal Activity Pattern', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 2:])
        spatial_pattern = np.mean(self.spike_data, axis=(0, 1, 3)) / 0.01

        ax6.bar(range(len(spatial_pattern)), spatial_pattern,
               color=colors[1], alpha=0.7, width=0.8)
        ax6.set_xlabel('Channel')
        ax6.set_ylabel('Mean Firing Rate (Hz)')
        ax6.set_title('Spatial Activity Pattern (by Channel)', fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')

        ax8 = fig.add_subplot(gs[2, 1:3])
        if self.txt_channels:
            channel_counts = self.txt_channels['channel_spike_counts']
            channels = channel_counts.index
            counts = channel_counts.values

            ax8.bar(channels, counts, color=colors[3], alpha=0.7, width=0.8)
            ax8.set_xlabel('Channel Number')
            ax8.set_ylabel('Number of Spikes')
            ax8.set_title(f'Spike Distribution by Channel (TXT Data: {self.txt_channels["total_txt_spikes"]:,} spikes)',
                         fontweight='bold')
            ax8.grid(True, alpha=0.3, axis='y')

            if self.txt_channels['missing_channels']:
                for missing_ch in self.txt_channels['missing_channels'][:10]:
                    if missing_ch <= max(channels):
                        ax8.axvline(x=missing_ch, color='red', linestyle='--', alpha=0.7)
        else:
            ax8.text(0.5, 0.5, 'TXT File Analysis\nNot Available',
                    ha='center', va='center', transform=ax8.transAxes,
                    fontsize=14, alpha=0.6)
            ax8.set_title('Channel Spike Distribution', fontweight='bold')

        ax9 = fig.add_subplot(gs[2, 3])
        ax9.axis('off')

        if self.txt_channels:
            missing_list = self.txt_channels['missing_channels']
            missing_str = f"Missing: {len(missing_list)}"
            if missing_list:
                missing_channels_str = "Missing Ch:\n" + textwrap.fill(
                    ", ".join(str(ch) for ch in missing_list), width=22)
            else:
                missing_channels_str = "Missing Ch: None"
            txt_spikes = f"TXT Spikes: {self.txt_channels['total_txt_spikes']:,}"
        else:
            missing_str = "Missing: N/A"
            missing_channels_str = "Missing Ch: N/A"
            txt_spikes = "TXT Spikes: N/A"

        summary_text = f"""SUMMARY

Data Type: {self.genotype}

NPZ Channels: {self.n_channels}
{missing_str}
{missing_channels_str}

Total Recording: {self.stats['recording_time_s']:.1f}s
JSON Stimuli: {self.n_stimuli_json}
{txt_spikes}

Active Units: {self.stats['active_units']:,}
Filler Units: {self.stats['filler_units']:,}
Mean Rate: {self.stats['mean_firing_rate']:.2f} Hz
Sparsity: {self.stats['sparsity']:.1f}%"""

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.5))

        plt.savefig(f"{output_dir}/comprehensive_stats_dashboard.png",
                   dpi=300, bbox_inches='tight')
        plt.close()


    def print_summary(self):
        """Print key statistics summary"""

        if self.txt_channels:
            if self.txt_channels['missing_channels']:
                pass


    def run_analysis(self, output_dir='clean_stats'):
        """Run the complete analysis"""

        self.create_comprehensive_stats_plot(output_dir)
        self.print_summary()


        return {
            'stats': self.stats,
            'genotype': self.genotype,
            'dataset_type': self.dataset_type,
            'txt_channels': self.txt_channels
        }

def main():
    if len(sys.argv) < 3:
        return

    npz_file = sys.argv[1]
    json_file = sys.argv[2]
    txt_file = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].startswith('output') else None
    output_dir = sys.argv[4] if len(sys.argv) > 4 else (sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].startswith('output') else 'clean_stats')

    try:
        analyzer = CleanStatsAnalysis(npz_file, json_file, txt_file)
        results = analyzer.run_analysis(output_dir)

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
