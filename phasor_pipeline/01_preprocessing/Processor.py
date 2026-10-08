#!/usr/bin/env python3
"""
Usage: python Processor.py spike_file.txt synctones_file.csv stims.json
"""

import numpy as np
import pandas as pd
import json
import sys
import os
from pathlib import Path

def generate_arrays(spike_file, synctones_file, stims_file):


    with open(stims_file, 'r') as f:
        stims_data = json.load(f)

    synctones_df = pd.read_csv(synctones_file)
    synctones = synctones_df.iloc[:, 0].tolist()

    original_stimuli = stims_data['stimuli']
    filtered_stimuli = []
    synctone_mapping = []

    synctone_index = 0

    for i, stim in enumerate(original_stimuli):
        meta_comment = stim.get('meta', {}).get('comment', '').lower()
        direct_comment = stim.get('comment', '').lower()
        stim_type = stim.get('stimType', '').lower()

        if ('rest' in meta_comment or 'rest' in direct_comment or
            meta_comment == 'final black' or direct_comment == 'final black' or
            meta_comment == 'initial black' or direct_comment == 'initial black'):
            if not (meta_comment == 'initial black' or direct_comment == 'initial black'):
                synctone_index += 1
            continue

        filtered_stimuli.append(stim)
        synctone_mapping.append(synctone_index)
        synctone_index += 1


    if not all(synctones[i] < synctones[i + 1] for i in range(len(synctones) - 1)):
        raise ValueError("Synctone timestamps are not strictly increasing. Aborting.")

    if len(synctones) == 0:
        raise ValueError("No synctones found in file. Aborting.")

    missing_mapped = [idx for idx, s_idx in enumerate(synctone_mapping) if s_idx >= len(synctones)]
    if missing_mapped:
        first_bad = missing_mapped[0]
        raise ValueError(
            f"Mismatch: filtered stimuli require synctone index {synctone_mapping[first_bad]}, "
            f"but only {len(synctones)} synctones are available. "
            f"First failure at filtered stimulus {first_bad}. Aborting without saving.")

    stim_durations_ms = [stim['durationMs'] for stim in filtered_stimuli]
    max_duration_ms = max(stim_durations_ms)


    time_bin_size = 0.01
    max_time_bins = int(np.ceil(max_duration_ms / 1000.0 / time_bin_size))


    stim_intervals = []


    for filtered_idx, synctone_idx in enumerate(synctone_mapping):
        if synctone_idx < len(synctones):
            start_time = synctones[synctone_idx]

            planned_duration_s = stim_durations_ms[filtered_idx] / 1000.0
            end_time = start_time + planned_duration_s

            stim_intervals.append((start_time, end_time))

            if filtered_idx < 5:
                pass

    num_stimuli = len(stim_intervals)

    if num_stimuli != len(filtered_stimuli):
        raise ValueError(
            f"Mismatch: created {num_stimuli} stimulus intervals but have {len(filtered_stimuli)} filtered stimuli. "
            f"This indicates synctones/stims misalignment. Aborting without saving.")

    if synctone_mapping:
        first_synctone_idx = synctone_mapping[0]
        first_synctone = synctones[first_synctone_idx]

    interval_starts = np.array([s for s, _ in stim_intervals], dtype=np.float64)
    interval_ends = np.array([e for _, e in stim_intervals], dtype=np.float64)

    spike_events = []
    unique_channels = set()
    unique_units = set()
    total_spikes = 0

    for chunk in pd.read_csv(
        spike_file,
        chunksize=200000,
        header=0,
        usecols=['Channel', 'Unit', 'Timestamp'],
        dtype={'Channel': np.int16, 'Unit': np.int16, 'Timestamp': np.float64},
        engine='c'
    ):
        ts = chunk['Timestamp'].to_numpy(copy=False)
        ch = chunk['Channel'].to_numpy(copy=False)
        un = chunk['Unit'].to_numpy(copy=False)

        candidate = np.searchsorted(interval_starts, ts, side='right') - 1
        valid = (candidate >= 0)
        valid &= ts < interval_ends[np.clip(candidate, 0, len(interval_ends)-1)]

        if not np.any(valid):
            continue

        si = candidate[valid]
        rel = ts[valid] - interval_starts[si]
        tb = (rel / time_bin_size).astype(np.int64)
        in_range = (tb >= 0) & (tb < max_time_bins)

        if not np.any(in_range):
            continue

        si = si[in_range]
        tb = tb[in_range]
        ch_v = ch[valid][in_range].astype(int, copy=False)
        un_v = un[valid][in_range].astype(int, copy=False)

        spike_events.extend(zip(si.tolist(), tb.tolist(), ch_v.tolist(), un_v.tolist()))
        unique_channels.update(np.unique(ch_v).tolist())
        unique_units.update(np.unique(un_v).tolist())
        total_spikes += si.size


    channels = sorted(unique_channels)
    units = sorted(unique_units)

    shape = (num_stimuli, max_time_bins, len(channels), len(units))

    spike_array = np.zeros(shape, dtype=np.uint16)

    channel_map = {ch: idx for idx, ch in enumerate(channels)}
    unit_map = {unit: idx for idx, unit in enumerate(units)}

    for stimulus_id, time_bin, channel, unit in spike_events:
        if (stimulus_id < shape[0] and time_bin < shape[1]):
            ch_idx = channel_map[channel]
            unit_idx = unit_map[unit]
            spike_array[stimulus_id, time_bin, ch_idx, unit_idx] += 1

    memory_mb = spike_array.nbytes / (1024**2)

    metadata = {
        'shape': shape,
        'time_bin_ms': 10,
        'max_stimulus_duration_ms': max_duration_ms,
        'max_time_bins': max_time_bins,
        'channels': channels,
        'units': units,
        'total_spikes': total_spikes,
        'stimulus_durations_ms': stim_durations_ms,
        'original_stimuli_count': len(original_stimuli),
        'filtered_stimuli_count': len(filtered_stimuli),
        'stimuli_removed': len(original_stimuli) - len(filtered_stimuli),
        'synctone_mapping': synctone_mapping,
        'filter_criteria': ['Rest', 'final black', 'initial black']
    }

    base_name = Path(spike_file).stem
    array_file = f"{base_name}_4d.npz"
    metadata_file = f"{base_name}_metadata.npz"

    np.savez_compressed(array_file, array=spike_array)
    np.savez_compressed(metadata_file, **metadata)


    return spike_array, metadata

def main():
    if len(sys.argv) != 4:
        sys.exit(1)

    spike_file = sys.argv[1]
    synctones_file = sys.argv[2]
    stims_file = sys.argv[3]

    for file_path in [spike_file, synctones_file, stims_file]:
        if not os.path.exists(file_path):
            sys.exit(1)

    try:
        spike_array, metadata = generate_arrays(spike_file, synctones_file, stims_file)
    except Exception as e:
        raise

if __name__ == "__main__":
    main()
