#!/usr/bin/env python

import argparse
import h5py
import numpy as np
from scipy.signal import savgol_filter, find_peaks, resample
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from h5_tools import get_date_and_duration, valid_filename
from plot_data import plot_data
from utils import centered_moving_average, find_square_wave_steps
import os, csv


def find_audio_channel(file_path, desiredLabelStr=""):
    """
    Search through all analog streams to find a channel containing the desired label string.
    If no label is provided, it defaults to returning the first analog channel found.
    Returns the group path and channel number if found.
    """
    with h5py.File(file_path, "r") as f:
        recording_group = "/Data/Recording_0/AnalogStream"

        if recording_group not in f:
            raise Exception(f"Recording group {recording_group} not found in file")

        analog_stream_group = f[recording_group]

        for stream_name in analog_stream_group.keys():
            if stream_name.startswith("Stream_"):
                stream_path = f"{recording_group}/{stream_name}"

                try:
                    group = f[stream_path]

                    if "InfoChannel" not in group or "ChannelData" not in group:
                        continue

                    labelRaw = group.attrs.get("Label", "")
                    if isinstance(labelRaw, bytes):
                        labelStr = labelRaw.decode("utf-8")
                    else:
                        labelStr = str(labelRaw)

                    if "Analog Data" not in labelStr:
                        continue

                    info = group["InfoChannel"][()]
                    nAnalogChannels = info.shape[0]

                    for i in range(nAnalogChannels):
                        channel_label = info["Label"][i].decode("utf-8")
                        if (
                            desiredLabelStr == ""
                            or desiredLabelStr.lower() in channel_label.lower()
                        ):
                            return stream_path, i

                except Exception as e:
                    continue

        raise Exception(
            f"Unable to find any channel with label containing '{desiredLabelStr}' in any analog stream"
        )


def get_analog_data(file_path, desiredLabelStr=""):
    stream_path, labeledChannelNumber = find_audio_channel(file_path, desiredLabelStr)

    with h5py.File(file_path, "r") as f:
        group = f[stream_path]

        info = group["InfoChannel"][()]
        label = info["Label"][labeledChannelNumber].decode("utf-8")

        microseconds_between_samples = info["Tick"][labeledChannelNumber]
        data_rate = round(1_000_000 / microseconds_between_samples)

        analog_data = group["ChannelData"][labeledChannelNumber, :]
        duration_seconds = len(analog_data) / data_rate
        return (
            analog_data.reshape(-1, 1),
            data_rate,
        )


def locate_synctones(file_path, do_plot=False):
    try:
        audio_data, _ = get_analog_data(file_path, "audio")
    except Exception as e:
        audio_data, _ = get_analog_data(file_path)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(audio_data)
    centered = audio_data - np.mean(audio_data)
    squared = scaler.fit_transform(centered**2)
    smoothed = scaler.fit_transform(
        savgol_filter((centered**2).flatten(), 1800, 3).reshape(audio_data.shape)
    )
    peak_indices, _ = find_peaks(smoothed.flatten(), height=0.5, distance=2000)
    if do_plot:
        min = peak_indices[0] - 2000
        max = peak_indices[0] + 2000
        peaks_in_range = peak_indices[(peak_indices > min) & (peak_indices < max)]
        plot_data(
            [scaled, squared, smoothed],
            min,
            max,
            ["scaled", "squared", "smoothed"],
            peaks_in_range - min,
        )

    diffs = (10_000 - np.diff(peak_indices)) / 10
    mean_value = np.mean(diffs)
    min_value = np.min(diffs)
    max_value = np.max(diffs)
    std_dev = np.std(diffs)
    return peak_indices / 10_000


def main():
    parser = argparse.ArgumentParser(description="Process a single HDF5 (.h5) file.")
    parser.add_argument("filename", type=valid_filename, help="Path to the .h5 file")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot start of data")

    args = parser.parse_args()
    get_date_and_duration(args.filename)

    np.set_printoptions(suppress=True)

    synctone_timestamps = locate_synctones(args.filename, args.plot)

    base_filename = os.path.splitext(args.filename)[0]
    csv_filename = f"{base_filename}_synctones.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Synctone Timestamp (s)"])
        for sync in synctone_timestamps:
            writer.writerow([sync])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pass
