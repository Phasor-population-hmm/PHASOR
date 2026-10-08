import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
import matplotlib.pyplot as plt
import numpy as np


def plot_data(arrays, minX=None, maxX=None, labels=None, peaks=None):
    for index, yfull in enumerate(arrays):
        y = yfull[minX:maxX] if minX is not None and maxX is not None else yfull
        x = np.arange(0, len(y))
        plt.plot(x, y, label=labels[index] if labels else None, alpha=0.4)

    if peaks is not None:
        plt.plot(peaks, y[peaks], "rx", markersize=15, label="Peaks")

    if labels:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
