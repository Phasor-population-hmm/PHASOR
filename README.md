# PHASOR

Population HMM Analysis of Stimulus-locked Output in the Retina.

PHASOR is a local application for multielectrode array recordings of retinal ganglion cell activity. It performs stimulus alignment, contrast filtering, clustering, cell-type classification, hidden Markov model fitting, Viterbi decoding, and post-HMM reporting. Computation runs on the local machine.

## Requirements

- Python 3.10 or newer
- Node.js 20 or newer
- Three CPU cores during hidden Markov model fitting

One pipeline job or model fit may run at a time.

## Installation

Clone the repository:

```bash
git clone https://github.com/Phasor-population-hmm/PHASOR.git
cd PHASOR
```

Create the Python environment and install the analysis dependencies. On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`, then run `pip install -r requirements.txt`.

Install and start the web application:

```bash
cd phasor_web
npm install
npm run dev
```

The application is served at http://127.0.0.1:3000. Keep the terminal session running while the interface is in use. A hidden Markov model fit that has already started continues if the browser is closed.

On a file system that does not permit symbolic links:

```bash
npm install --no-bin-links
npm run dev
```

## Included recordings

Sample recordings are stored in `example_data/`. The recording list is `example_data/catalog.json`.

Each recording directory contains:

- `spikes.txt`, the spike-time table
- `synctones.csv`, the stimulus-onset table
- `stims.json`, the stimulus description
- `array_4d.npz`, the aligned spike array

After the application starts, select a recording in the top bar. Those four files are assigned automatically, and the output directories under `phasor_output/runs/<recording>/` are created for that recording.

Other files clears the assigned paths and accepts a different dataset. Select each input with the file browser on the relevant stage.

## Run a recording

Work through Pipeline in this order.

1. Processor builds a 4D spike array from `spikes.txt`, `synctones.csv`, and `stims.json`. The included `array_4d.npz` is the array for the selected recording, so this stage can be skipped when that file is used.
2. Population stats reads `array_4d.npz`, `stims.json`, and `spikes.txt`, and writes tables to `phasor_output/runs/<recording>/01_preprocessing/stats`.
3. Raster sequence plots reads `array_4d.npz` and `stims.json`, and writes figures to `phasor_output/runs/<recording>/01_preprocessing/rasters`.
4. Filter by contrast reads the recording directory and `stims.json`, and writes contrast folders to `phasor_output/runs/<recording>/02_processing/filtered`.
5. UMAP + OPTICS clustering reads `02_processing/filtered` and clusters each contrast folder in place.
6. Label index maps read the recording array and those contrast folders, and write CSVs under `phasor_output/runs/<recording>/03_labels`. Fourier cell classification and sustained/transient classification read the clustered contrast and write under `phasor_output/runs/<recording>/04_fourier/<contrast>`.

Locate synctones requires an MEA `.h5` file. The included recordings already provide `synctones.csv`, so that stage is used only when a new `.h5` recording is supplied.

## Hidden Markov model

Open Hidden Markov Model. The stage uses the spike table, synctone table, and stimulus file for the selected recording.

Start runs the fit in a local terminal process. The fit uses three CPU cores. Only one fit or analysis can run at a time. Leaving the page does not stop the fit. Terminate ends it. Completion is reported in a dialog.

The result is `bundle.npz`, written under `phasor_output/runs/<recording>/05_hmm`.

Settings are in `phasor_web/hmm.config.json`.

`project_dir` is the HMM engine. The default is `mode_project`.

`python` is the interpreter for model fits. Leave it empty to use the virtual environment created during installation.

`max_parallel_searches` is the number of concurrent covariance searches. The default is 3.

## Post-HMM plots

Open Post-HMM plots after a `bundle.npz` is available. Each analysis writes its files directly into one directory under `phasor_output/runs/<recording>/06_analysis/`.

The analyses are firing rate, Subtypes MEA Spread, subtype maps, sustained/transient polar plots, polar plots by repetition and cycle, polar activity, dominant modes, mode summary plots, and interactive HMM Viterbi decoding.

Output paths are shown as `phasor_output/runs/...`. To use another output location, set it before starting the application:

```bash
export PHASOR_OUTPUT_ROOT=/path/to/output
```

## License

A license has not been selected. Until one is chosen, all rights are reserved. See `LICENSE`.
