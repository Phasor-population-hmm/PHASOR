# PHASOR

Population HMM Analysis of Stimulus-locked Output in the Retina.

PHASOR is a local application for analysis of multielectrode array recordings of retinal ganglion cell activity. It performs stimulus alignment, contrast filtering, clustering, cell-type classification, hidden Markov model fitting, Viterbi decoding, and post-HMM reporting. All computation is executed on the local machine.

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

The application is served at http://127.0.0.1:3000. The terminal session must remain active while the interface is in use. A hidden Markov model fit that has already started continues if the browser is closed.

On a file system that does not permit symbolic links, install the Node dependencies with:

```bash
npm install --no-bin-links
npm run dev
```

## Data directory

Recordings distributed with the paper are provided separately from this repository. Place that directory beside the cloned repository and retain the directory name `PHASOR_paper_data`:

```text
PHASOR/
PHASOR_paper_data/
```

If the data directory is stored elsewhere, set its absolute path before starting the application:

```bash
export PHASOR_PAPER_DATA=/path/to/PHASOR_paper_data
cd phasor_web
npm run dev
```

On Windows PowerShell:

```powershell
$env:PHASOR_PAPER_DATA="C:\path\to\PHASOR_paper_data"
npm run dev
```

After the data directory is resolved, the recording selector loads the `.h5` file, spike table, synctone table, stimulus file, and HMM bundle for the selected recording. The Other files option clears those assignments and accepts a user-specified dataset.

This repository includes the spike tables, stimulus files, and synctone tables used to initiate a run. The large `.h5` recordings and fitted HMM bundles are supplied in the data directory.

## Analysis interface

Overview summarizes the analysis sequence.

Pipeline executes the stages in order. Pre-processing writes the spike array, summary statistics, and raster plots. Processing filters each contrast, clusters units, and writes label-index maps. Fourier classification and sustained/transient classification are then run on the filtered data. Each stage writes into `phasor_output/runs/<recording>/`.

Hidden Markov Model loads an existing bundle when one is present for the selected recording. A new fit is started from that page and runs as a local terminal process. The fit requires three CPU cores. Leaving the page does not interrupt it. Terminate stops the active fit. Completion is reported in a dialog, which is dismissed from its close control.

Post-HMM plots runs one analysis at a time. Each analysis is written directly into its own directory under the recording output folder. The analyses are firing rate, Subtypes MEA Spread, subtype maps, sustained/transient polar plots, polar plots by repetition and cycle, polar activity, dominant modes, mode summary plots, and interactive HMM Viterbi decoding.

Displayed output paths use the form `phasor_output/runs/...`. To write results elsewhere, set `PHASOR_OUTPUT_ROOT` before starting the application:

```bash
export PHASOR_OUTPUT_ROOT=/path/to/output
```

## HMM configuration

HMM settings are stored in `phasor_web/hmm.config.json`.

`project_dir` is the path to the HMM engine. The default is `mode_project`.

`python` is the interpreter used for model fits. Leave it empty to use the virtual environment created above.

`max_parallel_searches` is the number of concurrent covariance searches. The default is 3.

## License

Released under the MIT License. See `LICENSE`.
