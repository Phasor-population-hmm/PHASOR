# PHASOR

PHASOR is the program for the retinal recordings in this paper. It runs on your computer. Nothing is uploaded.

The paper data is a separate ZIP. Open `index.html` in that folder to look at the finished figures. Use this repository when you want to run the analysis.

## What you need

Python 3.10 or newer, and Node.js 20 or newer.

The hidden Markov model uses 3 cores. Leave the computer on while that fit is running. Only one fit or analysis runs at a time.

## 1. Clone the repository

```bash
git clone https://github.com/Phasor-population-hmm/PHASOR.git
cd PHASOR
```

## 2. Install Python

On Linux or Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of `source .venv/bin/activate`. Then run the same `pip install` line.

## 3. Install and start the website

```bash
cd phasor_web
npm install
npm run dev
```

If `npm install` stops with a symlink error, run `npm install --no-bin-links` and then `npm run dev`.

Open http://127.0.0.1:3000 in a browser.

Leave that terminal open. Closing it stops the website. A hidden Markov model that has already started keeps running.

## 4. Use the paper data

Unzip the paper data so the two folders sit next to each other:

```text
PHASOR/
PHASOR_paper_data/
```

`PHASOR` is this repository. `PHASOR_paper_data` is the unzipped paper folder. The folder name has to stay `PHASOR_paper_data`.

On this layout the website finds the recordings by itself. If the paper folder is somewhere else, set the path before you start the website:

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

Choose a recording in the bar at the top. PHASOR fills in the matching files from that recording. Choose Other files when you want a different dataset. That clears the filled-in paths so you can browse to your own files.

## 5. What the pages do

Overview tells you what the steps are.

Pipeline runs the steps in order.

Pre-processing reads the recording and writes the spike array, the statistics, and the rasters.

Processing filters the recording by contrast, clusters the cells, and writes the label index maps.

Fourier classification and sustained or transient classification use the filtered data.

Hidden Markov Model uses the saved result when the paper data already has one. To fit a new model, press start. The fit runs in a terminal on this computer and needs 3 cores. You can leave the page. The bar at the bottom shows that it is still running. Terminate stops it. When it finishes, a box names the job. Close that box with the X in the corner.

Post-HMM plots runs one analysis at a time. Press the analysis. It shows that it is running. The figures go into one folder for that analysis.

Each recording gets one output folder. The path shown on screen is `phasor_output/runs/...`. The files are on your computer, under the PHASOR folder, unless you set another output folder:

```bash
export PHASOR_OUTPUT_ROOT=/path/to/output
```

Set that in the same terminal, before `npm run dev`.

## HMM settings

`phasor_web/hmm.config.json` points at the HMM code in `mode_project`. Leave `python` empty to use the Python environment from step 2. `max_parallel_searches` is 3. That is the core limit.

## What is not in this repository

The large recording files and the finished HMM bundles are in the paper data ZIP. This repository has the program, the spike tables, the stimulus files, and the synctone tables used to start a run.

Results you create are written to `phasor_output/`. That folder is not part of the repository.

## License

MIT. See LICENSE.
