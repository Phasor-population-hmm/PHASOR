"""WT17 Washout 20 uM Run1 — Chow-Liu HMM hyperparameter search (9 eta/alpha jobs)."""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

eta = float(sys.argv[1])
alpha = float(sys.argv[2])

print(f"Running Chow-Liu WT17 wash20: eta={eta}, alpha={alpha}")

import numpy as np
import pandas as pd

from data_processing import *
from hmm_pkg import *
from hmm_pkg.evaluation import splitcv_time_forward

DATA_ROOT = PROJECT_ROOT / "data" / "wt17_wash20"
RUN_DIR = DATA_ROOT / "run1"
OUT_DIR = DATA_ROOT / "results_run1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STIM_JSON = DATA_ROOT / "FFsine_2Hz_stims.json"
SYNCTONES = RUN_DIR / "synctones.csv"
SPIKES = RUN_DIR / "spikes.txt"

df_cond = load_stim_json(str(STIM_JSON))
df_time = load_synctone_csv(str(SYNCTONES))

df_cond = build_analysis_windows(
    df_cond,
    df_time=df_time,
    t0=0.0,
    synctone_col="onset_sec",
)

df_sel = select_stimulus_segments(
    df_cond,
    selection_strategy="sequential_repetition",
    target_hz=2,
    contrast_levels=[50, 60, 70, 80, 90],
    n_rep=15,
)

df_train, df_rep = split_train_test_by_rep(df_sel, n_train_rep=10)
print(f"Train reps: {df_train['rep_idx'].min()}..{df_train['rep_idx'].max()}")
print(f"Test  reps: {df_rep['rep_idx'].min()}..{df_rep['rep_idx'].max()}")

df_spk = load_spike_txt(str(SPIKES))
df_spk2, neuron_list, neuron_to_idx = make_neuron_index(df_spk)
N = len(neuron_list)
print("N neurons =", N)

df_spk_labeled_train, df_spk_sel_train = assign_spikes_to_condition(df_spk2, df_train)
print("Selected spikes =", len(df_spk_sel_train))

BIN_SIZE_S = 0.010
plan_train = compute_binning_plan(df_train, bin_size_s=BIN_SIZE_S)
print("Total bins =", int(plan_train["n_bins"].sum()))

X = build_binary_matrix(
    df_spk_sel_train,
    df_train,
    plan_train,
    bin_size_s=BIN_SIZE_S,
    n_neurons=N,
)
X_train = X
print("X_train shape:", X_train.shape)


def make_chowliu_model(K, X_tr, seed):
    em = ChowLiuEmission(
        K,
        X_tr.shape[1],
        init_from_mean=X_tr.mean(axis=0),
        random_state=seed,
        eta=eta,
        alpha=alpha,
    )
    return HMMModel(K, em, random_state=seed)


K_list = [3, 6, 9, 12, 15]
fit_kwargs = dict(n_iter=30, tol=1e-4, verbose=False)

t0 = time.time()
out = splitcv_time_forward(
    X_train,
    K_list=K_list,
    make_model_fn=make_chowliu_model,
    n_folds=3,
    min_train_frac=0.4,
    random_state=42,
    seeds=[0, 1, 2, 3, 4],
    fit_kwargs=fit_kwargs,
    verbose=True,
)
print(f"CV done for eta={eta}, alpha={alpha} in {(time.time() - t0) / 60:.2f} minutes")

rows = []
for K in K_list:
    train_scores = np.array(out[K]["train"])
    val_scores = np.array(out[K]["test"])
    val_scores_str = ";".join([f"{v:.6f}" for v in val_scores])
    rows.append(
        {
            "K": K,
            "eta": eta,
            "alpha": alpha,
            "train_mean": float(train_scores.mean()),
            "val_mean": float(val_scores.mean()),
            "val_std": float(val_scores.std()),
            "val_min": float(val_scores.min()),
            "val_max": float(val_scores.max()),
            "val_cv": float(val_scores.std() / (np.abs(val_scores.mean()) + 1e-12)),
            "val_scores": val_scores_str,
            "N_neurons": int(N),
        }
    )

df = pd.DataFrame(rows)
df["rank_val_within_eta_alpha"] = df["val_mean"].rank(ascending=False, method="dense").astype(int)

print("\nTop-5 by val_mean (within this eta/alpha):")
print(df.sort_values("val_mean", ascending=False).head(5).to_string(index=False))

outname = OUT_DIR / f"chow_eta{eta}_alpha{alpha}.csv"
df.to_csv(outname, index=False)
print("Saved:", outname)
