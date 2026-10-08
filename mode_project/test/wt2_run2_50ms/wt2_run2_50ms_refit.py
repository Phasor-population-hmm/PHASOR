# ============================================================
# wt2_run2_50ms_refit.py
# Copy of test/wt13/wt13_run1_refit.py — paths only changed for WT17 wash20 run1.
# ============================================================

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_processing import *
from hmm_pkg import *
from hmm_pkg.inference import forward_backward_log
from hmm_pkg.utils import save_chowliu_hmm

# ============================================================
# USER SETTINGS — WT2 Run2 50 ms
# ============================================================
DATA_ROOT = PROJECT_ROOT / "data" / "wt2_run2_50ms"
RUN_DIR = DATA_ROOT / "run1"

BASE_DATA_DIR = str(RUN_DIR)
STIM_JSON_PATH = str(DATA_ROOT / "FFsine_2Hz_stims.json")
SYNCTONE_CSV_PATH = str(RUN_DIR / "synctones.csv")
SPIKE_TXT_PATH = str(RUN_DIR / "spikes.txt")

SEARCH_CSV_DIR = str(DATA_ROOT / "results_run1")
CSV_FILES = [
    "chow_eta0.005_alpha0.1.csv",
    "chow_eta0.005_alpha0.5.csv",
    "chow_eta0.005_alpha1.0.csv",
    "chow_eta0.03_alpha0.1.csv",
    "chow_eta0.03_alpha0.5.csv",
    "chow_eta0.03_alpha1.0.csv",
    "chow_eta0.1_alpha0.1.csv",
    "chow_eta0.1_alpha0.5.csv",
    "chow_eta0.1_alpha1.0.csv",
]
SEARCH_CSV_PATHS = [os.path.join(SEARCH_CSV_DIR, f) for f in CSV_FILES]

OUTPUT_DIR = SEARCH_CSV_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# LOAD df_cond & df_time
# ============================================================
if not os.path.exists(STIM_JSON_PATH):
    raise FileNotFoundError(f"Stim JSON not found: {STIM_JSON_PATH}")
if not os.path.exists(SYNCTONE_CSV_PATH):
    raise FileNotFoundError(f"Synctone CSV not found: {SYNCTONE_CSV_PATH}")

df_cond = load_stim_json(STIM_JSON_PATH)
df_time = load_synctone_csv(SYNCTONE_CSV_PATH)
print("Loaded df_cond:", df_cond.shape)
print("Loaded df_time:", df_time.shape)

# ============================================================
# BUILD X_train / X_test
# ============================================================
if not os.path.exists(SPIKE_TXT_PATH):
    raise FileNotFoundError(f"Spike txt not found: {SPIKE_TXT_PATH}")

BIN_SIZE_S = 0.050

df_cond = build_analysis_windows(
    df_cond, df_time=df_time, t0=0.0, synctone_col="onset_sec"
)

df_sel = select_stimulus_segments(
    df_cond,
    selection_strategy="sequential_repetition",
    target_hz=2,
    contrast_levels=[50, 60, 70, 80, 90],
    n_rep=15,
)

df_train, df_rep = split_train_test_by_rep(df_sel, n_train_rep=10)

df_spk = load_spike_txt(SPIKE_TXT_PATH)
df_spk2, neuron_list, neuron_to_idx = make_neuron_index(df_spk)
N = len(neuron_list)
print("N neurons =", N)

df_spk_labeled_train, df_spk_sel_train = assign_spikes_to_condition(df_spk2, df_train)
plan_train = compute_binning_plan(df_train, bin_size_s=BIN_SIZE_S)
X_train = build_binary_matrix(
    df_spk_sel_train, df_train, plan_train, bin_size_s=BIN_SIZE_S, n_neurons=N
)
print("X_train shape:", X_train.shape)

df_spk_labeled_test, df_spk_sel_test = assign_spikes_to_condition(df_spk2, df_rep)
plan_test = compute_binning_plan(df_rep, bin_size_s=BIN_SIZE_S)
X_test = build_binary_matrix(
    df_spk_sel_test, df_rep, plan_test, bin_size_s=BIN_SIZE_S, n_neurons=N
)
print("X_test shape:", X_test.shape)

# ============================================================
# LOAD SEARCH CSVs
# ============================================================
for p in SEARCH_CSV_PATHS:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing search CSV: {p}")

df_search = pd.concat((pd.read_csv(p) for p in SEARCH_CSV_PATHS), ignore_index=True)


def parse_scores(s):
    return np.array([float(x) for x in str(s).split(";") if x.strip() != ""], dtype=float)


df_search["_scores"] = df_search["val_scores"].apply(parse_scores)
df_search["val_mean"] = df_search["_scores"].apply(np.mean)
df_search["val_std"] = df_search["_scores"].apply(np.std)

rows = []
for K, g in df_search.groupby("K"):
    g_sorted = g.sort_values(["val_mean", "val_std"], ascending=[False, True]).reset_index(
        drop=True
    )
    rows.append(g_sorted.iloc[0])

best_per_K = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
print("\nBest row per K:")
print(best_per_K[["K", "eta", "alpha", "val_mean", "val_std"]])

# ============================================================
# PLOTS
# ============================================================
best_per_K = best_per_K.sort_values("K").reset_index(drop=True)
K = best_per_K["K"].to_numpy()
metric = "val_mean"
y = best_per_K[metric].to_numpy()

gain = np.empty_like(y, dtype=float)
gain[:] = np.nan
gain[1:] = y[1:] - y[:-1]

dK = np.empty_like(K, dtype=float)
dK[:] = np.nan
dK[1:] = (K[1:] - K[:-1]).astype(float)
gain_per1K = gain / dK

plt.figure()
yerr = best_per_K["val_std"].to_numpy()
plt.errorbar(K, y, yerr=yerr, marker="o", capsize=4)
plt.xlabel("K")
plt.ylabel(metric)
plt.ylim(-80, -30)
plt.title(f"{metric} vs K (best hyperparam per K)")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIG_DIR, "metric_vs_K.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(K, gain, marker="o")
plt.axhline(0, linewidth=1)
plt.xlabel("K")
plt.ylabel(f"Δ {metric} (vs previous K)")
plt.title(f"Marginal gain of {metric} as K increases")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIG_DIR, "gain_curve.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(K, gain_per1K, marker="o")
plt.axhline(0, linewidth=1)
plt.xlabel("K")
plt.ylabel(f"Δ {metric} per +1 K")
plt.title(f"Normalized marginal gain of {metric}")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIG_DIR, "gain_per1K.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved plots to:", FIG_DIR)

summary = best_per_K[["K", "eta", "alpha", "val_mean", "val_std"]].copy()
summary["gain_val_mean"] = np.nan
summary.loc[1:, "gain_val_mean"] = (
    summary["val_mean"].to_numpy()[1:] - summary["val_mean"].to_numpy()[:-1]
)
print("\n===== Summary table for PPT =====")
print(summary)


def select_K_cliff_plateau(best_per_K, y_col="val_mean", x_col="K", eps=1e-12):
    dfk = best_per_K.sort_values(x_col).reset_index(drop=True)
    K_vals = dfk[x_col].to_numpy(dtype=int)
    y_vals = dfk[y_col].to_numpy(dtype=float)
    if len(K_vals) == 0:
        raise ValueError("best_per_K is empty.")
    if len(K_vals) == 1:
        return int(K_vals[0])
    if len(K_vals) == 2:
        return int(K_vals[np.argmax(y_vals)])
    gains = np.diff(y_vals)
    if np.all(np.abs(gains - gains[0]) <= eps):
        return int(K_vals[np.argmax(y_vals)])
    q20 = np.quantile(gains, 0.20)
    if gains[-1] <= q20 + eps:
        return int(K_vals[-2])
    tail_level = float(np.median(gains[-2:]))
    plateau_scale = max(1.5 * tail_level, eps)
    for i in range(len(gains) - 1):
        if np.max(gains[i:]) <= plateau_scale + eps:
            return int(K_vals[i])
    return int(K_vals[np.argmax(y_vals)])


K_star = select_K_cliff_plateau(best_per_K, y_col="val_mean", x_col="K")
print("\nSelected K_star =", K_star)

row = best_per_K[best_per_K["K"] == K_star].iloc[0]
eta_star = float(row["eta"])
alpha_star = float(row["alpha"])
print("Refit with:", K_star, eta_star, alpha_star)

results = []
models = []
T, N = X_train.shape
init_from_mean = X_train.mean(axis=0)

for seed in range(5):
    em = ChowLiuEmission(
        K_star, N, init_from_mean=init_from_mean, random_state=seed,
        eta=eta_star, alpha=alpha_star,
    )
    model = HMMModel(K_star, em, random_state=seed)
    model.fit(X_train, n_iter=60, tol=5e-4, verbose=True)
    log_B = model.emission.log_B(X_train)
    gamma, xi, logL, _, _ = forward_backward_log(log_B, model.pi, model.A)
    logL_per_bin = float(logL) / float(T)
    results.append((seed, logL_per_bin))
    models.append(model)
    print(f"Seed {seed}: logL_per_bin = {logL_per_bin:.6f}")

best_idx = int(np.argmax([r[1] for r in results]))
best_model = models[best_idx]
best_seed = results[best_idx][0]
print("\nBest seed:", best_seed)


def compute_logL_per_bin(model, X):
    T_local = X.shape[0]
    log_B = model.emission.log_B(X)
    gamma, xi, logL, _, _ = forward_backward_log(log_B, model.pi, model.A)
    return float(logL), float(logL) / float(T_local)


train_logL, train_logL_per_bin = compute_logL_per_bin(best_model, X_train)
test_logL, test_logL_per_bin = compute_logL_per_bin(best_model, X_test)
generalization_gap = train_logL_per_bin - test_logL_per_bin

print("\n===== GENERALIZATION =====")
print(f"Train logL total   = {train_logL:.6f}")
print(f"Train logL/bin     = {train_logL_per_bin:.6f}")
print(f"Test  logL total   = {test_logL:.6f}")
print(f"Test  logL/bin     = {test_logL_per_bin:.6f}")
print(f"Gap (train-test)   = {generalization_gap:.6f}")

gen_df = pd.DataFrame([{
    "K_star": K_star,
    "eta_star": eta_star,
    "alpha_star": alpha_star,
    "best_seed": best_seed,
    "train_T": int(X_train.shape[0]),
    "test_T": int(X_test.shape[0]),
    "N": int(N),
    "train_logL": train_logL,
    "train_logL_per_bin": train_logL_per_bin,
    "test_logL": test_logL,
    "test_logL_per_bin": test_logL_per_bin,
    "generalization_gap": generalization_gap,
}])
GEN_CSV_PATH = os.path.join(OUTPUT_DIR, "generalization_summary.csv")
gen_df.to_csv(GEN_CSV_PATH, index=False)
print("Saved generalization summary to:", GEN_CSV_PATH)

OUT_MODEL_PATH = os.path.join(
    OUTPUT_DIR, f"chowliu_K{K_star}_eta{eta_star}_alpha{alpha_star}.npz"
)
save_chowliu_hmm(
    best_model,
    OUT_MODEL_PATH,
    N_MODES=K_star,
    N=N,
    eta=eta_star,
    alpha=alpha_star,
    random_state=best_seed,
    n_iter=60,
)
print("\nSaved model to:", OUT_MODEL_PATH)
