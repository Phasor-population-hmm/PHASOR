import numpy as np
import json
import csv
import os

N_UNITS = 240
N_ON, N_OFF, N_BIP = 80, 80, 80
N_CHANNELS = 120
UNITS_PER_CHANNEL = 2

DT = 0.010
BINS_CYCLE = 50
HZ = 2
FFSINE_DUR_MS = 3000

BASELINE_HZ_RANGE = (1.0, 5.0)
GAIN_RANGE = (2.0, 4.0)

CONTRAST_CYCLE = [50, 60, 70, 80, 90]
N_ACTIVE_TOTAL = 75
N_C0_TOTAL = 75

BLOCK_ACTIVE_COUNTS = [19, 20, 20, 16]
REST_MS = 60000
FLASH_MS = 1040
FLASH_BODY_MS = 520
INITIAL_BLACK_MS = 10000
FINAL_BLACK_MS = 200
N_FLASHES_EACH_END = 4

K_TRUE_GRID = [3, 6, 9, 12, 15]

OUT_DIR = "synthetic_raw_datasets_phase"


def build_stim_template():
    stims = []
    stims.append({"durationMs": INITIAL_BLACK_MS, "bgColor": "black",
                   "colorTail": False, "stimType": "Solid",
                   "bodyMs": INITIAL_BLACK_MS})

    for _ in range(N_FLASHES_EACH_END):
        stims.append({"durationMs": FLASH_MS, "bgColor": "black",
                       "colorTail": False, "stimType": "Solid",
                       "bodyMs": FLASH_BODY_MS, "tailMs": FLASH_BODY_MS})

    active_seq = [CONTRAST_CYCLE[i % 5] for i in range(N_ACTIVE_TOTAL)]
    cursor = 0
    for bi, n_active in enumerate(BLOCK_ACTIVE_COUNTS):
        block_contrasts = active_seq[cursor:cursor + n_active]
        cursor += n_active
        for c in block_contrasts:
            stims.append({"durationMs": FFSINE_DUR_MS, "bgColor": "black",
                           "colorTail": False, "stimType": "FFSine",
                           "bodyMs": FFSINE_DUR_MS,
                           "c": 0, "m": 50, "hz": HZ, "delayMs": 0})
            stims.append({"durationMs": FFSINE_DUR_MS, "bgColor": "black",
                           "colorTail": False, "stimType": "FFSine",
                           "bodyMs": FFSINE_DUR_MS,
                           "c": c, "m": 50, "hz": HZ, "delayMs": 0})
        if bi < len(BLOCK_ACTIVE_COUNTS) - 1:
            for _ in range(2):
                stims.append({"durationMs": REST_MS, "bgColor": "black",
                               "colorTail": False, "stimType": "Solid",
                               "bodyMs": REST_MS})

    for _ in range(N_FLASHES_EACH_END):
        stims.append({"durationMs": FLASH_MS, "bgColor": "black",
                       "colorTail": False, "stimType": "Solid",
                       "bodyMs": FLASH_BODY_MS, "tailMs": FLASH_BODY_MS})

    stims.append({"durationMs": FINAL_BLACK_MS, "bgColor": "black",
                   "colorTail": False, "stimType": "Solid",
                   "bodyMs": FINAL_BLACK_MS})

    return stims


def compute_onset_times(stims):
    n = len(stims)
    onsets = np.zeros(n)
    for i in range(1, n):
        onsets[i] = onsets[i - 1] + stims[i - 1]["durationMs"] / 1000.0
    return onsets


OFF_START = int(np.floor(0.25 * BINS_CYCLE))
OFF_END = int(np.ceil(0.75 * BINS_CYCLE))
ON_CENTER = 0.0
OFF_CENTER = (OFF_START + OFF_END) / 2.0

TAU = np.arange(BINS_CYCLE)


def tuning_on(shift_per_bin):
    return 0.5 * (1 + np.cos(2 * np.pi * (TAU - ON_CENTER - shift_per_bin) / BINS_CYCLE))


def tuning_off(shift_per_bin):
    return 0.5 * (1 + np.cos(2 * np.pi * (TAU - OFF_CENTER - shift_per_bin) / BINS_CYCLE))


def tuning_bip(shift_per_bin):
    return 0.5 * (1 + np.cos(2 * np.pi * 2 * (TAU - ON_CENTER - shift_per_bin) / BINS_CYCLE))


def build_tuning_curves(shift_per_bin):
    tuning = np.zeros((N_UNITS, BINS_CYCLE))
    tuning[0:N_ON] = tuning_on(shift_per_bin)
    tuning[N_ON:N_ON + N_OFF] = tuning_off(shift_per_bin)
    tuning[N_ON + N_OFF:] = tuning_bip(shift_per_bin)
    return tuning


def active_phase_state_sequence(K_active):
    t = np.arange(BINS_CYCLE)
    z = (t * K_active) // BINS_CYCLE
    return z


EPS = 1e-9
ONOFF_THRESHOLD = 0.6
SNR_THRESHOLD = 0.3
GAIN_GRID = np.linspace(GAIN_RANGE[0], GAIN_RANGE[1], 5)
CONTRAST_GRID = np.array(CONTRAST_CYCLE)

SEARCH_SPREAD_CANDIDATES = np.arange(6, 40, 2)
SEARCH_SHUFFLES_PER_SPREAD = 1500


def _check_valid_full_grid_vectorized(shift_per_bin):
    f_on = tuning_on(shift_per_bin)
    f_off = tuning_off(shift_per_bin)
    f_bip = tuning_bip(shift_per_bin)

    c_frac = CONTRAST_GRID / 100.0
    A = GAIN_GRID[:, None] * c_frac[None, :]

    g_on = np.exp(A[:, :, None] * f_on[None, None, :])
    g_off = np.exp(A[:, :, None] * f_off[None, None, :])
    g_bip = np.exp(A[:, :, None] * f_bip[None, None, :])

    F_on = np.fft.fft(g_on, axis=-1)
    F_off = np.fft.fft(g_off, axis=-1)
    F_bip = np.fft.fft(g_bip, axis=-1)

    a1_on, a2_on = np.abs(F_on[..., 1]), np.abs(F_on[..., 2])
    a1_off, a2_off = np.abs(F_off[..., 1]), np.abs(F_off[..., 2])
    a1_bip, a2_bip = np.abs(F_bip[..., 1]), np.abs(F_bip[..., 2])

    on_ratio = a2_on / (a1_on + EPS)
    off_ratio = a2_off / (a1_off + EPS)
    bip_ratio = a2_bip / (a1_bip + EPS)

    Ff = np.zeros_like(F_bip)
    Ff[..., 2] = F_bip[..., 2]
    Ff[..., -2] = F_bip[..., -2]
    recon = np.real(np.fft.ifft(Ff, axis=-1))
    resid = g_bip - recon
    bip_snr = np.std(recon, axis=-1) / (np.std(resid, axis=-1) + EPS)

    on_ok = on_ratio < ONOFF_THRESHOLD
    off_ok = off_ratio < ONOFF_THRESHOLD
    bip_ok = (bip_ratio >= ONOFF_THRESHOLD) & (bip_snr >= SNR_THRESHOLD)

    on_peak = np.argmax(g_on, axis=-1)
    off_peak = np.argmax(g_off, axis=-1)
    on_peak_ok = (on_peak < OFF_START) | (on_peak >= OFF_END)
    off_peak_ok = (OFF_START <= off_peak) & (off_peak < OFF_END)

    all_ok = on_ok & off_ok & bip_ok & on_peak_ok & off_peak_ok
    return bool(np.all(all_ok))


def draw_valid_phase_shifts(K_active, z_active, rng):
    if K_active == 1:
        return np.array([0.0]), np.inf, 1, 1

    best_gap = -1
    best_shifts = None
    n_valid = 0
    n_tried = 0
    for S in SEARCH_SPREAD_CANDIDATES:
        base_shifts = np.linspace(-S / 2, S / 2, K_active)
        for _ in range(SEARCH_SHUFFLES_PER_SPREAD):
            n_tried += 1
            order = rng.permutation(K_active)
            shifts = base_shifts[order]
            shift_per_bin = shifts[z_active]
            if _check_valid_full_grid_vectorized(shift_per_bin):
                n_valid += 1
                gap = np.abs(np.diff(shifts)).min()
                if gap > best_gap:
                    best_gap = gap
                    best_shifts = shifts.copy()

    if best_shifts is None:
        raise RuntimeError(
            f"No classifier-valid phase-shift assignment found for K_active={K_active} "
            f"after {n_tried} candidates."
        )

    return best_shifts, best_gap, n_valid, n_tried


def spikes_in_epoch(rate_hz, duration_s, rng):
    if rate_hz <= 0:
        return np.array([])
    n_expected = rate_hz * duration_s
    n = rng.poisson(n_expected)
    return np.sort(rng.uniform(0, duration_s, size=n))


def spikes_in_ffsine_trial(lambda_bin_profile, rng):
    counts = rng.poisson(lambda_bin_profile)
    times = []
    for b, n in enumerate(counts):
        if n == 0:
            continue
        bin_start = b * DT
        times.append(bin_start + rng.uniform(0, DT, size=n))
    if not times:
        return np.array([])
    return np.sort(np.concatenate(times))


def generate_dataset(K_true, seed, out_dir):
    rng = np.random.default_rng(seed)

    lambda_base_hz = rng.uniform(*BASELINE_HZ_RANGE, size=N_UNITS)
    lambda_base = lambda_base_hz * DT
    gain = rng.uniform(*GAIN_RANGE, size=N_UNITS)

    K_active = K_true - 1
    z_active = active_phase_state_sequence(K_active)
    z_active_external = z_active + 1

    phase_shift, min_adj_gap, n_valid, n_tried = draw_valid_phase_shifts(K_active, z_active, rng)
    shift_per_bin = phase_shift[z_active]
    tuning = build_tuning_curves(shift_per_bin)

    stims = build_stim_template()
    onsets = compute_onset_times(stims)

    all_spike_rows = []
    trial_log = []
    session_state_log = []

    for i, s in enumerate(stims):
        onset = onsets[i]
        dur_s = s["durationMs"] / 1000.0

        if s["stimType"] == "Solid":
            for u in range(N_UNITS):
                t_rel = spikes_in_epoch(lambda_base_hz[u], dur_s, rng)
                for t in t_rel:
                    all_spike_rows.append((u, onset + t))
            session_state_log.append({
                "stim_index": i, "stim_type": "Solid", "contrast": None,
                "onset_s": float(onset), "duration_s": dur_s, "state": "baseline",
            })

        elif s["stimType"] == "FFSine":
            c = s["c"]
            if c == 0:
                for u in range(N_UNITS):
                    t_rel = spikes_in_epoch(lambda_base_hz[u], dur_s, rng)
                    for t in t_rel:
                        all_spike_rows.append((u, onset + t))
                session_state_log.append({
                    "stim_index": i, "stim_type": "FFSine", "contrast": 0,
                    "onset_s": float(onset), "duration_s": dur_s, "state": "baseline",
                })
            else:
                log_lambda_base = np.log(lambda_base)[:, None]
                lambda_cycle = np.exp(log_lambda_base + gain[:, None] * (c / 100.0) * tuning)
                n_cycles = int(round(dur_s * HZ))
                cycle_idx = np.tile(np.arange(BINS_CYCLE), n_cycles)
                lambda_rep = lambda_cycle[:, cycle_idx]

                for u in range(N_UNITS):
                    t_rel = spikes_in_ffsine_trial(lambda_rep[u], rng)
                    for t in t_rel:
                        all_spike_rows.append((u, onset + t))

                trial_log.append({
                    "stim_index": i, "contrast": c, "onset_s": float(onset),
                    "duration_s": dur_s, "n_cycles": n_cycles,
                })
                session_state_log.append({
                    "stim_index": i, "stim_type": "FFSine", "contrast": c,
                    "onset_s": float(onset), "duration_s": dur_s, "state": "active",
                    "state_sequence_per_cycle": z_active_external.tolist(),
                })

    all_spike_rows.sort(key=lambda r: r[1])

    os.makedirs(out_dir, exist_ok=True)

    stims_json = {"title": f"Synthetic_FFsine_2Hz_Ktrue{K_true}_PHASE",
                  "description": "Synthetic replica of FFsine_2Hz protocol (phase-based state design).",
                  "appVersion": "synthetic-phase-0.1", "stimuli": stims}
    with open(os.path.join(out_dir, f"synthetic_Ktrue{K_true}_stims.json"), "w") as f:
        json.dump(stims_json, f, indent=2)

    with open(os.path.join(out_dir, f"synthetic_Ktrue{K_true}_synctones.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Synctone Timestamp (s)"])
        for t in onsets[1:]:
            w.writerow([round(float(t), 4)])

    with open(os.path.join(out_dir, f"synthetic_Ktrue{K_true}_spikes.txt"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Channel", "Unit", "Timestamp"])
        for u, t in all_spike_rows:
            channel = (u // UNITS_PER_CHANNEL) + 1
            unit_in_channel = (u % UNITS_PER_CHANNEL) + 1
            w.writerow([channel, unit_in_channel, round(float(t), 4)])

    polarity = np.array([0] * N_ON + [1] * N_OFF + [2] * N_BIP)
    n_baseline_stims = sum(1 for e in session_state_log if e["state"] == "baseline")
    n_active_stims = sum(1 for e in session_state_log if e["state"] == "active")
    np.savez_compressed(
        os.path.join(out_dir, f"synthetic_Ktrue{K_true}_groundtruth.npz"),
        K_true=K_true,
        baseline_state_label=0,
        n_active_states=K_active,
        active_state_labels=np.arange(1, K_true),
        active_phase_state_sequence=z_active_external,
        phase_shift=phase_shift,
        min_adjacent_phase_gap=min_adj_gap,
        polarity=polarity,
        lambda_base_hz=lambda_base_hz, gain=gain,
        trial_log=trial_log,
        session_state_log=session_state_log,
        seed=seed,
        design="phase_shift_v2",
    )

    return {
        "K_true": K_true, "seed": seed,
        "n_stims": len(stims), "n_ffsine": len(trial_log),
        "n_baseline_stims": n_baseline_stims, "n_active_stims": n_active_stims,
        "n_spikes": len(all_spike_rows),
        "min_adjacent_phase_gap": float(min_adj_gap),
        "phase_shift_candidates_valid": n_valid,
        "phase_shift_candidates_tried": n_tried,
        "session_duration_s": float(onsets[-1] + stims[-1]["durationMs"] / 1000.0),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for i, K_true in enumerate(K_TRUE_GRID):
        seed = 5000 + i
        d = os.path.join(OUT_DIR, f"Ktrue{K_true}")
        summary = generate_dataset(K_true, seed, d)
        manifest.append(summary)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
