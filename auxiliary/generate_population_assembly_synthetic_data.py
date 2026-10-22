import numpy as np
import json
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_raw_synthetic_data_phase as G

OUT_DIR = "population_assembly_synthetic_datasets"
K_TRUE_GRID = [7, 10]

ASSEMBLY_UNIT_BUDGET = 180          # units belonging to a state-defining assembly
NON_SELECTIVE_HZ_RANGE = (0.2, 1.0)  # minimal ambient rate for the remaining, non-selective units


def build_assemblies(K_active, rng):
    n_per_state = ASSEMBLY_UNIT_BUDGET // K_active
    perm = rng.permutation(G.N_UNITS)
    assembly_pool = perm[: n_per_state * K_active]
    non_selective_units = perm[n_per_state * K_active:]
    assembly_units = [assembly_pool[k * n_per_state:(k + 1) * n_per_state] for k in range(K_active)]
    return assembly_units, non_selective_units, n_per_state


def block_ranges(z_active, K_active):
    ranges = []
    for k in range(K_active):
        idx = np.where(z_active == k)[0]
        ranges.append((int(idx.min()), int(idx.max()) + 1))
    return ranges


def poisson_spikes_in_window(rate_hz, n_bins, dt, rng):
    lam_per_bin = rate_hz * dt
    counts = rng.poisson(lam_per_bin, size=n_bins)
    times = []
    for b, n in enumerate(counts):
        if n == 0:
            continue
        times.append(b * dt + rng.uniform(0, dt, size=n))
    if not times:
        return np.array([])
    return np.sort(np.concatenate(times))


def generate_dataset(K_true, seed, out_dir):
    rng = np.random.default_rng(seed)

    K_active = K_true - 1
    assembly_units, non_selective_units, n_per_state = build_assemblies(K_active, rng)
    non_selective_hz = rng.uniform(*NON_SELECTIVE_HZ_RANGE, size=len(non_selective_units))

    z_active = G.active_phase_state_sequence(K_active)
    z_active_external = z_active + 1
    block_bins = block_ranges(z_active, K_active)

    stims = G.build_stim_template()
    onsets = G.compute_onset_times(stims)

    trial_log = []
    session_state_log = []
    all_spike_rows = []

    for i, s in enumerate(stims):
        dur_s = s["durationMs"] / 1000.0
        n_bins = int(round(dur_s / G.DT))
        onset = onsets[i]

        # non-selective units: minimal ambient firing, present in every window regardless of state
        for u, hz in zip(non_selective_units, non_selective_hz):
            times = poisson_spikes_in_window(hz, n_bins, G.DT, rng)
            for t in times:
                all_spike_rows.append((int(u), onset + t))

        if s["stimType"] == "Solid":
            session_state_log.append({
                "stim_index": i, "stim_type": "Solid", "contrast": None,
                "onset_s": float(onset), "duration_s": dur_s, "state": "baseline",
            })

        elif s["stimType"] == "FFSine":
            c = s["c"]
            if c == 0:
                session_state_log.append({
                    "stim_index": i, "stim_type": "FFSine", "contrast": 0,
                    "onset_s": float(onset), "duration_s": dur_s, "state": "baseline",
                })
            else:
                n_cycles = int(round(dur_s * G.HZ))
                cycle_dur = 1.0 / G.HZ
                for cyc in range(n_cycles):
                    cyc_onset = onset + cyc * cycle_dur
                    for k in range(K_active):
                        b0, b1 = block_bins[k]
                        for b in range(b0, b1):
                            t_abs = cyc_onset + b * G.DT + G.DT / 2.0
                            for u in assembly_units[k]:
                                all_spike_rows.append((int(u), t_abs))

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

    prefix = f"PHASOR_synthetic_assembly_Ktrue{K_true}"

    stims_json = {"title": f"PHASOR_Synthetic_Assembly_Ktrue{K_true}",
                  "description": "Noiseless disjoint population-assembly synthetic data. "
                                  "Assembly units are silent outside their own state's phase "
                                  "block and fire on every bin within it. Non-selective units "
                                  "carry a minimal ambient rate throughout the session.",
                  "appVersion": "phasor-synthetic-assembly-1.0", "stimuli": stims}
    with open(os.path.join(out_dir, f"{prefix}_stims.json"), "w") as f:
        json.dump(stims_json, f, indent=2)

    with open(os.path.join(out_dir, f"{prefix}_synctones.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Synctone Timestamp (s)"])
        for t in onsets[1:]:
            w.writerow([round(float(t), 6)])

    with open(os.path.join(out_dir, f"{prefix}_spikes.txt"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Channel", "Unit", "Timestamp"])
        for u, t in all_spike_rows:
            channel = (u // G.UNITS_PER_CHANNEL) + 1
            unit_in_channel = (u % G.UNITS_PER_CHANNEL) + 1
            w.writerow([channel, unit_in_channel, round(float(t), 6)])

    polarity = np.array([0] * G.N_ON + [1] * G.N_OFF + [2] * G.N_BIP)

    np.savez_compressed(
        os.path.join(out_dir, f"{prefix}_groundtruth.npz"),
        K_true=K_true, baseline_state_label=0, n_active_states=K_active,
        active_state_labels=np.arange(1, K_true),
        active_phase_state_sequence=z_active_external,
        polarity=polarity,
        assembly_units=np.array(assembly_units, dtype=object),
        non_selective_units=non_selective_units,
        non_selective_hz=non_selective_hz,
        n_units_per_assembly=n_per_state,
        trial_log=trial_log, session_state_log=session_state_log,
        seed=seed,
        design="population_assembly_v1",
    )

    n_baseline_s = sum(s["durationMs"] for s in stims if s["stimType"] == "Solid") / 1000.0
    n_baseline_s += sum(s["durationMs"] for s in stims if s["stimType"] == "FFSine" and s["c"] == 0) / 1000.0
    n_active_s = sum(s["durationMs"] for s in stims if s["stimType"] == "FFSine" and s["c"] != 0) / 1000.0
    session_s = float(onsets[-1] + stims[-1]["durationMs"] / 1000.0)

    return {
        "K_true": K_true, "seed": seed, "n_active_states": K_active,
        "n_units_per_assembly": int(n_per_state), "n_non_selective_units": int(len(non_selective_units)),
        "n_stims": len(stims), "n_ffsine": len(trial_log),
        "n_spikes": len(all_spike_rows),
        "session_duration_s": session_s,
        "baseline_time_s": n_baseline_s,
        "active_time_s": n_active_s,
        "baseline_fraction": n_baseline_s / session_s,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for i, K_true in enumerate(K_TRUE_GRID):
        seed = 9300 + i
        d = os.path.join(OUT_DIR, f"Ktrue{K_true}")
        summary = generate_dataset(K_true, seed, d)
        manifest.append(summary)
        print(summary)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
