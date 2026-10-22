import numpy as np
import json
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_raw_synthetic_data_phase as G

OUT_DIR = "population_assembly_phase_tree_noisy_datasets"
K_TRUE_GRID = [7, 10]

TYPE_BUDGET_PER_TYPE = 60            # of 80 cells per type, 60 become assembly members, 20 stay non-selective
NON_SELECTIVE_HZ_RANGE = (0.2, 1.0)

ROOT_PROB_LOW, ROOT_PROB_HIGH = 0.60, 0.95
EDGE_PROB_HIGH = 0.85                 
EDGE_PROB_LOW = 0.15
TRIAL_GAIN_JITTER_SD = 0.08            #
TUNING_FLOOR = 0.25                   


def root_prob(c):
    return float(np.interp(c, [min(G.CONTRAST_CYCLE), max(G.CONTRAST_CYCLE)],
                            [ROOT_PROB_LOW, ROOT_PROB_HIGH]))


def build_type_pools(rng):
    on_ids = rng.permutation(np.arange(0, G.N_ON))
    off_ids = rng.permutation(np.arange(G.N_ON, G.N_ON + G.N_OFF))
    bip_ids = rng.permutation(np.arange(G.N_ON + G.N_OFF, G.N_UNITS))
    return {"ON": on_ids, "OFF": off_ids, "BIP": bip_ids}


def build_assemblies(K_active, rng):
    pools = build_type_pools(rng)
    n_per_type_per_state = TYPE_BUDGET_PER_TYPE // K_active

    assembly_units = [[] for _ in range(K_active)]
    assembly_types = [[] for _ in range(K_active)]
    non_selective_units = []

    for type_name, ids in pools.items():
        assigned = ids[: n_per_type_per_state * K_active]
        leftover = ids[n_per_type_per_state * K_active:]
        non_selective_units.extend(leftover.tolist())
        for k in range(K_active):
            chunk = assigned[k * n_per_type_per_state:(k + 1) * n_per_type_per_state]
            assembly_units[k].extend(chunk.tolist())
            assembly_types[k].extend([type_name] * len(chunk))

    tree_edges = []
    for k in range(K_active):
        order = rng.permutation(len(assembly_units[k]))
        members = np.array(assembly_units[k])[order]
        types = np.array(assembly_types[k])[order]
        assembly_units[k] = members
        assembly_types[k] = types
        tree_edges.append([(int(members[i - 1]), int(members[i])) for i in range(1, len(members))])

    return assembly_units, assembly_types, tree_edges, np.array(non_selective_units), n_per_type_per_state * 3


def block_ranges(z_active, K_active):
    ranges = []
    for k in range(K_active):
        idx = np.where(z_active == k)[0]
        ranges.append((int(idx.min()), int(idx.max()) + 1))
    return ranges


def tuning_curves():
    """One canonical 50-bin phase-tuning curve per cell type, values in [0,1],
    floored so no type is ever fully unable to fire."""
    on = G.tuning_on(0.0)
    off = G.tuning_off(0.0)
    bip = G.tuning_bip(0.0)
    curves = {}
    for name, c in [("ON", on), ("OFF", off), ("BIP", bip)]:
        curves[name] = TUNING_FLOOR + (1 - TUNING_FLOOR) * c
    return curves


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


def tree_block_spikes(members, types, tcurves, p_root_trial, block_onset, b0_abs, n_block_bins, dt, rng):
    """Same conditional Bernoulli cascade as before, but each node's firing
    probability is additionally scaled by its own cell type's phase-tuning
    value at that bin's absolute position in the 50-bin cycle."""
    rows = []
    for b in range(n_block_bins):
        bin_t0 = block_onset + b * dt
        abs_bin = (b0_abs + b) % G.BINS_CYCLE

        p_r = p_root_trial * tcurves[types[0]][abs_bin]
        parent_fired = rng.random() < p_r
        if parent_fired:
            rows.append((int(members[0]), bin_t0 + rng.uniform(0, dt)))

        for i, m in enumerate(members[1:], start=1):
            base_p = EDGE_PROB_HIGH if parent_fired else EDGE_PROB_LOW
            p = base_p * tcurves[types[i]][abs_bin]
            fired = rng.random() < p
            if fired:
                rows.append((int(m), bin_t0 + rng.uniform(0, dt)))
            parent_fired = fired
    return rows


def generate_dataset(K_true, seed, out_dir):
    rng = np.random.default_rng(seed)

    K_active = K_true - 1
    assembly_units, assembly_types, tree_edges, non_selective_units, n_per_state = build_assemblies(K_active, rng)
    non_selective_hz = rng.uniform(*NON_SELECTIVE_HZ_RANGE, size=len(non_selective_units))
    tcurves = tuning_curves()

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
                p_root_base = root_prob(c)
                n_cycles = int(round(dur_s * G.HZ))
                cycle_dur = 1.0 / G.HZ
                for cyc in range(n_cycles):
                    cyc_onset = onset + cyc * cycle_dur
                    gain = np.clip(rng.normal(1.0, TRIAL_GAIN_JITTER_SD), 0.3, 1.6)
                    p_root_trial = float(np.clip(p_root_base * gain, 0.05, 0.98))
                    for k in range(K_active):
                        b0, b1 = block_bins[k]
                        block_onset = cyc_onset + b0 * G.DT
                        rows = tree_block_spikes(assembly_units[k], assembly_types[k], tcurves,
                                                  p_root_trial, block_onset, b0, b1 - b0, G.DT, rng)
                        all_spike_rows.extend(rows)

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

    prefix = f"PHASOR_synthetic_assembly_phase_tree_noisy_Ktrue{K_true}"

    stims_json = {"title": f"PHASOR_Synthetic_Assembly_Phase_Tree_Noisy_Ktrue{K_true}",
                  "description": "Tree-structured stochastic population-assembly synthetic data "
                                  "with ON/OFF/Biphasic phase-locked tuning. Each assembly mixes "
                                  "units from all three cell types. A conditional Bernoulli cascade "
                                  "(root -> chain of children, real tree, hop-distance-decaying "
                                  "correlation) gates state-selective firing; each unit's cascade "
                                  "probability is further scaled by its own cell type's phase-tuning "
                                  "curve evaluated at the bin's true phase in the 50-bin cycle. "
                                  "Per-trial multiplicative gain jitter adds trial-to-trial drive "
                                  "variability on top of the cascade's own stochasticity.",
                  "appVersion": "phasor-synthetic-assembly-phase-tree-noisy-1.0", "stimuli": stims}
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
        assembly_types=np.array(assembly_types, dtype=object),
        tree_edges=np.array(tree_edges, dtype=object),
        non_selective_units=non_selective_units,
        non_selective_hz=non_selective_hz,
        n_units_per_assembly=n_per_state,
        root_prob_low=ROOT_PROB_LOW, root_prob_high=ROOT_PROB_HIGH,
        edge_prob_high=EDGE_PROB_HIGH, edge_prob_low=EDGE_PROB_LOW,
        trial_gain_jitter_sd=TRIAL_GAIN_JITTER_SD, tuning_floor=TUNING_FLOOR,
        trial_log=trial_log, session_state_log=session_state_log,
        seed=seed,
        design="population_assembly_phase_tree_v1_noisy",
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
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = []
    for i, K_true in enumerate(K_TRUE_GRID):
        seed = 9500 + i
        d = os.path.join(OUT_DIR, f"Ktrue{K_true}")
        summary = generate_dataset(K_true, seed, d)
        manifest.append(summary)
        print(summary)
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
