#!/usr/bin/env python3
"""
cluster_Viz.py - Simple cluster visualization using Stage 1 & Stage 2 results


Usage:
python cluster_Viz.py --umap_components 15 --umap_neighbors 20 --min_samples 50 --min_cluster_size 6 --xi 0.05
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from typing import List, Tuple, Optional
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy import stats
from sklearn.decomposition import PCA

from data_processor import DataProcessor


def run_final_clustering(X: np.ndarray, umap_components: int, umap_neighbors: int,
                        min_samples: int, min_cluster_size: int, xi: float,
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Run clustering with best parameters from Stage 1 & 2"""


    embed = (os.environ.get("GLIA_EMBEDDING", "").strip().lower() or ("pca" if sys.platform == "darwin" else "umap"))
    if sys.platform == "darwin" and embed == "umap":
        embed = "pca"

    if embed == "umap":
        import umap  # noqa: PLC0415

        reducer = umap.UMAP(
            n_components=umap_components,
            n_neighbors=umap_neighbors,
            min_dist=0.05,
            metric="correlation",
            random_state=random_state,
        )
        embedding = reducer.fit_transform(X)

        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=umap_neighbors,
            min_dist=0.05,
            metric="correlation",
            random_state=random_state,
        )
        embedding_2d = reducer_2d.fit_transform(X)
    else:
        reducer = PCA(n_components=int(umap_components))
        embedding = reducer.fit_transform(X)
        reducer_2d = PCA(n_components=2)
        embedding_2d = reducer_2d.fit_transform(X)

    optics = OPTICS(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        xi=xi,
        metric='euclidean'
    )
    labels = optics.fit_predict(embedding)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])
    n_noise = np.sum(labels == -1)


    return labels, embedding_2d


def plot_umap_2d(embedding_2d: np.ndarray, labels: np.ndarray, genotype_labels: List[str], output_dir: str):
    """Plot UMAP 2D projection colored by clusters"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            ax1.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                       c='gray', alpha=0.6, s=20, label='Noise')
        else:
            ax1.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                       c=[colors[i]], alpha=0.8, s=30,
                       label=f'Cluster {label} (n={np.sum(mask)})')

    ax1.set_title('UMAP Projection Colored by Clusters', fontsize=14)
    ax1.set_xlabel('UMAP Component 1')
    ax1.set_ylabel('UMAP Component 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    unique_genotypes = list(set(genotype_labels))
    genotype_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, genotype in enumerate(unique_genotypes):
        mask = [g == genotype for g in genotype_labels]
        ax2.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=genotype_colors[i % len(genotype_colors)],
                   alpha=0.7, s=25, label=f'{genotype} (n={np.sum(mask)})')

    ax2.set_title('UMAP Projection by Label', fontsize=14)
    ax2.set_xlabel('UMAP Component 1')
    ax2.set_ylabel('UMAP Component 2')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('UMAP 2D Projections', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_2d_projections.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _validate_trace_stat(trace_stat: str) -> str:
    ts = (trace_stat or "").strip().lower()
    if ts not in {"mean", "median"}:
        raise ValueError(f"trace_stat must be 'mean' or 'median' (got {trace_stat!r})")
    return ts


def plot_cluster_average_traces(
    X: np.ndarray,
    labels: np.ndarray,
    genotype_labels: List[str],
    output_dir: str,
    trace_stat: str = "mean",
):
    """Plot cluster summary traces (mean or median) with an uncertainty band."""
    trace_stat = _validate_trace_stat(trace_stat)

    clusters_only = sorted([l for l in np.unique(labels) if l != -1])
    n_clusters = len(clusters_only)

    if n_clusters == 0:
        return

    fig, axes = plt.subplots(n_clusters, 1, figsize=(16, 4*n_clusters))
    if n_clusters == 1:
        axes = [axes]

    cluster_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    time_ms = np.arange(X.shape[1]) * 10

    for i, cluster_id in enumerate(clusters_only):
        mask = labels == cluster_id
        cluster_data = X[mask]
        cluster_genotypes = np.array(genotype_labels)[mask]

        ax = axes[i]

        cluster_color = cluster_colors[i % 10]

        for j in range(min(20, len(cluster_data))):
            ax.plot(time_ms, cluster_data[j], color='darkgray', alpha=0.6, linewidth=0.8)

        n_samples = int(len(cluster_data))
        if n_samples == 0:
            continue

        if trace_stat == "mean":
            center = cluster_data.mean(axis=0)
            std_trace = cluster_data.std(axis=0)
            if n_samples > 1:
                se_trace = std_trace / np.sqrt(n_samples)
                band = stats.t.ppf(0.975, n_samples - 1) * se_trace
            else:
                band = np.zeros_like(center)
            band_low = center - band
            band_high = center + band
            band_label = "95% CI"
            center_label = f"Cluster {cluster_id} Mean"
        else:
            center = np.median(cluster_data, axis=0)
            band_low = np.percentile(cluster_data, 25, axis=0)
            band_high = np.percentile(cluster_data, 75, axis=0)
            band_label = "IQR (25–75%)"
            center_label = f"Cluster {cluster_id} Median"

        ax.fill_between(time_ms, band_low, band_high, alpha=0.3, color=cluster_color, label=band_label)
        ax.plot(time_ms, center, color=cluster_color, linewidth=3.5, label=center_label)

        unique_in_cluster = list(set(cluster_genotypes.tolist()))
        if len(unique_in_cluster) >= 2:
            sub_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            for g_idx, genotype in enumerate(sorted(unique_in_cluster)):
                genotype_mask = cluster_genotypes == genotype
                count = int(np.sum(genotype_mask))
                if count > 1:
                    genotype_data = cluster_data[genotype_mask]
                    if trace_stat == "mean":
                        genotype_center = genotype_data.mean(axis=0)
                        genotype_std = genotype_data.std(axis=0)
                        genotype_se = genotype_std / np.sqrt(count)
                        genotype_band = stats.t.ppf(0.975, count - 1) * genotype_se
                        genotype_low = genotype_center - genotype_band
                        genotype_high = genotype_center + genotype_band
                    else:
                        genotype_center = np.median(genotype_data, axis=0)
                        genotype_low = np.percentile(genotype_data, 25, axis=0)
                        genotype_high = np.percentile(genotype_data, 75, axis=0)
                    color = sub_colors[g_idx % len(sub_colors)]
                    ax.fill_between(time_ms, genotype_low, genotype_high, alpha=0.2, color=color)
                    ax.plot(time_ms, genotype_center, color=color, linewidth=2,
                           linestyle='--', label=f'{genotype} (n={count})')

        ax.set_title(f'Cluster {cluster_id} (n={len(cluster_data)} samples)',
                    fontsize=14, fontweight='bold', color=cluster_color)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Z-score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')

    plt.suptitle('', fontsize=18, fontweight='bold')
    plt.tight_layout()
    out_name = "cluster_average_traces.png" if trace_stat == "mean" else "cluster_median_traces.png"
    plt.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches='tight')
    plt.close()


def plot_cluster_traces_only(
    X: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    max_traces_per_cluster: int = 50,
):
    """Plot individual cell traces only, colored by cluster. No mean/median overlay."""
    clusters_only = sorted([l for l in np.unique(labels) if l != -1])
    n_clusters = len(clusters_only)
    if n_clusters == 0:
        return

    fig, axes = plt.subplots(n_clusters, 1, figsize=(16, 4 * n_clusters))
    if n_clusters == 1:
        axes = [axes]

    cluster_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    time_ms = np.arange(X.shape[1]) * 10

    for i, cluster_id in enumerate(clusters_only):
        mask = labels == cluster_id
        cluster_data = X[mask]
        ax = axes[i]
        color = cluster_colors[i % 10]
        n_plot = min(max_traces_per_cluster, len(cluster_data))
        for j in range(n_plot):
            ax.plot(time_ms, cluster_data[j], color=color, alpha=0.7, linewidth=0.9)
        ax.set_title(f"Cluster {cluster_id} (n={len(cluster_data)} traces)", fontsize=14, fontweight="bold", color=color)
        ax.set_xlabel("Time (ms)", fontsize=12)
        ax.set_ylabel("Z-score", fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_name = "cluster_traces_only.png"
    plt.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches="tight")
    plt.close()


def save_cluster_composition(labels: np.ndarray, genotype_labels: List[str], output_dir: str):
    """Save cluster composition. Labels can be any strings (wt, gnat2, unknown, etc.); not genotype-dependent."""

    unique_labels = sorted(set(labels.tolist()))
    all_genotypes = sorted(set(genotype_labels))

    composition = []
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_genotypes = np.array(genotype_labels)[mask]
        total = len(cluster_genotypes)

        row = {'cluster_id': cluster_id, 'total_samples': total}
        for g in all_genotypes:
            c = int(np.sum(cluster_genotypes == g))
            row[f'{g}_samples'] = c
            row[f'{g}_percentage'] = (c / total * 100.0) if total > 0 else 0.0
        composition.append(row)

    df = pd.DataFrame(composition)
    df.to_csv(os.path.join(output_dir, 'cluster_composition.csv'), index=False)

    for _, row in df.iterrows():
        parts = [f"{g}:{row[f'{g}_samples']}" for g in all_genotypes]
        if row['cluster_id'] == -1:
            pass
        else:
            pcts = [f"{g}:{row[f'{g}_percentage']:.1f}%" for g in all_genotypes]


def remove_noise_and_recluster(X: np.ndarray, labels: np.ndarray, genotype_labels: List[str],
                              sample_metadata: List[Tuple],
                              umap_components: int, umap_neighbors: int,
                              min_samples: int, min_cluster_size: int, xi: float,
                              output_dir: str,
                              trace_stat: str = "mean"):
    """Remove noise samples and re-cluster"""

    non_noise_mask = labels != -1
    X_clean = X[non_noise_mask]
    genotype_clean = [genotype_labels[i] for i in range(len(genotype_labels)) if non_noise_mask[i]]
    metadata_clean = [sample_metadata[i] for i in range(len(sample_metadata)) if non_noise_mask[i]]


    if len(X_clean) < 50:
        return

    labels_clean, embedding_2d_clean = run_final_clustering(
        X_clean, umap_components, umap_neighbors, min_samples, min_cluster_size, xi
    )

    clean_output_dir = os.path.join(output_dir, "noise_removed")
    os.makedirs(clean_output_dir, exist_ok=True)

    plot_umap_2d(embedding_2d_clean, labels_clean, genotype_clean, clean_output_dir)
    plot_cluster_average_traces(X_clean, labels_clean, genotype_clean, clean_output_dir, trace_stat=trace_stat)
    save_cluster_composition(labels_clean, genotype_clean, clean_output_dir)


def compute_metrics_over_seeds(X: np.ndarray, umap_components: int, umap_neighbors: int,
                               min_samples: int, min_cluster_size: int, xi: float,
                               seeds: List[int]) -> Tuple[pd.DataFrame, dict]:
    """Run multiple seeds to compute stability metrics (ARI) and silhouette."""
    run_rows = []
    labels_list = []

    for seed in seeds:
        labels, embedding_2d = run_final_clustering(
            X, umap_components, umap_neighbors, min_samples, min_cluster_size, xi, random_state=seed
        )
        labels_list.append(labels)

        mask = labels != -1
        sil = np.nan
        if np.sum(mask) > 1:
            unique_non_noise = np.unique(labels[mask])
            if len(unique_non_noise) > 1:
                try:
                    sil = float(silhouette_score(embedding_2d[mask], labels[mask], metric='euclidean'))
                except Exception:
                    sil = np.nan
        n_clusters = len(np.unique(labels[labels != -1]))
        noise_ratio = float(np.mean(labels == -1))
        run_rows.append({
            'seed': seed,
            'num_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'silhouette': sil
        })

    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            li = labels_list[i]
            lj = labels_list[j]
            mask = (li != -1) & (lj != -1)
            if np.sum(mask) > 1 and len(np.unique(li[mask])) > 1 and len(np.unique(lj[mask])) > 1:
                ari = adjusted_rand_score(li[mask], lj[mask])
                aris.append(ari)

    median_ari = float(np.median(aris)) if aris else 0.0
    ari_std = float(np.std(aris)) if aris else 0.0

    df_runs = pd.DataFrame(run_rows)
    summary = {
        'median_ari': median_ari,
        'ari_std': ari_std,
        'median_silhouette': df_runs['silhouette'].median(),
        'silhouette_std': df_runs['silhouette'].std(),
        'median_clusters': df_runs['num_clusters'].median(),
        'median_noise_ratio': df_runs['noise_ratio'].median()
    }

    return df_runs, summary


def main():
    parser = argparse.ArgumentParser(description="Visualize clusters using best parameters from Stage 1 & 2")

    parser.add_argument("--data_dir", default="Data", help="Data directory")
    parser.add_argument("--output_dir", default="Results_Visualization", help="Output directory")
    parser.add_argument("--keep_timebins", type=int, default=100, help="Number of timebins to keep (e.g., 50 or 100)")

    parser.add_argument("--umap_components", type=int, required=True, help="Best UMAP components from Stage 1")
    parser.add_argument("--umap_neighbors", type=int, required=True, help="Best UMAP neighbors from Stage 1")
    parser.add_argument("--min_samples", type=int, required=True, help="Best OPTICS min_samples from Stage 2")
    parser.add_argument("--min_cluster_size", type=int, required=True, help="Best OPTICS min_cluster_size from Stage 2")
    parser.add_argument("--xi", type=float, required=True, help="Best OPTICS xi from Stage 2")

    parser.add_argument("--remove_noise", action="store_true", help="Also run clustering after removing noise")
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2], help="Seeds for stability metrics (ARI)")
    parser.add_argument("--trace_stat", choices=["mean", "median"], default="mean", help="Cluster trace aggregation")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)


    processor = DataProcessor(args.data_dir)
    X, genotype_labels, sample_metadata = processor.process_all_data(keep_timebins=args.keep_timebins)

    labels, embedding_2d = run_final_clustering(
        X, args.umap_components, args.umap_neighbors,
        args.min_samples, args.min_cluster_size, args.xi,
        random_state=42
    )

    plot_umap_2d(embedding_2d, labels, genotype_labels, args.output_dir)
    plot_cluster_average_traces(X, labels, genotype_labels, args.output_dir, trace_stat=args.trace_stat)
    save_cluster_composition(labels, genotype_labels, args.output_dir)

    df_runs, summary = compute_metrics_over_seeds(
        X, args.umap_components, args.umap_neighbors,
        args.min_samples, args.min_cluster_size, args.xi,
        seeds=args.seeds
    )
    df_runs.to_csv(os.path.join(args.output_dir, 'final_run_metrics.csv'), index=False)
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, 'final_metrics_summary.csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recluster(
            X, labels, genotype_labels, sample_metadata,
            args.umap_components, args.umap_neighbors,
            args.min_samples, args.min_cluster_size, args.xi,
            args.output_dir,
            trace_stat=args.trace_stat,
        )


if __name__ == "__main__":
    main()
