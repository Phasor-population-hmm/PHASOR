#!/usr/bin/env python3
"""
automated pipeline with parameter selection
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


from data_processor import DataProcessor
from clustering import ClusteringEvaluator
from cluster_Viz import run_final_clustering, plot_umap_2d, plot_cluster_average_traces, plot_cluster_traces_only, save_cluster_composition


def select_best(results_df: pd.DataFrame, criteria: Dict[str, float]) -> Tuple[Dict, bool]:
    """Select best parameter row based on criteria; fallback to best overall.
    """
    if results_df.empty:
        raise ValueError("No results to select from")

    def non_degenerate(df: pd.DataFrame) -> pd.DataFrame:
        cols = df.columns
        has_clusters = 'median_num_clusters' in cols
        mask = (df['noise_ratio'] > 0) & (df['median_silhouette'] > 0)
        if has_clusters:
            mask &= (df['median_num_clusters'] >= 2)
        return df[mask].copy()

    ari_t = criteria.get('ari_threshold', 0.85)
    sil_t = criteria.get('sil_threshold', 0.45)
    noise_t = criteria.get('noise_threshold', 0.45)

    filtered = non_degenerate(results_df)

    qualified = filtered[(filtered['median_ari'] > ari_t) &
                         (filtered['median_silhouette'] > sil_t) &
                         (filtered['noise_ratio'] < noise_t)].copy()

    def sort_key(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(by=['noise_ratio', 'median_silhouette', 'median_ari'],
                              ascending=[True, False, False])

    if not qualified.empty:
        ranked = sort_key(qualified)
        return ranked.iloc[0].to_dict(), True

    if not filtered.empty:
        ranked_all = sort_key(filtered)
        return ranked_all.iloc[0].to_dict(), False

    last_resort = results_df[results_df['noise_ratio'] > 0]
    if not last_resort.empty:
        ranked_any = last_resort.sort_values(by=['noise_ratio', 'median_silhouette', 'median_ari'],
                                            ascending=[True, False, False])
        return ranked_any.iloc[0].to_dict(), False

    ranked_any = results_df.sort_values(by=['noise_ratio', 'median_silhouette', 'median_ari'],
                                        ascending=[True, False, False])
    return ranked_any.iloc[0].to_dict(), False


def run_pipeline_for_dataset(data_dir: str,
                             umap_components: List[int], umap_neighbors: List[int],
                             optics_min_samples: List[int], optics_min_cluster_frac: List[float], optics_xi: List[float],
                             seeds: List[int],
                             criteria: Dict[str, float],
                             sigma_bins: float = 3.0,
                             keep_timebins: int = None,
                             trace_stat: str = "mean") -> Dict:
    data_dir = str(Path(data_dir))
    dataset_name = Path(data_dir).name
    out_root = Path(data_dir)

    processor = DataProcessor(data_dir)

    if keep_timebins is None:
        npz_files = [f for f in os.listdir(data_dir) if f.endswith("_4d.npz")]
        if npz_files and 'rest' in npz_files[0].lower():
            test_load = np.load(os.path.join(data_dir, npz_files[0]))
            test_array = test_load['array'] if 'array' in test_load.files else test_load[list(test_load.files)[0]]
            keep_timebins = test_array.shape[1]
        else:
            keep_timebins = 300
    else:
        pass

    X, genotype_labels, sample_metadata = processor.process_all_data(keep_timebins=keep_timebins, sigma_bins=sigma_bins)

    stage1_dir = out_root / 'Stage1_UMAP_Tuning'
    if stage1_dir.exists() and any(stage1_dir.iterdir()):
        pass
    stage1_dir.mkdir(exist_ok=True, parents=True)

    evaluator = ClusteringEvaluator()

    stage1_rows = []
    default_min_samples, default_min_cluster_size, default_xi = evaluator.calculate_optics_defaults(num_samples=X.shape[0])

    for comp in umap_components:
        for neigh in umap_neighbors:
            res = evaluator.evaluate_parameter_combination(
                X,
                comp, neigh,
                default_min_samples, default_min_cluster_size, default_xi,
                seeds=seeds,
                verbose=False
            )
            stage1_rows.append(res)

    s1_df = pd.DataFrame(stage1_rows)
    s1_df.to_csv(stage1_dir / 'stage1_umap_results.csv', index=False)

    best_umap, umap_ok = select_best(s1_df, criteria)

    stage2_dir = out_root / 'Stage2_OPTICS_Tuning'
    if stage2_dir.exists() and any(stage2_dir.iterdir()):
        pass
    stage2_dir.mkdir(exist_ok=True, parents=True)

    stage2_rows = []
    for ms in optics_min_samples:
        for frac in optics_min_cluster_frac:
            min_cluster_size = max(5, int(frac * X.shape[0]))
            for xi in optics_xi:
                res = evaluator.evaluate_parameter_combination(
                    X,
                    int(best_umap['umap_components']), int(best_umap['umap_neighbors']),
                    int(ms), int(min_cluster_size), float(xi),
                    seeds=seeds,
                    verbose=False
                )
                stage2_rows.append(res)

    s2_df = pd.DataFrame(stage2_rows)
    s2_df.to_csv(stage2_dir / 'stage2_optics_results.csv', index=False)

    best_optics, optics_ok = select_best(s2_df, criteria)

    stage3_dir = out_root / 'Stage3_Final_Clustering'
    if stage3_dir.exists() and any(stage3_dir.iterdir()):
        pass
    stage3_dir.mkdir(exist_ok=True, parents=True)

    final_metrics = evaluator.evaluate_parameter_combination(
        X,
        int(best_optics['umap_components']), int(best_optics['umap_neighbors']),
        int(best_optics['optics_min_samples']), int(best_optics['optics_min_cluster_size']), float(best_optics['optics_xi']),
        seeds=seeds,
        verbose=True
    )
    pd.DataFrame([final_metrics]).to_csv(stage3_dir / 'final_metrics.csv', index=False)

    stage4_dir = out_root / 'Stage4_Visualization'
    if stage4_dir.exists() and any(stage4_dir.iterdir()):
        pass
    stage4_dir.mkdir(exist_ok=True, parents=True)

    labels, embedding_2d = run_final_clustering(
        X,
        int(best_optics['umap_components']), int(best_optics['umap_neighbors']),
        int(best_optics['optics_min_samples']), int(best_optics['optics_min_cluster_size']), float(best_optics['optics_xi']),
        random_state=42
    )
    plot_umap_2d(embedding_2d, labels, genotype_labels, str(stage4_dir))
    plot_cluster_average_traces(X, labels, genotype_labels, str(stage4_dir), trace_stat="mean")
    plot_cluster_average_traces(X, labels, genotype_labels, str(stage4_dir), trace_stat="median")
    plot_cluster_traces_only(X, labels, str(stage4_dir))
    save_cluster_composition(labels, genotype_labels, str(stage4_dir))

    np.save(stage4_dir / 'cluster_labels.npy', labels)
    np.save(stage4_dir / 'X_processed.npy', X)
    np.save(stage4_dir / 'embedding_2d.npy', embedding_2d)

    report_lines = []
    report_lines += [
        f"Dataset: {dataset_name}",
        f"Data: {data_dir}",
        f"Samples × Timebins: {X.shape}",
        "",
        f"Criteria: ARI>{criteria['ari_threshold']}  Sil>{criteria['sil_threshold']}  Noise<{criteria['noise_threshold']}",
        "",
        f"Stage 1 (UMAP): components={int(best_umap['umap_components'])}, neighbors={int(best_umap['umap_neighbors'])}",
        f"  Metrics: ARI={best_umap['median_ari']:.3f}, Sil={best_umap['median_silhouette']:.3f}, Noise={best_umap['noise_ratio']:.3f}, Clusters={best_umap['median_num_clusters']:.1f}",
        f"  Meets criteria: {'Yes' if umap_ok else 'No'}",
        "",
        f"Stage 2 (OPTICS): min_samples={int(best_optics['optics_min_samples'])}, min_cluster_size={int(best_optics['optics_min_cluster_size'])}, xi={float(best_optics['optics_xi'])}",
        f"  Metrics: ARI={best_optics['median_ari']:.3f}, Sil={best_optics['median_silhouette']:.3f}, Noise={best_optics['noise_ratio']:.3f}, Clusters={best_optics['median_num_clusters']:.1f}",
        f"  Meets criteria: {'Yes' if optics_ok else 'No'}",
        "",
        f"Final (median over seeds): ARI={final_metrics['median_ari']:.3f}, Silhouette={final_metrics['median_silhouette']:.3f}, Noise={final_metrics['noise_ratio']:.3f}, Clusters={final_metrics['median_num_clusters']:.1f}",
    ]

    with open(out_root / 'pipeline_summary_report.txt', 'w') as f:
        f.write("\n".join(report_lines))

    prev_dir = Path('pipeline_clustering/Filtered Data/Data_results/Results_F10_C90_Full_300bins')
    compare_path = prev_dir / 'final_metrics_summary.csv'
    if compare_path.exists():
        prev = pd.read_csv(compare_path)
        cmp_lines = [
            "",
            "Manual comparison:",
            f"  Manual median_clusters={prev['median_clusters'].iloc[0]}",
            f"  Manual median_noise_ratio={prev['median_noise_ratio'].iloc[0]:.3f}",
            f"  Manual median_silhouette={prev['median_silhouette'].iloc[0]:.3f}",
        ]
        with open(out_root / 'pipeline_summary_report.txt', 'a') as f:
            f.write("\n".join(cmp_lines))

    return {
        'best_umap': best_umap,
        'best_optics': best_optics,
        'final_metrics': final_metrics,
        'output_dir': str(out_root)
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run automated UMAP+OPTICS clustering pipeline')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset folder containing filtered *_4d.npz files')

    parser.add_argument('--umap_components', type=int, nargs='+', default=[3, 5, 8], help='List of UMAP n_components values')
    parser.add_argument('--umap_neighbors', type=int, nargs='+', default=[5, 10, 15], help='List of UMAP n_neighbors values')

    parser.add_argument('--optics_min_samples', type=int, nargs='+', default=[30, 50, 70], help='List for OPTICS min_samples')
    parser.add_argument('--optics_min_cluster_frac', type=float, nargs='+', default=[0.003, 0.005, 0.01], help='List of fractions for min_cluster_size (size = max(5, frac * n_samples))')
    parser.add_argument('--optics_xi', type=float, nargs='+', default=[0.03, 0.05, 0.08], help='List for OPTICS xi')

    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4], help='Random seeds for stability/ARI')

    parser.add_argument('--sigma_bins', type=float, default=3.0, help='Gaussian smoothing sigma (set to 0 for no smoothing)')
    parser.add_argument('--keep_timebins', type=int, default=None, help='Number of timebins to keep (None = auto-detect from data, default=300 if not rest data)')

    parser.add_argument('--ari_threshold', type=float, default=0.85, help='Minimum ARI to qualify')
    parser.add_argument('--sil_threshold', type=float, default=0.45, help='Minimum silhouette to qualify')
    parser.add_argument('--noise_threshold', type=float, default=0.45, help='Maximum noise ratio to qualify')
    parser.add_argument('--trace_stat', choices=['mean', 'median'], default='mean', help='Cluster trace aggregation for Stage4 traces plot')

    args = parser.parse_args()

    results = run_pipeline_for_dataset(
        data_dir=args.data_dir,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
        optics_min_samples=args.optics_min_samples,
        optics_min_cluster_frac=args.optics_min_cluster_frac,
        optics_xi=args.optics_xi,
        seeds=args.seeds,
        criteria={'ari_threshold': args.ari_threshold,
                  'sil_threshold': args.sil_threshold,
                  'noise_threshold': args.noise_threshold},
        sigma_bins=args.sigma_bins,
        keep_timebins=args.keep_timebins,
        trace_stat=args.trace_stat,
    )

