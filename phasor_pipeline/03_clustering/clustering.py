#!/usr/bin/env python3
"""
clustering.py - UMAP + OPTICS clustering and evaluation module
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA

_EMBED = os.environ.get("GLIA_EMBEDDING", "pca" if sys.platform == "darwin" else "umap").strip().lower()
if sys.platform == "darwin" and _EMBED == "umap":
    _EMBED = "pca"


class ClusteringEvaluator:
    """Handles UMAP + OPTICS clustering and evaluation"""

    def __init__(self):
        self.results_history = []

    def cluster_once(self, X: np.ndarray, umap_components: int, umap_neighbors: int,
                     min_samples: int, min_cluster_size: int, xi: float,
                     random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run UMAP + OPTICS clustering once with given parameters

        Args:
            X: Data matrix (samples, features)
            umap_components: UMAP dimensionality for clustering space
            umap_neighbors: UMAP n_neighbors parameter
            min_samples: OPTICS min_samples parameter
            min_cluster_size: OPTICS min_cluster_size parameter
            xi: OPTICS xi parameter
            random_seed: Random seed for reproducibility

        Returns:
            labels: Cluster labels (-1 for noise)
            embedding: UMAP embedding used for clustering
        """

        if _EMBED == "umap":
            import umap  # noqa: PLC0415

            reducer = umap.UMAP(
                n_components=umap_components,
                n_neighbors=umap_neighbors,
                min_dist=0.05,
                metric="correlation",
                random_state=random_seed,
            )
            embedding = reducer.fit_transform(X)
        else:
            reducer = PCA(n_components=int(umap_components))
            embedding = reducer.fit_transform(X)

        optics = OPTICS(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            xi=xi,
            metric='euclidean'
        )

        labels = optics.fit_predict(embedding)

        return labels, embedding

    def calculate_metrics(self, labels_list: List[np.ndarray],
                         embeddings_list: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate clustering quality metrics across multiple runs
        """

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

        silhouette_scores = []
        noise_ratios = []
        num_clusters_list = []

        for labels, embedding in zip(labels_list, embeddings_list):
            noise_ratio = float(np.mean(labels == -1))
            noise_ratios.append(noise_ratio)

            unique_clusters = np.unique(labels[labels != -1])
            num_clusters_list.append(len(unique_clusters))

            mask = labels != -1
            if np.sum(mask) > 1 and len(unique_clusters) > 1:
                sil_score = silhouette_score(embedding[mask], labels[mask], metric='euclidean')
                silhouette_scores.append(float(sil_score))
            else:
                silhouette_scores.append(-1.0)

        return {
            "median_ari": median_ari,
            "median_silhouette": float(np.median(silhouette_scores)) if silhouette_scores else -1.0,
            "noise_ratio": float(np.median(noise_ratios)),
            "median_num_clusters": float(np.median(num_clusters_list)) if num_clusters_list else 0.0,
            "ari_std": float(np.std(aris)) if aris else 0.0,
            "silhouette_std": float(np.std(silhouette_scores)) if silhouette_scores else 0.0
        }

    def evaluate_parameter_combination(self, X: np.ndarray,
                                     umap_components: int, umap_neighbors: int,
                                     min_samples: int, min_cluster_size: int, xi: float,
                                     seeds: List[int], verbose: bool = False) -> Dict:
        """
        Evaluating one parameter combination across multiple random seeds
        """

        if verbose:
            pass

        labels_list = []
        embeddings_list = []

        for seed in seeds:
            labels, embedding = self.cluster_once(
                X, umap_components, umap_neighbors,
                min_samples, min_cluster_size, xi, seed
            )
            labels_list.append(labels)
            embeddings_list.append(embedding)

        metrics = self.calculate_metrics(labels_list, embeddings_list)

        result = {
            "umap_components": umap_components,
            "umap_neighbors": umap_neighbors,
            "optics_min_samples": min_samples,
            "optics_min_cluster_size": min_cluster_size,
            "optics_xi": xi,
            **metrics,
            "num_seeds": len(seeds),
            "example_labels": labels_list[0]
        }

        if verbose:
            pass

        self.results_history.append(result)
        return result

    def calculate_optics_defaults(self, num_samples: int) -> Tuple[int, int, float]:
        """
        Calculate default OPTICS parameters according to guidelines

        Args:
            num_samples: Total number of samples (unit×channel pairs)

        Returns:
            (min_samples, min_cluster_size, xi)
        """
        min_samples = max(5, int(np.sqrt(num_samples)))
        min_cluster_size = max(5, int(0.05 * num_samples))
        xi = 0.05

        return min_samples, min_cluster_size, xi

    def compare_results(self, result1: Dict, result2: Dict) -> int:
        """
        Compare two clustering results according to selection criteria

        Selection priority:
        1. Higher median ARI
        2. Higher median silhouette
        3. Lower noise ratio

        Returns:
            1 if result1 is better, -1 if result2 is better, 0 if tied
        """

        if result1["median_ari"] > result2["median_ari"]:
            return 1
        elif result1["median_ari"] < result2["median_ari"]:
            return -1

        if result1["median_silhouette"] > result2["median_silhouette"]:
            return 1
        elif result1["median_silhouette"] < result2["median_silhouette"]:
            return -1

        if result1["noise_ratio"] < result2["noise_ratio"]:
            return 1
        elif result1["noise_ratio"] > result2["noise_ratio"]:
            return -1

        return 0

    def print_detailed_metrics(self, result: Dict, title: str = "Clustering Results"):
        """Print detailed metrics for a clustering result"""


def print_clustering_explanation():
    """explanation of the clustering approach"""


if __name__ == "__main__":
    print_clustering_explanation()

    np.random.seed(42)
    X_dummy = np.random.randn(100, 50)

    evaluator = ClusteringEvaluator()

    min_samples, min_cluster_size, xi = evaluator.calculate_optics_defaults(X_dummy.shape[0])

    result = evaluator.evaluate_parameter_combination(
        X_dummy,
        umap_components=5, umap_neighbors=15,
        min_samples=min_samples, min_cluster_size=min_cluster_size, xi=xi,
        seeds=[0, 1, 2], verbose=True
    )

    evaluator.print_detailed_metrics(result, "Test Results")
