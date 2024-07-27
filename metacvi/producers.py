from typing import List

import numpy as np
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AffinityPropagation,
    MeanShift,
    AgglomerativeClustering,
    BisectingKMeans,
    Birch,
    SpectralClustering,
)

from metacvi.collector import DatasetForMetaCVI


class Producer:
    def __init__(self, algo, name):
        self.algo, self.name = algo, name

    def fit_predict(self, dataset: DatasetForMetaCVI) -> np.ndarray:
        return self.algo.fit_predict(dataset.data)


class ProducerProvider:
    SEEDS = [42]

    def get_all(self) -> List[Producer]:
        return [
            # *self._affprop(),
            *self._agglomerative(),
            *self._dbscan(),
            *self._kmeans(),
            *self._bisecting(),
            *self._birch(),
            *self._meanshift(),
            *self._spectral(),
        ]

    def _kmeans(self):
        producers = list()
        for n_clusters in range(2, 8):
            for seed in self.SEEDS:
                producers.append(
                    Producer(
                        KMeans(n_clusters=n_clusters, random_state=seed),
                        f'KMeans'
                    )
                )
        return producers

    def _bisecting(self):
        producers = list()
        for n_clusters in range(2, 8):
            for seed in self.SEEDS:
                producers.append(
                    Producer(
                        BisectingKMeans(n_clusters=n_clusters, random_state=seed),
                        f'BisectingKMeans'
                    )
                )
        return producers

    def _birch(self):
        producers = list()
        for n_clusters in range(2, 8):
            for branching_factor in [20, 30, 50, 80, 120, 150]:
                for threshold in [0.1, 0.2, 0.35, 0.5, 0.7]:
                    producers.append(
                        Producer(
                            Birch(n_clusters=n_clusters, branching_factor=branching_factor, threshold=threshold),
                            f'Birch'
                        )
                    )
        return producers

    def _meanshift(self):
        producers = list()
        for bandwidth in [0.05, 0.11, 0.18, 0.26, 0.35, 0.45, 0.56, 0.68]:
            for cluster_all in [True, False]:
                producers.append(
                    Producer(
                        MeanShift(bandwidth=bandwidth, cluster_all=cluster_all, n_jobs=-1),
                        'MeanShift'
                    )
                )
        return producers

    def _dbscan(self):
        producers = list()
        for min_samples in [5, 12, 21, 33, 55, 68]:
            for eps in [0.05, 0.12, 0.21, 0.33, 0.55, 0.68]:
                producers.append(
                    Producer(
                        DBSCAN(eps=eps, min_samples=min_samples),
                        f'DBSCAN'
                    )
                )

        return producers

    def _affprop(self):
        producers = list()
        for damping in [0.5, 0.65, 0.80, 0.95]:
            for preference in [-0.05, -0.20, -0.35, -0.50]:
                for seed in self.SEEDS:
                    producers.append(
                        Producer(
                            AffinityPropagation(damping=damping, preference=preference, random_state=seed),
                            f'AffinityPropagation'
                        )
                    )
        return producers

    def _agglomerative(self):
        producers = list()
        for n_clusters in range(2, 8):
            for linkage in ['ward', 'average']:
                producers.append(
                    Producer(
                        AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage),
                        f'AgglomerativeClustering'
                    )
                )
        return producers

    def _spectral(self):
        producers = list()
        for n_clusters in range(2, 8):
            producers.append(
                Producer(
                    SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels='cluster_qr'),
                    f'SpectralClustering'
                )
            )
        return producers
