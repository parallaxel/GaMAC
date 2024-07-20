from typing import List

import numpy as np
# import umap
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA


class Reducer:
    def __init__(self, algo, name):
        self.algo, self.name = algo, name

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.algo.fit_transform(x)


class ReducerProvider:
    def get_all(self) -> List[Reducer]:
        return [
            *self._pca_reducers(),
            *self._tsne_reducers(),
            *self._mds_reducers(),
            *self._umap_reducers(),
        ]
    
    def _pca_reducers(self):
        return [Reducer(PCA(n_components=2, random_state=42), 'pca')]

    def _tsne_reducers(self):
        return [
            # Reducer(TSNE(n_components=2, perplexity=10, random_state=42), 'tsne-10'),
            Reducer(TSNE(n_components=2, perplexity=25, random_state=42), 'tsne'),
            # Reducer(TSNE(n_components=2, perplexity=40, random_state=42), 'tsne-40'),
            # Reducer(TSNE(n_components=2, perplexity=55, random_state=42), 'tsne-55'),
        ]

    def _mds_reducers(self):
        return [
            Reducer(MDS(n_components=2, random_state=42), 'mds'),
        ]

    def _umap_reducers(self):
        return [
            # Reducer(umap.UMAP(n_components=2, n_neighbors=50), 'umap'),
        ]
