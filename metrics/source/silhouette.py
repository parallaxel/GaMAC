from numba import cuda
import numpy as np
from sklearn.preprocessing import LabelEncoder  # TODO: заменить кодирование меток

from ..utils.funcs import euclidean_distances_mean, euclidean_distances_numba, euclidean_distances_sum
from ..utils.checks import check_number_of_labels


@cuda.jit  # TODO: убрать LabelEncoder, ядро не компилируется
def silhouette_samples_memory_saving(X, labels):
    # X, labels = sklearn.utils.check_X_y(X, labels, accept_sparse=['csc', 'csr'])
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    check_number_of_labels(len(le.classes_), X.shape[0])

    unique_labels = le.classes_
    n_samples_per_label = np.bincount(labels, minlength=len(unique_labels))

    intra_clust_dists = np.zeros(X.shape[0], dtype=X.dtype)

    inter_clust_dists = np.inf + intra_clust_dists

    for curr_label in range(len(unique_labels)):

        mask = labels == curr_label

        n_samples_curr_lab = n_samples_per_label[curr_label] - 1
        if n_samples_curr_lab != 0:
            intra_clust_dists[mask] = euclidean_distances_sum(
                X[mask, :]) / n_samples_curr_lab

        for other_label in range(len(unique_labels)):
            if other_label != curr_label:
                other_mask = labels == other_label
                other_distances = euclidean_distances_mean(
                    X[mask, :], X[other_mask, :])
                inter_clust_dists[mask] = np.minimum(
                    inter_clust_dists[mask], other_distances)

    sil_samples = inter_clust_dists - intra_clust_dists
    sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    sil_samples[n_samples_per_label.take(labels) == 1] = 0
    return sil_samples
