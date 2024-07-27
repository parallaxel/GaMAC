import numpy as np
from scipy.stats import tmean, tstd, skew, kurtosis
from sklearn.metrics import pairwise_distances

from metacvi.utils import traverse_data, read_gen_data


class FeatureExtractor:
    BUCKETS = 100

    @staticmethod
    def extract(distances: np.ndarray) -> np.ndarray:
        norm_distances = distances / np.max(distances)
        n, data_features = len(distances), list()
        for obj_dists in norm_distances:
            sorted_dists = sorted(obj_dists)
            buckets = np.array_split(sorted_dists, FeatureExtractor.BUCKETS)
            obj_features = [np.mean(bucket) for bucket in buckets]
            data_features.append(obj_features)
        data_features = np.array(data_features)
        return np.array([
            *tmean(data_features, axis=0),
            *tstd(data_features, axis=0),
            *skew(data_features, axis=0),
            *kurtosis(data_features, axis=0)
        ])


def features(data_path):
    data = read_gen_data(data_path)
    d_matrix = pairwise_distances(data)
    meta_features = FeatureExtractor.extract(d_matrix)
    with open(f'data/{data_path}/features.txt', 'w') as fp:
        fp.write(str(meta_features.tolist()))


if __name__ == "__main__":
    traverse_data(features)
