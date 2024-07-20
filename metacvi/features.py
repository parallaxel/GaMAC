import os

import numpy as np
import pandas as pd
import scipy.spatial
from scipy.stats import tmean, tstd, skew, kurtosis
from scipy.spatial import minkowski_distance
from sklearn.metrics import pairwise_distances


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


def launch(data_name, reducer):
    path = f'data/{data_name}/{reducer}'
    data = pd.read_csv(f"{path}/gen.csv").values
    d_matrix = pairwise_distances(data)
    meta_features = FeatureExtractor.extract(d_matrix)
    with open(f'{path}/features.txt', 'w') as fp:
        fp.write(str(meta_features.tolist()))
    return meta_features


if __name__ == "__main__":
    # for data_name in os.listdir('data'):
    for data_name in ['wine-quality-red']:
        data_variations = list()
        for reducer in os.listdir(f'data/{data_name}'):
            if reducer == "orig.csv":
                continue
            print(f"=== {data_name}/{reducer} ===")
            features = launch(data_name, reducer)
            data_variations.append(features)
        print(np.array([[minkowski_distance(x, y) for x in data_variations] for y in data_variations]))
