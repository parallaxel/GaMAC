import numpy as np
import pandas as pd
from scipy.stats import tmean, tstd, skew, kurtosis
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler


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


if __name__ == "__main__":
    dataframe = pd.read_csv("data/wine-quality-red/orig.csv")
    data = dataframe.drop(columns=['class']).values
    normalised = MinMaxScaler().fit_transform(data)
    d_matrix = pairwise_distances(normalised)
    meta_features = FeatureExtractor.extract(d_matrix)
    print(meta_features)
