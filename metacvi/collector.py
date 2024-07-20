import json
import os.path
import random

import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler

from metacvi.reducers import Reducer

matplotlib.rcParams['figure.dpi'] = 500
matplotlib.rcParams['savefig.pad_inches'] = 0


class DatasetForMetaCVI:
    def __init__(self, name: str, reducer: Reducer, original: np.ndarray):
        self.name, self.reducer = name, reducer
        self.data = reducer.fit_transform(original)
        self.data = MinMaxScaler().fit_transform(self.data)
        self.dir_name = f'data/{name}/{reducer.name}'

        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)

        pd.DataFrame(data=self.data).to_csv(f'{self.dir_name}/gen.csv', header=False, index=False)


class DatasetInfoCollector:
    PARTITIONS_TO_ESTIMATE = 15

    COLORS = {
        -1: "gray",
        0: "blue",
        1: "orange",
        2: "green",
        3: "red",
        4: "purple",
        5: "brown",
        6: "pink",
        7: "olive",
        8: "cyan"
    }

    def __init__(self, dataset: DatasetForMetaCVI):
        self.dataset = dataset
        self.registered = list()

        # fake_labels = np.full(shape=len(self.dataset.data), fill_value=2, dtype=int)
        # self.scatter(fake_labels, 0)

    def save(self, partition, producer):
        if np.max(partition) > 8 or np.max(partition) < 1 or self._is_too_noisy(partition):
            print("INVALID LABELS")
        else:
            self.registered.append((partition, producer))

    def _is_too_noisy(self, partition):
        noise = [1 for label in partition if label == -1]
        return len(noise) / len(partition) > 0.1

    def persist(self):
        random.shuffle(self.registered)
        chosen_indices = self._choose_most_different()
        producers, partitions = list(), list()
        for idx, chose_index in enumerate(chosen_indices):
            partition, producer = self.registered[chose_index]
            producers.append(
                {
                    "algo": producer.name,
                    "params": producer.algo.get_params()
                }
            )
            partitions.append(partition)
            self._scatter(partition, idx)
        with open(f'{self.dataset.dir_name}/producers.json', 'w') as fp:
            json.dump(producers, fp)
        pd.DataFrame(partitions).to_csv(f'{self.dataset.dir_name}/partitions.csv', header=False, index=False)

    def _choose_most_different(self):
        n = len(self.registered)
        print(f"OBTAINED {n} PARTITIONS for {self.dataset.dir_name}")
        similarity_matrix = np.zeros((n, n))
        for x_idx in range(n):
            for y_idx in range(x_idx):
                x, y = self.registered[x_idx][0], self.registered[y_idx][0]
                score = metrics.fowlkes_mallows_score(x, y)
                similarity_matrix[x_idx, y_idx] = score
                similarity_matrix[y_idx, x_idx] = score
        evicted = set()
        while len(evicted) < n - self.PARTITIONS_TO_ESTIMATE:
            most_similar_idx = np.argmax(similarity_matrix)
            cur_evicted = most_similar_idx % n
            evicted.add(cur_evicted)
            similarity_matrix[cur_evicted, :] = 0
            similarity_matrix[:, cur_evicted] = 0

        return set(np.arange(n).tolist()) - evicted

    def _scatter(self, labels: np.ndarray, p_idx: int):
        x, y = self.dataset.data[:, 0], self.dataset.data[:, 1]
        colors = [self.COLORS[label] for label in labels]
        ax = plt.axes((0, 0, 1, 1), frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.autoscale(tight=True)
        plt.scatter(x, y, marker='.', c=colors, s=1)
        plt.savefig(f'{self.dataset.dir_name}/img-{p_idx}.png')
        plt.clf()
