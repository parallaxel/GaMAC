import numpy as np
import matplotlib.pyplot as plt


class MeanShift:
    def __init__(self, radius=4):
        self.radius = radius
        self.centroids = {}

    def fit(self, data):
        centroids = {i: data[i] for i in range(len(data))}
        optimized = False

        while not optimized:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)
                new_centroid = np.mean(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))
            uniques = sorted(list(set(new_centroids)))
            optimized = len(uniques) == len(centroids)
            centroids = {i: uniques[i] for i in range(len(uniques))}

        self.centroids = centroids

    def predict(self, data):
        predictions = []
        for featureset in data:
            distances = [np.linalg.norm(featureset - centroid) for centroid in self.centroids.values()]
            closest_centroid = np.argmin(distances)
            predictions.append(closest_centroid)
        return predictions


# # Example
# X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11], [8, 2], [10, 2], [9, 3]])
# mean_shift = MeanShift()
# mean_shift.fit(X)
# plt.scatter(X[:, 0], X[:, 1], s=150)
# for c in mean_shift.centroids.values():
#     plt.scatter(c[0], c[1], color='k', marker='*', s=150)
# plt.show()
