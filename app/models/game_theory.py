
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class GameTheoryClusterer:
    def __init__(self, data, gamma=1.0):
        self.data = data
        self.gamma = gamma
        self.dist_matrix = euclidean_distances(data)

    def utility(self, coalition):
        if not coalition:
            return 0
        sims = np.exp(-self.dist_matrix[np.ix_(coalition, coalition)] ** 2 / self.gamma)
        return sims.sum() / 2

    def shapley_values(self):
        n = len(self.data)
        shapley = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                u_i = self.utility([i])
                u_ij = self.utility([i, j])
                shapley[i][j] = u_ij - u_i
        return shapley

    def fit(self, threshold=0.4):
        n = len(self.data)
        labels = -np.ones(n, dtype=int)
        cluster_id = 0
        unassigned = set(range(n))
        shapley = self.shapley_values()
        while unassigned:
            i = unassigned.pop()
            labels[i] = cluster_id
            for j in list(unassigned):
                if shapley[i][j] > threshold:
                    labels[j] = cluster_id
                    unassigned.remove(j)
            cluster_id += 1
        return labels
