import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from itertools import combinations
from scipy import sparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import ensure_dense_array, safe_len, validate_clustering_input
import warnings
warnings.filterwarnings('ignore')

class GameTheoryClusterer:
    def __init__(self, data, gamma=1.0, similarity_metric='euclidean'):
        """
        Game Theory based clustering using coalition formation and Shapley values.
        
        Args:
            data: Input data matrix
            gamma: Temperature parameter for similarity function
            similarity_metric: 'euclidean' or 'cosine'
        """
        # Ensure data is dense
        if sparse.issparse(data):
            data = data.toarray()
        self.data = np.asarray(data, dtype=np.float64)
        self.gamma = gamma
        self.n_samples = self.data.shape[0]
        self.similarity_metric = similarity_metric
        
        # Compute similarity matrix
        if similarity_metric == 'euclidean':
            self.dist_matrix = euclidean_distances(self.data)
            # Convert distances to similarities
            self.similarity_matrix = np.exp(-self.dist_matrix ** 2 / (2 * self.gamma ** 2))
        else:  # cosine
            self.similarity_matrix = cosine_similarity(self.data)
            self.similarity_matrix = np.clip(self.similarity_matrix, 0, 1)  # Ensure non-negative
        
        # Set diagonal to 1 (self-similarity)
        np.fill_diagonal(self.similarity_matrix, 1.0)
    
    def coalition_utility(self, coalition):
        """
        Compute utility of a coalition based on internal cohesion.
        Higher utility = more cohesive coalition.
        """
        if len(coalition) <= 1:
            return 0.0
        
        coalition = list(coalition)
        
        # Internal cohesion: average pairwise similarity within coalition
        internal_similarities = []
        for i in range(len(coalition)):
            for j in range(i + 1, len(coalition)):
                internal_similarities.append(self.similarity_matrix[coalition[i], coalition[j]])
        
        if not internal_similarities:
            return 0.0
            
        # Utility is the average internal similarity multiplied by coalition size advantage
        avg_internal_sim = np.mean(internal_similarities)
        size_bonus = np.log(len(coalition) + 1)  # Logarithmic size bonus
        
        return avg_internal_sim * size_bonus
    
    def marginal_contribution(self, player, coalition):
        """
        Compute marginal contribution of a player to a coalition.
        """
        coalition_without_player = [p for p in coalition if p != player]
        
        utility_with = self.coalition_utility(coalition)
        utility_without = self.coalition_utility(coalition_without_player)
        
        return utility_with - utility_without
    
    def compute_shapley_values(self, max_coalition_size=None):
        """
        Compute Shapley values for coalition formation.
        Due to computational complexity, we use sampling for large datasets.
        """
        n = self.n_samples
        shapley_matrix = np.zeros((n, n))
        
        # For large datasets, limit coalition size to make computation feasible
        if max_coalition_size is None:
            max_coalition_size = min(8, n)  # Limit to reasonable size
        
        # For each pair of players, compute their Shapley interaction value
        for i in range(n):
            for j in range(i + 1, n):
                shapley_value = 0.0
                count = 0
                
                # Sample coalitions containing both i and j
                for size in range(2, min(max_coalition_size + 1, n + 1)):
                    # Sample random coalitions of this size containing i and j
                    other_players = [k for k in range(n) if k != i and k != j]
                    
                    if len(other_players) + 2 < size:
                        continue
                        
                    num_samples = min(50, len(list(combinations(other_players, size - 2))))
                    
                    for _ in range(num_samples):
                        if size == 2:
                            coalition = [i, j]
                        else:
                            # Randomly sample other players
                            sampled_others = np.random.choice(
                                other_players, 
                                size=min(size - 2, len(other_players)), 
                                replace=False
                            )
                            coalition = [i, j] + list(sampled_others)
                        
                        # Compute marginal contributions
                        mc_i = self.marginal_contribution(i, coalition)
                        mc_j = self.marginal_contribution(j, coalition)
                        
                        # Average marginal contribution represents interaction strength
                        shapley_value += (mc_i + mc_j) / 2
                        count += 1
                
                if count > 0:
                    shapley_value /= count
                    shapley_matrix[i, j] = shapley_value
                    shapley_matrix[j, i] = shapley_value
        
        return shapley_matrix
    
    def form_coalitions(self, shapley_matrix, threshold=0.3):
        """
        Form coalitions (clusters) based on Shapley values.
        """
        n = self.n_samples
        labels = -np.ones(n, dtype=int)
        cluster_id = 0
        unassigned = set(range(n))
        
        # Sort potential pairs by their Shapley values
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if shapley_matrix[i, j] > threshold:
                    pairs.append((shapley_matrix[i, j], i, j))
        
        pairs.sort(reverse=True)  # Start with strongest connections
        
        # Greedy coalition formation
        for shapley_val, i, j in pairs:
            if i in unassigned and j in unassigned:
                # Start new coalition
                labels[i] = cluster_id
                labels[j] = cluster_id
                unassigned.remove(i)
                unassigned.remove(j)
                
                # Try to add more members to this coalition
                current_coalition = {i, j}
                improved = True
                
                while improved and unassigned:
                    improved = False
                    best_candidate = None
                    best_score = threshold
                    
                    for candidate in list(unassigned):
                        # Compute average Shapley value with current coalition members
                        avg_shapley = np.mean([shapley_matrix[candidate, member] 
                                             for member in current_coalition])
                        
                        if avg_shapley > best_score:
                            best_score = avg_shapley
                            best_candidate = candidate
                            improved = True
                    
                    if best_candidate is not None:
                        labels[best_candidate] = cluster_id
                        current_coalition.add(best_candidate)
                        unassigned.remove(best_candidate)
                
                cluster_id += 1
            
            elif i in unassigned and labels[j] != -1:
                # Add i to j's coalition if beneficial
                coalition_members = np.where(labels == labels[j])[0]
                avg_shapley = np.mean([shapley_matrix[i, member] for member in coalition_members])
                
                if avg_shapley > threshold:
                    labels[i] = labels[j]
                    unassigned.remove(i)
            
            elif j in unassigned and labels[i] != -1:
                # Add j to i's coalition if beneficial
                coalition_members = np.where(labels == labels[i])[0]
                avg_shapley = np.mean([shapley_matrix[j, member] for member in coalition_members])
                
                if avg_shapley > threshold:
                    labels[j] = labels[i]
                    unassigned.remove(j)
        
        # Assign remaining unassigned points to singleton clusters
        for point in unassigned:
            labels[point] = cluster_id
            cluster_id += 1
        
        return labels
    
    def fit(self, threshold=0.3, max_coalition_size=None):
        """
        Perform game theory clustering.
        
        Args:
            threshold: Minimum Shapley value for coalition formation
            max_coalition_size: Maximum size of coalitions to consider (for computational efficiency)
        
        Returns:
            Cluster labels
        """
        # Compute Shapley values
        shapley_matrix = self.compute_shapley_values(max_coalition_size)
        
        # Form coalitions based on Shapley values
        labels = self.form_coalitions(shapley_matrix, threshold)
        
        # Store results for analysis
        self.shapley_matrix_ = shapley_matrix
        self.labels_ = labels
        
        return labels
    
    def get_cluster_statistics(self):
        """
        Get statistics about the formed clusters.
        """
        if not hasattr(self, 'labels_'):
            raise ValueError("Must call fit() first")
        
        unique_labels = np.unique(self.labels_)
        stats = {}
        
        for label in unique_labels:
            cluster_members = np.where(self.labels_ == label)[0]
            cluster_size = len(cluster_members)
            
            if cluster_size > 1:
                # Internal cohesion
                internal_similarities = []
                for i in range(len(cluster_members)):
                    for j in range(i + 1, len(cluster_members)):
                        internal_similarities.append(
                            self.similarity_matrix[cluster_members[i], cluster_members[j]]
                        )
                avg_internal_sim = np.mean(internal_similarities) if internal_similarities else 0
            else:
                avg_internal_sim = 1.0
            
            stats[f'Cluster_{label}'] = {
                'size': cluster_size,
                'members': cluster_members.tolist(),
                'avg_internal_similarity': avg_internal_sim
            }
        
        return stats
