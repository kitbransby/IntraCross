import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, 
    fowlkes_mallows_score
)
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt

class ClusteringFramework:
    def __init__(self, data, labels, feature_weights=None):
        """
        Initialize the clustering framework.

        Args:
            data (numpy.ndarray): Shape (N, 4), features already scaled in [0,1].
            labels (numpy.ndarray): Shape (N,), ground truth cluster labels.
        """
        self.original_data = data[:,[0,2,3,4]]
        self.angles = data[:,1]
        self.labels = labels

        # Apply custom feature weighting
        if feature_weights is None:
            feature_weights = [1, 1, 1, 1]  # Equal importance by default
        self.feature_weights = np.array(feature_weights)

        self.data = self.original_data * feature_weights
            

    def fit_dbscan(self, eps=0.2, min_samples=5):
        """
        Fit DBSCAN clustering.

        Args:
            eps (float): Maximum distance between points to be considered neighbors.
            min_samples (int): Minimum number of points to form a cluster.
        
        Returns:
            numpy.ndarray: Predicted cluster labels.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(self.data)
        self.cluster_labels = cluster_labels
        return cluster_labels

    def hungarian_matching_accuracy(self):
        """
        Compute clustering accuracy using Hungarian Matching.

        Returns:
            float: Accuracy score (0-1).
        """
        if not hasattr(self, "cluster_labels"):
            raise ValueError("Run clustering before evaluation.")

        pred_labels = self.cluster_labels
        true_labels = self.labels

        unique_pred = np.unique(pred_labels)
        unique_true = np.unique(true_labels)

        num_pred_clusters = len(unique_pred)
        num_true_clusters = len(unique_true)

        # Count occurrences in contingency table
        contingency_matrix = np.zeros((num_true_clusters, num_pred_clusters), dtype=np.int32)

        for i in range(len(pred_labels)):
            if pred_labels[i] == -1:  # Ignore noise points in DBSCAN
                continue
            true_idx = np.where(unique_true == true_labels[i])[0][0]
            pred_idx = np.where(unique_pred == pred_labels[i])[0][0]
            contingency_matrix[true_idx, pred_idx] += 1

        # Solve Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(contingency_matrix, maximize=True)
        total_correct = sum(contingency_matrix[row, col] for row, col in zip(row_ind, col_ind))

        return total_correct / len(true_labels)

    def evaluate_clustering(self):
        """
        Evaluate clustering results with the best 4 metrics and compute a combined score.

        Returns:
            dict: Dictionary containing ARI, NMI, Hungarian Accuracy, FMI, and Combined Score.
        """
        if not hasattr(self, "cluster_labels"):
            raise ValueError("Run clustering before evaluation.")

        unique_clusters = set(self.cluster_labels)

        if len(unique_clusters) <= 1:
            print("Warning: Only one cluster found. Metrics may not be valid.")

        ari = adjusted_rand_score(self.labels, self.cluster_labels)
        nmi = normalized_mutual_info_score(self.labels, self.cluster_labels)
        hma = self.hungarian_matching_accuracy()
        fmi = fowlkes_mallows_score(self.labels, self.cluster_labels)

        combined_score = np.mean([ari, nmi, hma, fmi])

        results = {
            "ARI": ari,
            "NMI": nmi,
            "Hungarian Matching Accuracy": hma,
            "Fowlkes-Mallows Score": fmi,
            "Combined Score": combined_score
        }

        return results

    def visualize_clusters(self):
        """
        Visualize clustering results.

        Args:
            feature_x (int): Feature index for X-axis.
            feature_y (int): Feature index for Y-axis.
        """
        if not hasattr(self, "cluster_labels"):
            raise ValueError("Run clustering before visualization.")

        plt.figure(figsize=(15,3))

        unique_clusters = np.unique(self.cluster_labels)

        # Scatter plot of all points
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                color = 'gray'  # Noise points in DBSCAN
                label = 'Noise'
            else:
                color = None  # Default color map
                label = f'Cluster {cluster_id}'
    
            # Select points for this cluster
            cluster_points = np.stack([self.data[:, 0], self.angles], axis=-1)
            cluster_points = cluster_points[self.cluster_labels == cluster_id]
    
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=label, s=40, alpha=0.5)

        plt.colorbar(label="Cluster ID")
        plt.title("DBSCAN Clustering Results")
        plt.savefig('clustering.png', dpi=400, transparent=True)
        #plt.show()

