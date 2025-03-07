import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def get_sample_stat(
    test_value: float, 
    test_cluster: int, 
    train_distribution: list, 
    criterion: str) -> float:
    """
    Calculate test sample statistics with respect to train_distribution.

    Args:
        test_value (float): Value of the sample.
        test_cluster (int): Cluster of the sample.
        train_distribution (list): List of train values distributions across clusters [num_clusters, num_samples].
        criterion (str): A way to calculate statistics of test sample. Could be 'left-sided' or 'right-sided'.

    Returns:
        float: Test sample statistics.
    """

    cluster_size = len(train_distribution[test_cluster])
    
    if cluster_size == 0:
        return 0 if criterion == 'left-sided' else 1
        
    if criterion == 'left-sided':
        return sum(train_distribution[test_cluster] < test_value) / cluster_size
    elif criterion == 'right-sided':
        return 1 - sum(train_distribution[test_cluster] < test_value) / cluster_size


def get_scores(
    train_features: torch.Tensor, 
    test_features: torch.Tensor, 
    features_type: str = 'clusters_probas', 
    criterion: str = 'left-sided', 
    clusters_num: int = None) -> np.ndarray:
    """
    Calculate test samples statistics with respect to train_distribution.

    Args:
        train_features (torch.Tensor): Train values.
        test_features (torch.Tensor): Test values.
        features_type (str): If 'clusters_probas' then values are probs of samples to belong to clusters, if 'features' then values are abstract features.
        criterion (str): A way to calculate statistics of test sample. Could be 'left-sided' or 'right-sided'.
        clusters_num (int | None): If features_type = 'features' then number of clusters for the data clustering.
        
    Returns:
        np.ndarray: Test samples statistics.
    """
    
    train_features, test_features = train_features.numpy(), test_features.numpy()

    if features_type == 'clusters_probas':
        train_clusters = train_features.argmax(-1)
        train_distribution = [train_features[train_clusters == i, i] for i in range(train_features.shape[1])]
        test_clusters = test_features.argmax(-1)
        test_values = test_features.max(-1)

    elif features_type == 'features':
        assert clusters_num is not None, "clusters_num must be provided when features_type is 'features'"

        k_means = KMeans(n_clusters=clusters_num)
        pca = PCA(n_components=20)

        train_features = pca.fit_transform(train_features)
        test_features = pca.transform(test_features)
        
        train_clusters = k_means.fit_predict(train_features)
        cluster_centers = k_means.cluster_centers_

        train_distribution = [
            np.sqrt(np.sum((train_features[train_clusters == i] - cluster_centers[i][np.newaxis, :])**2, axis=-1)).reshape(-1)
            for i in range(clusters_num)
        ]

        dists = np.sqrt(np.sum((test_features[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :])**2, axis=-1))
        test_clusters = dists.argmin(-1)
        test_values = dists.min(-1)

    scores = [
        get_sample_stat(test_values[i], test_clusters[i], train_distribution, criterion) 
        for i in range(test_features.shape[0])
    ]

    return np.array(scores)

def metrics(scores: np.ndarray, targets: np.ndarray, threshold: float) -> dict:
    """
    Calculate evaluation metrics.

    Args:
        scores (np.ndarray): Predicted scores.
        targets (np.ndarray): True labels.
        threshold (float): Threshold for binary classification.

    Returns:
        dict: Dictionary containing 'roc_auc' and 'f1' scores.
    """
    roc_auc = roc_auc_score(targets, scores)
    f1 = f1_score(targets, scores > threshold)
    
    return {
        'roc_auc': roc_auc,
        'f1': f1,
    }