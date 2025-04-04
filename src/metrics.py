import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score,  accuracy_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
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

        print(test_features.max(), train_features.max())

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

def get_metrics(scores: np.ndarray, targets: np.ndarray, threshold: float, prefix: str) -> dict:
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
    precision = precision_score(targets, scores > threshold)
    recall = recall_score(targets, scores > threshold)
    accuracy = accuracy_score(targets, scores > threshold)
    
    return {
        f"{prefix}_roc_auc": roc_auc,
        f"{prefix}_f1": f1,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,  
        f"{prefix}_accuracy": accuracy
    }


def cluster_features(train_features, test_features, clusters_num):
    k_means = KMeans(n_clusters=clusters_num)
    pca = PCA(n_components=20)

    train_features = pca.fit_transform(train_features)
    test_features = pca.transform(test_features)
    
    train_clusters = k_means.fit_predict(train_features)
    test_clusters = k_means.predict(test_features)
    cluster_centers = k_means.cluster_centers_

    return train_features, test_features, train_clusters, test_clusters, cluster_centers


def clusters_metrics(train_features, test_features, clusters_num, prefix):
    
    train_features,\
    test_features,\
    train_clusters,\
    test_clusters, cluster_centers = cluster_features(train_features, test_features, clusters_num)

    silhouette_train = silhouette_score(train_features, train_clusters)
    silhouette_test = silhouette_score(test_features, test_clusters)

    davies_bouldin_train = davies_bouldin_score(train_features, train_clusters)
    davies_bouldin_test = davies_bouldin_score(test_features, test_clusters)

    calinski_harabasz_train = calinski_harabasz_score(train_features, train_clusters)
    calinski_harabasz_test = calinski_harabasz_score(test_features, test_clusters)
    
    metrics = {
        f"{prefix}_silhouette_train": silhouette_train,
        f"{prefix}_silhouette_test": silhouette_test,

        f"{prefix}_davies_bouldin_train": davies_bouldin_train,
        f"{prefix}_davies_bouldin_test": davies_bouldin_test,

        f"{prefix}_calinski_harabasz_train": calinski_harabasz_train,
        f"{prefix}_calinski_harabasz_test": calinski_harabasz_test
    }

    return metrics
    
    
def get_supervised_metrics_features(train_features, test_features, targets, clusters_numbers):
    all_metrics = {}
    
    for clusters_number in clusters_numbers:
        scores = get_scores(
            train_features, 
            test_features, 
            criterion = 'right-sided',
            features_type ='features',
            clusters_num = clusters_number)

        scores = 1 - scores
        
        all_metrics = all_metrics | get_metrics(scores, targets, 0.95, clusters_number)
    
    return all_metrics


def get_supervised_metrics_probas(train_features, test_features, targets, clusters_numbers):
    all_metrics = {}
    
    scores = get_scores(
        train_features, 
        test_features, 
        criterion = 'left-sided',
        features_type ='clusters_probas')

    scores = 1 - scores
        
    all_metrics = get_metrics(scores, targets, 0.95, 'probas')
    
    return all_metrics

def get_unsupervised_metrics_features(train_features, test_features, clusters_numbers):
    all_metrics = {}
    for clusters_num in clusters_numbers:
        all_metrics = all_metrics | clusters_metrics(train_features, test_features, clusters_num, clusters_num)

    return all_metrics





        
    