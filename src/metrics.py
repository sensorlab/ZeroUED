import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.cluster import KMeans

def get_sample_stat(
    test_value, 
    test_cluster, 
    train_distibution, 
    criterion):
    """
    Calculate test samlple statistics with respect to train_distibution.

    Args:
        test_value (float): value of the sample.
        test_cluster (int): cluster of the sample.
        train_distibution (list): list of train values distributions across clusters [num_clustres, num_samples].
        criterion (str): a way to calculate statistics of test sample. Could be 'left-sided', 'right-sided'.

    Returns:
        test samlple statistics (float).
    """

    cluster_size = len(train_distibution[test_cluster])
    
    if cluster_size == 0:
        
        return 0 if criterion == 'left-sided' else 1
        
    if criterion == 'left-sided':
        
        return sum(train_distibution[test_cluster] < test_value) / cluster_size

    elif criterion == 'right-sided':
        
        return 1 - sum(train_distibution[test_cluster] < test_value) / cluster_size


def get_scores(
    train_features, 
    test_features, 
    features_type='clusters_probas', 
    criterion='left-sided', 
    clusters_num=None):
    """
    Calculate test samlples statistics with respect to train_distibution.

    Args:
        train_features (torch.tensor): train values.
        test_features (torch.tensor): test values.
        features_type (str): if 'clusters_probas' then values are probs of samples to belong to clusters, if 'features' then values are abstract features.
        criterion (str): a way to calculate statistics of test sample. Could be 'left-sided', 'right-sided'.
        clusters_num (int | None): if features_type = 'features' then number of clusters for the data clustering.
        
    Returns:
        test samlples statistics (np.ndarray).
    """
    
    train_features, test_features = train_features.numpy(), test_features.numpy()

    if features_type == 'clusters_probas':
        
        train_clusters = train_features.argmax(-1)
    
        train_distibution = []
        
        for i in range(train_features.shape[1]):
            train_distibution.append(train_features[train_clusters == i, i])
        
        test_clusters = test_features.argmax(-1)
        test_values = test_features.max(-1)

    if features_type == 'features':

        assert not (clusters_num is None)

        k_means = KMeans(n_clusters = clusters_num)

        train_clusters = k_means.fit_predict(train_features)
        
        # (n_clustres, n_features)
        cluster_centers = k_means.cluster_centers_

        train_distibution = []
        
        for i in range(clusters_num):

            dists = np.sqrt(np.sum((
                train_features[train_clusters == i] - 
                cluster_centers[i][np.newaxis, :])**2
            ))
            
            train_distibution.append(
               dists.reshape(-1)
            )

        dists = np.sqrt(np.sum((
                test_features[:, np.newaxis, :] - 
                cluster_center[np.newaxis, :, :])**2, -1
        ))
        
        test_clusters = dists.argmin(-1)
        test_values = dists.min(-1)

        
    scores = [
        get_sample_stat(test_values[i], test_clusters[i], train_distibution, criterion) 
        for i in range(test_features.shape[0])
    ]

    return np.array(scores)

def metrics(scores, targets, treshold):
    # higher score -> higher prob that target is true
    
    roc_auc = roc_auc_score(targets, scores)
    f1 = f1_score(targets, scores > treshold)
    
    return {
        'roc_auc': roc_auc,
        'f1': f1,
    }
        

    