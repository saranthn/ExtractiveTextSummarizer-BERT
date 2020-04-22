def cluster_features(features, ratio: float = 0.2) -> List[int]:
        """
        Clusters sentences based on the ratio
        :param ratio: Ratio to use for clustering
        :return: Sentences index that qualify for summary
        """

        k = 1 if ratio * len(features) < 1 else int(len(features) * ratio)
        model = get_model(k).fit(features)
        centroids = get_centroids(model)
        cluster_args = find_closest_args(centroids, features)
        sorted_values = sorted(cluster_args.values())
        return sorted_values

def get_model(k: int):
        """
        Retrieve clustering model
        :param k: amount of clusters
        :return: Clustering model
        """
        return KMeans(n_clusters=k, random_state=12345)

def get_centroids(model):
    """
    Retrieve centroids of model
    :param model: Clustering model
    :return: Centroids
    """
    return model.cluster_centers_

def find_closest_args(centroids: np.ndarray, features):
    """
    Find the closest arguments to centroid
    :param centroids: Centroids to find closest
    :return: Closest arguments
    """

    centroid_min = 1e10
    cur_arg = -1
    args = {}
    used_idx = []

    for j, centroid in enumerate(centroids):

        for i, feature in enumerate(features):
            value = np.linalg.norm(feature - centroid)

            if value < centroid_min and i not in used_idx:
                cur_arg = i
                centroid_min = value

        used_idx.append(cur_arg)
        args[j] = cur_arg
        centroid_min = 1e10
        cur_arg = -1

    return args