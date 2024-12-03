def merge_clusters(cluster_1, cluster_2):
    """
    Merge two clusters
    :param cluster_1: Cluster object
    :param cluster_2: Cluster object
    :return: Cluster object
    """
    cluster_1.merge(cluster_2)
    return cluster_1


def update_differences():
    pass
