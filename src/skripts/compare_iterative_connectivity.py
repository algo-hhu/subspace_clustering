import pickle


def compare_iterative_merging():
    """
    for k-means clustering, ward, thompson, aggl connected clustering with both distance functions, compare the
    iterative merging
    :return:
    """
    # read pkl files with summed distances to subspaces per iteration
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/k_means_clustering_with_connectivity_euclidean_distance"
            "/subspace_clustering_15"
            "/establish_connectivity_every_iteration/components_15/sum_distances_to_subspaces.pkl",
            "rb") as infile:
        k_means_euclidean = pickle.load(infile)
    # plot the data

    return


if __name__ == "__main__":
    compare_iterative_merging()
