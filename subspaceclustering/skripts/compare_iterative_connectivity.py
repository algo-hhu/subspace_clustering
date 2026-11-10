import os
import pickle

from matplotlib import pyplot as plt


def compare_iterative_merging(output_dir: str):
    """
    for k-means clustering, ward, thompson, aggl connected clustering with both distance functions, compare the
    iterative merging
    :return:
    """
    # read pkl files with summed distances to subspaces per iteration
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/k_means_clustering_with_connectivity_euclidean_distance"
            "/subspace_clustering_8"
            "/establish_connectivity_every_iteration/components_15/sum_distances_to_subspaces.pkl",
            "rb") as infile:
        k_means_euclidean = pickle.load(infile)
    # wards method with connectivity
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/wards_method_connected_distance_for_wards_method"
            "/subspace_clustering_8/establish_connectivity_every_iteration/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        wards_euclidean = pickle.load(infile)
    # thompson with connectivity
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/agglomerative_clustering_spatio_temporal_distance_function"
            "/subspace_clustering_8/establish_connectivity_every_iteration/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        thompson_spatio_temporal = pickle.load(infile)
    # aggl con euclidean
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/agglomerative_connected_clustering_euclidean_distance"
            "/subspace_clustering_8/establish_connectivity_every_iteration/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        aggl_connected_euclidean = pickle.load(infile)
    # aggl con spatio temporal
    with open(
            "../output/results_1_2/filter_500/2_degree_grid"
            "/agglomerative_connected_clustering_spatio_temporal_distance_function"
            "/subspace_clustering_8/establish_connectivity_every_iteration/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        aggl_connected_spatio_temporal = pickle.load(infile)
    # plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(k_means_euclidean.keys(), k_means_euclidean.values(), marker='o', label="Conn-KMeans++",
             color='darkgreen',
             linewidth=2)
    plt.plot(wards_euclidean.keys(), wards_euclidean.values(), marker='x', label="Conn-Ward", color='blue',
             linewidth=2)
    plt.plot(thompson_spatio_temporal.keys(), thompson_spatio_temporal.values(), marker='s', label="Agglo-ST",
             color='yellow', linewidth=2)
    plt.plot(aggl_connected_euclidean.keys(), aggl_connected_euclidean.values(), marker='^',
             label="Conn-Agglo-Euc",
             color='red', linewidth=2)
    plt.plot(aggl_connected_spatio_temporal.keys(), aggl_connected_spatio_temporal.values(), marker='d', color='orange',
             label="Conn-Agglo-ST", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Sum of squared distances to subspaces")
    plt.legend()
    # plt.title("Iterative Merging - K-Means with Euclidean Distance")
    plt.grid()
    plt.savefig(
        f"{output_dir}/iterative_merging.jpg", bbox_inches='tight', dpi=300)
    plt.close()
    return


def compare_merging_once(outdir: str):
    """
    Compare the merging once for k-means, wards, thompson and aggl connected clustering
    :param outdir:
    :return:
    """
    # read pkl files with summed distances to subspaces per iteration
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/k_means_clustering_with_connectivity_euclidean_distance"
            "/subspace_clustering_8/establish_connectivity_once/components_15/sum_distances_to_subspaces.pkl",
            "rb") as infile:
        k_means_euclidean = pickle.load(infile)
    # wards method with connectivity
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/wards_method_connected_distance_for_wards_method"
            "/subspace_clustering_8/establish_connectivity_once/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        wards_euclidean = pickle.load(infile)
    # thompson with connectivity
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/agglomerative_clustering_spatio_temporal_distance_function"
            "/subspace_clustering_8/establish_connectivity_once/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        thompson_spatio_temporal = pickle.load(infile)
    # aggl con euclidean
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/agglomerative_connected_clustering_euclidean_distance"
            "/subspace_clustering_8/establish_connectivity_once/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        aggl_connected_euclidean = pickle.load(infile)
    # aggl con spatio temporal
    with open(
            "../output/results_1_2/filter_500/2_degree_grid"
            "/agglomerative_connected_clustering_spatio_temporal_distance_function"
            "/subspace_clustering_8/establish_connectivity_once/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        aggl_connected_spatio_temporal = pickle.load(infile)

    # plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(k_means_euclidean.keys(), k_means_euclidean.values(), marker='o', label="Conn-KMeans++",
             color='darkgreen',
             linewidth=2)
    plt.plot(wards_euclidean.keys(), wards_euclidean.values(), marker='x', label="Conn-Ward", color='blue',
             linewidth=2)
    plt.plot(thompson_spatio_temporal.keys(), thompson_spatio_temporal.values(), marker='s', label="Agglo-ST",
             color='yellow', linewidth=2)
    plt.plot(aggl_connected_euclidean.keys(), aggl_connected_euclidean.values(), marker='^',
             label="Conn-Agglo-Euc",
             color='red', linewidth=2)
    plt.plot(aggl_connected_spatio_temporal.keys(), aggl_connected_spatio_temporal.values(), marker='d',
             label="Conn-Agglo-ST", color='orange', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Sum of squared distances to subspaces")
    plt.legend()
    # plt.title("Iterative Merging - K-Means with Euclidean Distance")
    plt.grid()
    plt.savefig(
        f"{outdir}/iterative_merging_once.jpg", bbox_inches='tight', dpi=300)
    plt.close()


def iteratively_filtering(outdir: str):
    """
    Compare the filtering and merging once for k-means, wards, thompson and aggl connected clustering
    :param outdir:
    :return:
    """
    # read pkl files with summed distances to subspaces per iteration
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/k_means_clustering_with_connectivity_euclidean_distance"
            "/subspace_clustering_8/filter_every_round_connectivity_once/components_15/sum_distances_to_subspaces.pkl",
            "rb") as infile:
        k_means_euclidean = pickle.load(infile)
    # wards method with connectivity
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/wards_method_connected_distance_for_wards_method"
            "/subspace_clustering_8/filter_every_round_connectivity_once/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        wards_euclidean = pickle.load(infile)
    # thompson with connectivity
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/agglomerative_clustering_spatio_temporal_distance_function"
            "/subspace_clustering_8/filter_every_round_connectivity_once/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        thompson_spatio_temporal = pickle.load(infile)
    # aggl con euclidean
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/agglomerative_connected_clustering_euclidean_distance"
            "/subspace_clustering_8/filter_every_round_connectivity_once/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        aggl_connected_euclidean = pickle.load(infile)
    # aggl con spatio temporal
    with open(
            "../output/results_1_2/filter_500/2_degree_grid"
            "/agglomerative_connected_clustering_spatio_temporal_distance_function"
            "/subspace_clustering_8/filter_every_round_connectivity_once/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        aggl_connected_spatio_temporal = pickle.load(infile)

    # plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(k_means_euclidean.keys(), k_means_euclidean.values(), marker='o', label="Con-KMeans++",
             color='darkgreen',
             linewidth=2)
    plt.plot(wards_euclidean.keys(), wards_euclidean.values(), marker='x', label="Conn-Ward", color='blue',
             linewidth=2)
    plt.plot(thompson_spatio_temporal.keys(), thompson_spatio_temporal.values(), marker='s', label="Agglo-ST",
             color='yellow', linewidth=2)
    plt.plot(aggl_connected_euclidean.keys(), aggl_connected_euclidean.values(), marker='^',
             label="Conn-Agglo-Euc",
             color='red', linewidth=2)
    plt.plot(aggl_connected_spatio_temporal.keys(), aggl_connected_spatio_temporal.values(), marker='d',
             label="Conn-Agglo-ST", linewidth=2, color='orange')
    plt.xlabel("Iteration")
    plt.ylabel("Sum of squared distances to subspaces")
    plt.legend()
    # plt.title("Iterative Merging - K-Means with Euclidean Distance")
    plt.grid()
    plt.savefig(f"{outdir}/iterative_filtering.jpg", bbox_inches='tight', dpi=300)
    plt.close()


def integrated_connectivity(outdir: str):
    """
    Compare the integrated connectivity for k-means, wards, thompson and aggl connected clustering
    :param outdir:
    :return:
    """
    # read pkl files with summed distances to subspaces per iteration
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/k_means_clustering_with_connectivity_euclidean_distance"
            "/subspace_clustering_8/integrated_connectivity/components_15/sum_distances_to_subspaces.pkl",
            "rb") as infile:
        k_means_euclidean = pickle.load(infile)
    # wards method with connectivity
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/wards_method_connected_distance_for_wards_method"
            "/subspace_clustering_8/integrated_connectivity/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        wards_euclidean = pickle.load(infile)
    # thompson with connectivity
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/agglomerative_clustering_spatio_temporal_distance_function"
            "/subspace_clustering_8/integrated_connectivity/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        thompson_spatio_temporal = pickle.load(infile)
    # aggl con euclidean
    with open(
            "../output/results_1_2/filter_500/2_degree_grid/agglomerative_connected_clustering_euclidean_distance"
            "/subspace_clustering_8/integrated_connectivity/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        aggl_connected_euclidean = pickle.load(infile)
    # aggl con spatio temporal
    with open(
            "../output/results_1_2/filter_500/2_degree_grid"
            "/agglomerative_connected_clustering_spatio_temporal_distance_function"
            "/subspace_clustering_8/integrated_connectivity/components_15/"
            "sum_distances_to_subspaces.pkl", "rb") as infile:
        aggl_connected_spatio_temporal = pickle.load(infile)

    # plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(k_means_euclidean.keys(), k_means_euclidean.values(), marker='o', label="Conn-KMeans++",
             color='darkgreen',
             linewidth=2)
    plt.plot(wards_euclidean.keys(), wards_euclidean.values(), marker='x', label="Conn-Ward", color='blue',
             linewidth=2)
    plt.plot(thompson_spatio_temporal.keys(), thompson_spatio_temporal.values(), marker='s', label="Agglo-ST",
             color='yellow', linewidth=2)
    plt.plot(aggl_connected_euclidean.keys(), aggl_connected_euclidean.values(), marker='^', color='red',
             label="Conn-Agglo-Euc", linewidth=2)
    plt.plot(aggl_connected_spatio_temporal.keys(), aggl_connected_spatio_temporal.values(), marker='d',
             label="Conn-Agglo-ST", color='orange', linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Sum of squared distances to subspaces")
    plt.legend()
    # plt.title("Iterative Merging - K-Means with Euclidean Distance")
    plt.grid()
    plt.savefig(f"{outdir}/IntegratedConn.jpg", bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    out_dir = "../output/comparison/merging"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    compare_iterative_merging(out_dir)
    compare_merging_once(out_dir)
    iteratively_filtering(out_dir)
    integrated_connectivity(out_dir)
