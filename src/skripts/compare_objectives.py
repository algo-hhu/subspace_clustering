import os

import numpy as np
import xarray as xr

from src.clustering.subspace_clustering import evaluate_distances_to_subspaces
from src.helper import extract_clusters_from_xarray_dataset


def calculate_kmeans_objective(cluster_id_to_grid_point_id, current_sla_data_array, time_shape):
    total_k_means_distance = 0
    for cluster_id, grid_points in cluster_id_to_grid_point_id.items():
        # compute centroid of cluster
        all_time_series = np.empty((len(grid_points), time_shape), dtype=np.float32)
        for counter, grid_point in enumerate(grid_points):
            # get time series of grid point
            time_series = current_sla_data_array[:, grid_point[0], grid_point[1]]
            all_time_series[counter, :] = time_series
        centroid = np.mean(all_time_series, axis=0)
        for i in range(len(grid_points)):
            time_series = all_time_series[i, :]
            distance_squared = np.sum((time_series - centroid) ** 2)
            total_k_means_distance += distance_squared
    return total_k_means_distance


def evaluate_k_means_objective():
    """
    """
    # evaluate the results of wards algo and the merged k-means clustering regarding the objective function
    outdir = "../output/evaluate_k_means_objective/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # load sla
    sla_data = xr.open_dataset("../output/resolutions/sea_level_anomaly_data_filtered_500_2_degree.nc")
    sla_data_array = sla_data["sla"].values
    time, indices_x, indices_y = sla_data_array.shape
    min_lat = sla_data.latitude.min().values
    min_lon = sla_data.longitude.min().values
    resolution = float(sla_data.latitude[1].values - sla_data.latitude[0].values)
    cluster_sizes = [8, 10, 12, 15, 20, 25]
    wards_dist = {}
    k_means_dist = {}
    thompson_dist = {}
    aggl_conn = {}
    aggl_conn_st = {}
    for k in cluster_sizes:
        # load wards algo results
        clustering_wards_dataset = xr.open_dataset(
            f"../output/results_1_1/filter_500/2_degree_grid/wards_method_connected_distance_for_wards_method/clustering_{k}.nc")
        # load k-means with merging results
        clustering_kmeans_merged_dataset = xr.open_dataset(
            f"../output/results_1_1/filter_500/2_degree_grid/k_means_clustering_with_connectivity_euclidean_distance/clustering_{k}.nc")
        # load thompson results
        clustering_thompson_dataset = xr.open_dataset(
            f"../output/results_1_1/filter_500/2_degree_grid/agglomerative_clustering_spatio_temporal_distance_function/clustering_{k}.nc")
        aggl_con_dataset = xr.open_dataset(
            f"../output/results_1_1/filter_500/2_degree_grid/agglomerative_connected_clustering_euclidean_distance/clustering_{k}.nc")
        aggl_con_st_dataset = xr.open_dataset(
            f"../output/results_1_1/filter_500/2_degree_grid/agglomerative_connected_clustering_spatio_temporal_distance_function/clustering_{k}.nc")
        # load clustering results
        k_means_cluster_id_to_lat_lon_pairs, k_means_cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
            clustering_kmeans_merged_dataset, min_lat, min_lon, resolution,
            sla_data_array)
        wards_cluster_id_to_lat_lon_pairs, wards_cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
            clustering_wards_dataset, min_lat, min_lon, resolution,
            sla_data_array)
        thompson_cluster_id_to_lat_lon_pairs, thompson_cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
            clustering_thompson_dataset, min_lat, min_lon, resolution,
            sla_data_array)
        aggl_conn_cluster_id_to_lat_lon_pairs, aggl_conn_cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
            aggl_con_dataset, min_lat, min_lon, resolution,
            sla_data_array)
        aggl_conn_st_cluster_id_to_lat_lon_pairs, aggl_conn_st_cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
            aggl_con_st_dataset, min_lat, min_lon, resolution, sla_data_array)
        # compare
        k_means_distances = calculate_kmeans_objective(
            k_means_cluster_id_to_grid_point_id, sla_data_array, time)
        wards_distances = calculate_kmeans_objective(
            wards_cluster_id_to_grid_point_id, sla_data_array, time)
        thompson_distances = calculate_kmeans_objective(thompson_cluster_id_to_grid_point_id, sla_data_array, time)
        aggl_conn_distances = calculate_kmeans_objective(aggl_conn_cluster_id_to_grid_point_id, sla_data_array, time)
        aggl_conn_st_distances = calculate_kmeans_objective(aggl_conn_st_cluster_id_to_grid_point_id, sla_data_array,
                                                            time)
        k_means_dist[k] = k_means_distances
        wards_dist[k] = wards_distances
        thompson_dist[k] = thompson_distances
        aggl_conn[k] = aggl_conn_distances
        aggl_conn_st[k] = aggl_conn_st_distances
    # plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(cluster_sizes, list(k_means_dist.values()), label="Conn-KMeans++", marker='o', color='darkgreen')
    plt.plot(cluster_sizes, list(wards_dist.values()), label="Conn-Wards", marker='x', color='blue')
    plt.plot(cluster_sizes, list(thompson_dist.values()), label="Agglo-ST", marker='s', color='yellow')
    plt.plot(cluster_sizes, list(aggl_conn.values()), label="Conn-Agglo-Euc", marker='^', color='red')
    plt.plot(cluster_sizes, list(aggl_conn_st.values()), label="Conn-Agglo-ST", marker='d',
             color='orange')
    plt.xlabel("Number of clusters")
    plt.ylabel("K-means objective function value")
    # plt.title("K-means Objective Function Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "k_means_objective_comparison_filter_500.jpg"), dpi=300)
    plt.close()

    # evaluate regarding our clustering objective function, which is the sum of squared distances to the 15 dimensional subspace spanned by the cluster
    k_means_sc = {}
    wards_sc = {}
    thompson_sc = {}
    aggl_conn_sc = {}
    aggl_conn_st = {}
    for k in cluster_sizes:
        # load wards algo results
        clustering_wards_dataset = xr.open_dataset(
            f"../output/results_1_1/filter_500/2_degree_grid/wards_method_connected_distance_for_wards_method/clustering_{k}.nc")
        # load k-means with merging results
        clustering_kmeans_merged_dataset = xr.open_dataset(
            f"../output/results_1_1/filter_500/2_degree_grid/k_means_clustering_with_connectivity_euclidean_distance/clustering_{k}.nc")
        # load thompson results
        clustering_thompson_dataset = xr.open_dataset(
            f"../output/results_1_1/filter_500/2_degree_grid/agglomerative_clustering_spatio_temporal_distance_function/clustering_{k}.nc")
        aggl_con_dataset = xr.open_dataset(
            f"../output/results_1_1/filter_500/2_degree_grid/agglomerative_connected_clustering_euclidean_distance/clustering_{k}.nc")
        aggl_con_st_dataset = xr.open_dataset(
            f"../output/results_1_1/filter_500/2_degree_grid/agglomerative_connected_clustering_spatio_temporal_distance_function/clustering_{k}.nc")
        # load clustering results
        k_means_cluster_id_to_lat_lon_pairs, k_means_cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
            clustering_kmeans_merged_dataset, min_lat, min_lon, resolution,
            sla_data_array)
        wards_cluster_id_to_lat_lon_pairs, wards_cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
            clustering_wards_dataset, min_lat, min_lon, resolution,
            sla_data_array)
        thompson_cluster_id_to_lat_lon_pairs, thompson_cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
            clustering_thompson_dataset, min_lat, min_lon, resolution,
            sla_data_array)
        aggl_conn_cluster_id_to_lat_lon_pairs, aggl_conn_cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
            aggl_con_dataset, min_lat, min_lon, resolution,
            sla_data_array)
        aggl_conn_st_cluster_id_to_lat_lon_pairs, aggl_conn_st_cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
            aggl_con_st_dataset, min_lat, min_lon, resolution, sla_data_array)
        # compare
        print("k-means distances to subspace for k =", k)
        k_means_distances, _ = evaluate_distances_to_subspaces(
            k_means_cluster_id_to_grid_point_id, sla_data_array, 15, outdir)
        print("wards distances to subspace for k =", k)
        wards_distances, _ = evaluate_distances_to_subspaces(
            wards_cluster_id_to_grid_point_id, sla_data_array, 15, outdir)
        print("thompson distances to subspace for k =", k)
        thompson_distances, _ = evaluate_distances_to_subspaces(
            thompson_cluster_id_to_grid_point_id, sla_data_array, 15, outdir)
        print("agglomerative connected distances to subspace for k =", k)
        aggl_conn_distances, _ = evaluate_distances_to_subspaces(
            aggl_conn_cluster_id_to_grid_point_id, sla_data_array, 15, outdir)
        aggl_conn_st_distances, _ = evaluate_distances_to_subspaces(aggl_conn_st_cluster_id_to_grid_point_id,
                                                                    sla_data_array, 5, outdir)
        # save results
        k_means_sc[k] = k_means_distances
        wards_sc[k] = wards_distances
        thompson_sc[k] = thompson_distances
        aggl_conn_sc[k] = aggl_conn_distances
        aggl_conn_st[k] = aggl_conn_st_distances
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_sizes, list(k_means_sc.values()), label="Conn-KMeans++", marker='o', color='darkgreen')
    plt.plot(cluster_sizes, list(wards_sc.values()), label="Conn-Wards", marker='x', color='blue')
    plt.plot(cluster_sizes, list(thompson_sc.values()), label="Agglo-ST", marker='s', color='yellow')
    plt.plot(cluster_sizes, list(aggl_conn_sc.values()), label="Conn-Agglo-Euc", marker='^', color='red')
    plt.plot(cluster_sizes, list(aggl_conn_st.values()), label="Conn-Agglo-ST", marker='d', color='orange')
    plt.xlabel("Number of clusters")
    plt.ylabel("Sum of squared distances to subspace")
    # plt.title("Sum of Squared Distances to Subspace Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "sum_squared_distances_to_subspace_comparison_filter_500_15_dims.jpg"), dpi=300)
    plt.close()


if __name__ == "__main__":
    evaluate_k_means_objective()
