import os

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans, AgglomerativeClustering

from src import plotting
from src.helper import extract_clusters_from_xarray_dataset


def plot_thompson_clustering():
    """ Plot the results of the thompson clustering with and without filtering."""
    min_lat, min_lon, resolution, sla_data_array = load_filtered_sla()
    # plot thompson clustering results with and without filtering
    outdir = "../output/thompson_clustering_img/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    thompson_dataset_with_filter = xr.open_dataset(
        "../output/results_1_1/filter_500/2_degree_grid/agglomerative_clustering_spatio_temporal_distance_function/clustering_25.nc")

    thompson_dataset_without_filter = xr.open_dataset(
        "../output/results_1_1/no_filtering/2_degree_grid/agglomerative_clustering_spatio_temporal_distance_function/clustering_25.nc")

    colors = assign_colors()
    thompson_clustering_with_filter, _ = extract_clusters_from_xarray_dataset(thompson_dataset_with_filter, min_lat,
                                                                              min_lon, resolution, sla_data_array)
    thompson_clustering_without_filter, _ = extract_clusters_from_xarray_dataset(thompson_dataset_without_filter,
                                                                                 min_lat,
                                                                                 min_lon, resolution, sla_data_array)

    plotting.plot_clustering(thompson_clustering_with_filter, outdir, 2, "thompson_25_clusters_with_filtering", colors)
    plotting.plot_clustering(thompson_clustering_without_filter, outdir, 2, "thompson_25_clusters_without_filtering",
                             colors)


def assign_colors():
    # named colors with decent contrast
    colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red", 4: "tab:purple", 5: "tab:brown",
              6: "tab:pink", 7: "tab:gray", 8: "tab:olive", 9: "tab:cyan", 10: "blue", 11: "darkorange",
              12: "forestgreen", 13: "firebrick", 14: "mediumpurple", 15: "saddlebrown", 16: "deeppink", 17: "dimgray",
              18: "olivedrab", 19: "deepskyblue", 20: "gold", 21: "limegreen", 22: "indigo", 23: "coral",
              24: "turquoise", 25: "mediumvioletred", 26: "peru", 27: "dodgerblue", 28: "darkgreen", 29: "tomato"}
    return colors


def load_filtered_sla():
    """
    Load the sea level anomaly data from the netCDF file and return the minimum latitude, minimum longitude,
    :return:
    """
    # load sla
    sla_data = xr.open_dataset("../output/resolutions/sea_level_anomaly_data_filtered_500_2_degree.nc")
    sla_data_array = sla_data["sla"].values
    time, indices_x, indices_y = sla_data_array.shape
    min_lat = sla_data.latitude.min().values
    min_lon = sla_data.longitude.min().values
    resolution = float(sla_data.latitude[1].values - sla_data.latitude[0].values)
    return min_lat, min_lon, resolution, sla_data_array


def load_unfiltered_sla():
    """
    Load the sea level anomaly data from the netCDF file and return the minimum latitude, minimum longitude,
    :return:
    """
    # load sla
    sla_data = xr.open_dataset("../output/resolutions/sea_level_anomaly_data_no_filter_2_degree.nc")
    sla_data_array = sla_data["sla"].values
    time, indices_x, indices_y = sla_data_array.shape
    min_lat = sla_data.latitude.min().values
    min_lon = sla_data.longitude.min().values
    resolution = float(sla_data.latitude[1].values - sla_data.latitude[0].values)
    return min_lat, min_lon, resolution, sla_data_array


def calculate_kmeans_without_connectivity(min_lat, min_lon, resolution, sla_data_array):
    """
    Calculate k-means without connectivity.
    :param min_lat: minimum latitude of the data
    :param min_lon: minimum longitude of the data
    :param resolution: resolution of the data
    :param sla_data_array: sea level anomaly data array
    :return: k-means clustering result
    """

    index_to_grid_point, index_to_lat_lon, time_series_data, nan_mask = make_input_data(min_lat, min_lon, resolution,
                                                                                        sla_data_array)
    k_means = KMeans(n_clusters=25, random_state=0)
    labels = k_means.fit_predict(time_series_data.T)  # Transpose, because KMeans expects (n_samples, n_features)
    clustering_data_array = np.full(nan_mask.shape, np.nan)
    cluster_id_to_lat_lon = {}
    cluster_to_grid_point_dict = {}
    for i, label in enumerate(labels):
        idx, idy = index_to_grid_point[i]
        clustering_data_array[idx, idy] = float(label)
        try:
            cluster_id_to_lat_lon[float(label)].append(index_to_lat_lon[i])
            cluster_to_grid_point_dict[float(label)].append((idx, idy))
        except KeyError:
            cluster_id_to_lat_lon[float(label)] = [index_to_lat_lon[i]]
            cluster_to_grid_point_dict[float(label)] = [(idx, idy)]
    return cluster_id_to_lat_lon


def make_input_data(min_lat, min_lon, resolution, sla_data_array):
    """
    Prepare the input data for clustering by extracting time series data from the sea level anomaly array.
    :param min_lat:
    :param min_lon:
    :param resolution:
    :param sla_data_array:
    :return:
    """
    nan_mask = np.isnan(sla_data_array).any(axis=0)
    number_of_grid_points_with_valid_data = np.sum(~nan_mask)
    time_series_data = np.full((sla_data_array.shape[0], number_of_grid_points_with_valid_data), np.nan)
    index_to_grid_point = {}
    index_to_lat_lon = {}
    counter = 0
    lat_size = sla_data_array.shape[1]
    lon_size = sla_data_array.shape[2]
    for idx in range(lat_size):
        for idy in range(lon_size):
            current_time_series_data = sla_data_array[:, idx, idy]
            if np.isnan(current_time_series_data).any():
                continue
            else:
                time_series_data[:, counter] = current_time_series_data
                index_to_grid_point[counter] = (idx, idy)
                index_to_lat_lon[counter] = (min_lat + idx * resolution, min_lon + idy * resolution)
                counter += 1
    return index_to_grid_point, index_to_lat_lon, time_series_data, nan_mask


def calculate_wards_without_connectivity(min_lat, min_lon, resolution, sla_data_array):
    """
    Calculate Wards method without connectivity.
    :param min_lat:
    :param min_lon:
    :param resolution:
    :param sla_data_array:
    :return:
    """
    index_to_grid_point, index_to_lat_lon, time_series_data, nan_mask = make_input_data(min_lat, min_lon, resolution,
                                                                                        sla_data_array)
    model = AgglomerativeClustering(n_clusters=25, linkage="ward", connectivity=None)
    labels = model.fit_predict(time_series_data.T)
    cluster_dict_with_grid_points = {}
    cluster_dict_with_lat_lon = {}
    for index, label in enumerate(labels):
        grid_point = index_to_grid_point[index]
        lat = min_lat + grid_point[0] * resolution
        lon = min_lon + grid_point[1] * resolution
        if label not in cluster_dict_with_grid_points:
            cluster_dict_with_grid_points[label] = []
            cluster_dict_with_lat_lon[label] = []
        cluster_dict_with_grid_points[label].append(grid_point)
        cluster_dict_with_lat_lon[label].append((lat, lon))
    return cluster_dict_with_lat_lon


def plot_initial_images():
    """
    plot greedy k-means ++ without reconnecting, connected wards method, wards method unconnected,
    connected subspace clustering - all with 25 clusters
    :return:
    """
    min_lat, min_lon, resolution, sla_data_array = load_unfiltered_sla()
    outdir = "../output/initial_images/"
    colors = assign_colors()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # calculate k-means without connectivity
    kmeans_without_connectivity = calculate_kmeans_without_connectivity(min_lat, min_lon, resolution,
                                                                        sla_data_array)
    ward_without_connectivity = calculate_wards_without_connectivity(min_lat, min_lon, resolution,
                                                                     sla_data_array)

    plotting.plot_clustering(kmeans_without_connectivity, outdir, 2, "greedy_kmeans_25_clusters_no_filtering_no_conn",
                             colors)
    plotting.plot_clustering(ward_without_connectivity, outdir, 2, "wards_25_clusters_no_filtering_no_conn", colors)
    # read clustering result from wards method with connectivity
    ward_dataset = xr.open_dataset(
        "../output/results_1_2/no_filtering/2_degree_grid/wards_method_connected_distance_for_wards_method/clustering_25.nc")
    ward_with_connectivity, _ = extract_clusters_from_xarray_dataset(ward_dataset, min_lat, min_lon, resolution,
                                                                     sla_data_array)
    plotting.plot_clustering(ward_with_connectivity, outdir, 2, "wards_25_clusters_no_filtering_with_conn", colors)

    # read clustering result from subspace clustering with connectivity
    subspace_dataset = xr.open_dataset(
        "../output/results_1_2/no_filtering/2_degree_grid/agglomerative_connected_clustering_euclidean_distance/clustering_25.nc")
    subspace_with_connectivity, _ = extract_clusters_from_xarray_dataset(subspace_dataset, min_lat, min_lon, resolution,
                                                                         sla_data_array)
    plotting.plot_clustering(subspace_with_connectivity, outdir, 2, "subspace_25_clusters_no_filtering_with_conn",
                             colors)
    return


if __name__ == "__main__":
    # plot_thompson_clustering()
    plot_initial_images()
