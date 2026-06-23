import os

import numpy as np
import xarray
import xarray as xr
from loguru import logger
from sklearn.decomposition import PCA

from src import plotting
from src.helper import extract_clusters_from_xarray_dataset

# Seed for PCA's randomized SVD solver; set from GlobalSettings.random_seed in main() for reproducibility.
PCA_RANDOM_STATE = 42


def plot_first_two_components(clustering: dict[int: list[tuple[float, float]]], sla_data: np.array, output_dir: str,
                              sea_level_anomaly_data: xarray.Dataset):
    """
    For each cluster: calculate PCA and plot the time series of the first component
    :param sea_level_anomaly_data:
    :param sla_data:
    :param output_dir:
    :param clustering:
    :return:
    """
    first_component_per_cluster = {}
    second_component_per_cluster = {}
    for cluster_id, cluster_elements in clustering.items():
        # extract the time series for the current cluster as input data for PCA
        index_to_gridpoint = {}
        counter = 0
        data_for_pca = np.zeros((len(cluster_elements), sla_data.shape[0]))
        for idx, idy in cluster_elements:
            index_to_gridpoint[counter] = (idx, idy)
            time_series = sla_data[:, idx, idy]
            data_for_pca[counter] = time_series
            counter += 1
        U, s, Vt = np.linalg.svd(data_for_pca)
        if len(s) < 2:
            logger.warning(f"Cluster {cluster_id} has less than 2 components. Skipping.")
            continue
        first_pc = s[0] * Vt[0, :]
        first_component_per_cluster[cluster_id] = first_pc
        # plot the first component
        plotting.plot_time_series(first_pc, f"{output_dir}/first_component",
                                  f"cluster_{cluster_id}_first_component",
                                  sea_level_anomaly_data)
        first_EOF = U[:, 0]
        # plot the first EOF, each grid point has a different value
        eof_plot = np.nan * np.ones(sla_data.shape[1:])  # 2d array with shape (latitude, longitude)
        for index, grid_point in index_to_gridpoint.items():
            eof_plot[grid_point] = first_EOF[index]
        plotting.plot_eof(eof_plot, f"{output_dir}/first_component", f"cluster_{cluster_id}_first_EOF")

        # plot second component
        second_pc = s[1] * Vt[1, :]
        second_component_per_cluster[cluster_id] = second_pc
        plotting.plot_time_series(second_pc, f"{output_dir}/second_component",
                                  f"cluster_{cluster_id}_second_component",
                                  sea_level_anomaly_data)
        second_EOF = U[:, 1]
        # plot the second EOF, each grid point has a different value
        eof_plot = np.nan * np.ones(sla_data.shape[1:])  # 2d array with shape (latitude, longitude)
        for index, grid_point in index_to_gridpoint.items():
            eof_plot[grid_point] = second_EOF[index]
        plotting.plot_eof(eof_plot, f"{output_dir}/second_component", f"cluster_{cluster_id}_second_EOF")

    # determine mean first component
    first_component_mean = np.mean(list(first_component_per_cluster.values()), axis=0)
    # plot the mean first component
    plotting.plot_time_series(first_component_mean, output_dir, f"mean_first_component", sea_level_anomaly_data)
    return


def plot_first_component_for_entire_dataset(output_dir: str, sla_data: np.array, min_lat: float, min_lon: float,
                                            resolution: float, sea_level_anomaly_data: xarray.Dataset):
    """
    Plot the first component for the entire dataset
    :param output_dir:
    :param sla_data:
    :param min_lat:
    :param min_lon:
    :param resolution:
    :return:
    """
    grid_point_to_index = {}
    counter = 0
    data_for_pca = np.zeros((sla_data.shape[1] * sla_data.shape[2], sla_data.shape[0]))
    for idx in range(sla_data.shape[1]):
        for idy in range(sla_data.shape[2]):
            if np.isnan(sla_data[:, idx, idy]).any():
                continue
            grid_point_to_index[counter] = (idx, idy)
            time_series = sla_data[:, idx, idy]
            data_for_pca[counter] = time_series
            counter += 1
    pca = PCA(n_components=1, random_state=PCA_RANDOM_STATE)
    pca.fit(data_for_pca)
    first_component = pca.components_[0]
    # plot the first component
    plotting.plot_time_series(first_component, output_dir, f"entire_dataset_first_component", sea_level_anomaly_data)
    pass


def start_evaluation(clustering: xarray.Dataset, output_dir: str, sea_level_anomaly_data: xarray.Dataset):
    """
    Evaluate the current clustering
    :param
    :return:
    """
    sla_data = sea_level_anomaly_data["sla"].values
    min_lat = clustering["latitude"].values.min()
    min_lon = clustering["longitude"].values.min()
    resolution = float(clustering["latitude"].values[1]) - float(clustering["latitude"].values[0])
    cluster_id_to_lat_lon, cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(clustering, min_lat,
                                                                                              min_lon, resolution,
                                                                                              sla_data)
    plot_first_two_components(cluster_id_to_grid_point_id, sla_data, output_dir, sea_level_anomaly_data)
    plot_first_component_for_entire_dataset(output_dir, sla_data, min_lat, min_lon, resolution, sea_level_anomaly_data)
    return


def evaluate_clustering(evaluation_settings, out_dir: str,
                        unfiltered_sea_level_anomaly_data: xr.Dataset, subspace_clustering_settings):
    """
    Evaluate clustering
    :param subspace_clustering_settings:
    :param evaluation_settings:
    :param out_dir:
    :param unfiltered_sea_level_anomaly_data:
    :return:
    """
    if evaluation_settings.do_evaluation:
        options = ("establish_connectivity_every_iteration", "establish_connectivity_once",
                   "filter_every_round_connectivity_once", "integrated_connectivity")
        for connectivity_option in options:
            for number_of_components in subspace_clustering_settings.number_of_components:
                current_out_dir = (f"{out_dir}/subspace_clustering_{subspace_clustering_settings.number_of_clusters}/"
                                   f"{connectivity_option}/components_{number_of_components}/evaluation")
                eval_clustering_path = (f"{out_dir}/s"
                                        f"ubspace_clustering_{subspace_clustering_settings.number_of_clusters}/{
                                        connectivity_option}/components_{number_of_components}/clustering_"
                                        f"{evaluation_settings.number_of_clusters}.nc")
                if not os.path.exists(current_out_dir):
                    os.makedirs(current_out_dir)
                logger.info(f"output directory: {current_out_dir}")
                if not os.path.exists(eval_clustering_path):
                    logger.warning(f"Clustering file {eval_clustering_path} does not exist. Skipping evaluation.")
                    continue
                clustering = xr.open_dataset(eval_clustering_path)
                start_evaluation(clustering, current_out_dir, unfiltered_sea_level_anomaly_data)
