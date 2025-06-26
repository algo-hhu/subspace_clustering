import numpy as np
import xarray
from sklearn.decomposition import PCA

from src import plotting
from src.helper import extract_clusters_from_xarray_dataset


def plot_first_component(clustering: dict[int: list[tuple[float, float]]], sla_data: np.array, output_dir: str):
    """
    For each cluster: calculate PCA and plot the time series of the first component
    :param sla_data:
    :param output_dir:
    :param clustering:
    :return:
    """
    first_component_per_cluster = {}
    for cluster_id, cluster_elements in clustering.items():
        print(f"plotting first component for cluster {cluster_id}")
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
        first_component = Vt[0]
        first_component_per_cluster[cluster_id] = first_component
        # plot the first component
        plotting.plot_time_series(first_component, output_dir, f"cluster_{cluster_id}_first_component")
        first_EOF = U[:, 0]
        # plot the first EOF, each grid point has a different value
        eof_plot = np.nan * np.ones(sla_data.shape[1:])  # 2d array with shape (latitude, longitude)
        for index, grid_point in index_to_gridpoint.items():
            eof_plot[grid_point] = first_EOF[index]
        plotting.plot_eof(eof_plot, output_dir, f"cluster_{cluster_id}_first_EOF")

    # determine mean first component
    first_component_mean = np.mean(list(first_component_per_cluster.values()), axis=0)
    # plot the mean first component
    plotting.plot_time_series(first_component_mean, output_dir, f"mean_first_component")
    return


def plot_first_component_for_entire_dataset(output_dir: str, sla_data: np.array, min_lat: float, min_lon: float,
                                            resolution: float):
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
    pca = PCA(n_components=1)
    pca.fit(data_for_pca)
    first_component = pca.components_[0]
    # plot the first component
    plotting.plot_time_series(first_component, output_dir, f"entire_dataset_first_component")
    pass


def start_evaluation(clustering: xarray.Dataset, output_dir: str, sea_level_anomaly_data: xarray.Dataset):
    """
    Evaluate the current clustering
    :param
    :return:
    """
    sla_data = sea_level_anomaly_data["sla"].values
    cluster_data = clustering["__xarray_dataarray_variable__"].values
    min_lat = clustering["latitude"].values.min()
    min_lon = clustering["longitude"].values.min()
    resolution = float(clustering["latitude"].values[1]) - float(clustering["latitude"].values[0])
    cluster_id_to_lat_lon, cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(clustering, min_lat,
                                                                                              min_lon, resolution,
                                                                                              sla_data)
    plot_first_component(cluster_id_to_grid_point_id, sla_data, output_dir)
    plot_first_component_for_entire_dataset(output_dir, sla_data, min_lat, min_lon, resolution)
    return
