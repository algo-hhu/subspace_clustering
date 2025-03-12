# TODO: For each cluster compute the subspace from the time series of the grid points in the cluster
# TODO: For each point determine the subspace it is closest to
# TODO: Ensure connectedness of the clusters by doing a BFS
# TODO: Repeat everything until clusters are stable
import igraph as ig
import numpy as numpy
import xarray
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src import plotting, helper


def generate_subspace(cluster_grid_point_ids: [(int, int)], data: numpy.ndarray, number_of_components: int):
    """
    Generate the subspace of a cluster
    :param cluster_grid_point_ids:
    :param number_of_components:
    :param data:
    :return:
    """
    # create data matrix X = N x T where T is the number of time steps and N is the number of grid points
    if len(cluster_grid_point_ids) <= number_of_components:
        return None
    time_length = data.shape[0]
    data_matrix = numpy.zeros((len(cluster_grid_point_ids), time_length))
    counter = 0
    for grid_point_id_x, grid_point_id_y in cluster_grid_point_ids:
        time_series = data[:, grid_point_id_x, grid_point_id_y]
        data_matrix[counter] = time_series
        counter += 1
    # mean center each time series ?
    # SVD X' = USV^T
    # U = left singular vectors, V = right singular vectors
    # U = T x T, V = N x N
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data_matrix)
    pca = PCA(number_of_components)
    pca.fit(normalized_data)
    explained_variance = pca.explained_variance_ratio_
    # round explained variance to 2 decimal places
    explained_variance = [round((explained_variance[i] * 100), 4) for i in range(len(explained_variance))]
    logger.info(f"Explained variance ratio: {explained_variance}")
    components = pca.components_
    return components


def determine_closest_subspace(data: numpy.ndarray, subspaces):
    """
    Determine the closest subspace to a cluster
    :param data:
    :param subspaces:
    :return:
    """

    assignment_graph = ig.Graph()
    grid_point_assignment = {cluster_id: [] for cluster_id in subspaces}
    for id_x in tqdm(range(data.shape[1])):
        for id_y in range(data.shape[2]):
            if numpy.isnan(data[:, id_x, id_y]).any():
                continue
            current_time_series = data[:, id_x, id_y]
            current_time_series = (current_time_series - numpy.mean(current_time_series)) / numpy.std(
                current_time_series)
            min_error = numpy.inf
            closest_cluster = None
            for cluster_id, subspace in subspaces.items():
                # project current time series onto subspace
                projection = subspace.T @ (subspace @ current_time_series)
                # use squared Euclidean distance
                residual = current_time_series - projection
                distance = numpy.linalg.norm(residual) ** 2
                # otherwise could use the norm
                # distance = numpy.linalg.norm(current_time_series - x_proj)
                if distance < min_error:
                    min_error = distance
                    closest_cluster = cluster_id
            # insert node with closest cluster as attribute
            assignment_graph.add_vertex(name=f"{id_x}_{id_y}", cluster=closest_cluster)
            grid_point_assignment[closest_cluster].append((id_x, id_y))
            # add edges between adjacent nodes that have already been generated (x-1,y-1), (x-1,y), (x,y-1), (x-1,y+1)
            # if they exist in the graph
            for i in [-1, 0]:
                for j in [-1, 0, 1]:
                    if (i == 0 and j == 0) or (i == 0 and j == 1):
                        continue
                    if 0 <= id_x + i < data.shape[1] and 0 <= id_y + j < data.shape[2]:
                        try:
                            node = assignment_graph.vs.find(f"{id_x + i}_{id_y + j}")
                        except ValueError:
                            continue
                        if node is not None:
                            assignment_graph.add_edge(f"{id_x}_{id_y}", f"{id_x + i}_{id_y + j}")

    # for each point in the cluster determine the distance to the closest subspace
    # project each time series onto each subspace
    # compute the reconstruction error
    # return a graph where each point is marked with its closest subspace
    return assignment_graph, grid_point_assignment


def start_subspace_clustering(sea_level_anomaly_data: xarray.Dataset, clustering: xarray.Dataset, out_dir: str,
                              number_of_components: int):
    """
    Start the subspace clustering
    :param out_dir:
    :param number_of_components:
    :param sea_level_anomaly_data:
    :param clustering:
    :return:
    """
    # adjust resolution, such that it is the same for the sea level anomaly data as it is for the clustering
    resolution = clustering.latitude.values[1] - clustering.latitude.values[0]
    min_lat = clustering.latitude.values[0]
    min_lon = clustering.longitude.values[0]
    # interpolate the sea level anomaly data to the resolution of the clustering
    sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=clustering.latitude.values,
                                                           longitude=clustering.longitude.values)
    sla_data = sea_level_anomaly_data["sla"].values
    # print(f"nan values in sla data at any timestep: {numpy.isnan(sla_data).any(axis=0).sum()}")
    # print(f"nan values in sla data at all timesteps: {numpy.isnan(sla_data).all(axis=0).sum()}")
    # print(f"not nan values in sla data: {numpy.count_nonzero(~numpy.isnan(sla_data).any(axis=0))}")
    # nan mask
    non_nan_mask = ~numpy.isnan(sla_data).any(axis=0)
    # apply nan mask to clustering data
    cluster_data = clustering["__xarray_dataarray_variable__"].values
    # assign nans where there are all nans in the sla data
    cluster_data[~non_nan_mask] = numpy.nan
    unique_numbers, counts = numpy.unique(cluster_data, return_counts=True)
    # how many nan values are there?
    nan_count = numpy.isnan(cluster_data).sum()
    # for number, count in zip(unique_numbers, counts):
    #     print(f"Number {number} appears {count} times")
    # print(f"Number of nan values: {nan_count}")
    # #
    extended_lats = numpy.tile(clustering["latitude"].values[:, numpy.newaxis],
                               (1, clustering["longitude"].values.shape[0]))
    extended_lons = numpy.tile(clustering["longitude"].values[numpy.newaxis, :],
                               (clustering["latitude"].values.shape[0], 1))
    cluster_dict = {}
    cluster_id_dict = {}
    for cluster_id in unique_numbers:
        if cluster_id is numpy.nan:
            continue
        current_cluster_mask = cluster_data == cluster_id
        filtered_lats = extended_lats[current_cluster_mask]
        filtered_lons = extended_lons[current_cluster_mask]
        lat_lon_pairs = list(zip(filtered_lats, filtered_lons))
        cluster_dict[cluster_id] = lat_lon_pairs
        cluster_id_dict[cluster_id] = []
        for lat_lon_pair in lat_lon_pairs:
            id_x, id_y = helper.lat_lon_to_index(lat_lon_pair[0], lat_lon_pair[1], min_lat, min_lon, resolution)
            cluster_id_dict[cluster_id].append((id_x, id_y))

    plot_clustering(cluster_dict, out_dir, resolution, name="initial_clustering")
    logger.info("determine subspaces")
    subspaces = {}
    for cluster in cluster_id_dict.keys():
        logger.info(f"cluster: {cluster}")
        current_subspace = generate_subspace(cluster_id_dict[cluster], sla_data, number_of_components)
        if current_subspace is not None:
            subspaces[cluster] = current_subspace
    assignment_graph, grid_point_assignment = determine_closest_subspace(sla_data, subspaces)
    grid_point_assignment_lat_lon = {}
    for cluster_id in grid_point_assignment.keys():
        grid_point_assignment_lat_lon[cluster_id] = []
        for grid_point in grid_point_assignment[cluster_id]:
            lat, lon = helper.index_to_lat_lon(grid_point[0], grid_point[1], min_lat, min_lon, resolution)
            grid_point_assignment_lat_lon[cluster_id].append((lat, lon))
    plot_clustering(grid_point_assignment_lat_lon, out_dir, resolution, name="grid_point_assignment")


def plot_clustering(cluster_dict, out_dir, resolution, name):
    """
    Plot the clustering
    :param cluster_dict:
    :param out_dir:
    :param resolution:
    :param name:
    :return:
    """
    cluster_colors = ["darkmagenta", "lightseagreen", "green", "gold", "orchid", "darkorange", "yellowgreen",
                      "cadetblue", "red", "yellow", "blue", "olive", "powderblue", "lavenderblush",
                      "midnightblue", "lavender", "darkslateblue", "purple", "pink", ]
    cluster_gdf, land_gdf = plotting.turn_dict_into_gdf(cluster_dict, out_dir, name, resolution / 2,
                                                        cluster_colors)
    plotting.plot_regions(land_gdf, out_dir, cluster_gdf, name)
