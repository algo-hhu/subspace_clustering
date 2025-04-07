# TODO: For each cluster compute the subspace from the time series of the grid points in the cluster
# TODO: For each point determine the subspace it is closest to
# TODO: Ensure connectedness of the clusters by doing a BFS
# TODO: Repeat everything until clusters are stable
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import numpy as numpy
import xarray
from loguru import logger
from scipy.linalg import orth
from sklearn.decomposition import PCA
from tqdm import tqdm

from src import helper, plotting
from src.plotting import plot_clustering

OUT_DIR = None
EXPLAINED_VARIANCE = []


def determine_subspace(cluster_grid_point_ids: [(int, int)], data: numpy.ndarray, number_of_components: int):
    """
    Generate the subspace of a cluster
    :param cluster_grid_point_ids:
    :param number_of_components:
    :param data:
    :return:
    """
    global EXPLAINED_VARIANCE
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
    column_means = data_matrix.mean(axis=0)
    # print(f"column means: {column_means}")
    # SVD X' = USV^T
    # U = left singular vectors, V = right singular vectors
    # U = T x T, V = N x N
    pca = PCA(number_of_components)
    pca.fit(data_matrix)
    explained_variance = pca.explained_variance_ratio_
    # print sum of explained variance for current subspace
    print(f"explained variance: {numpy.sum(explained_variance)}")
    EXPLAINED_VARIANCE[number_of_components].append(numpy.sum(explained_variance))
    # round explained variance to 2 decimal places
    explained_variance = [round((explained_variance[i] * 100), 4) for i in range(len(explained_variance))]
    # logger.info(f"Explained variance ratio: {explained_variance}")
    components = pca.components_
    # add mean back to components
    # print(f"component: {components}")
    mean = pca.mean_
    return components, mean


def determine_closest_subspace(data: numpy.ndarray, subspaces):
    """
    Determine the closest subspace to a cluster
    :param data:
    :param subspaces:
    :return:
    """

    assignment_graph = ig.Graph()
    grid_point_assignment = {}
    grid_point_assignment = {cluster_id: [] for cluster_id in subspaces}
    all_average_distances = [0] * len(subspaces)
    average_distances_to_each_subspace = {cluster_id: 0 for cluster_id in subspaces}
    number_of_data_points = 0
    for id_x in tqdm(range(data.shape[1])):
        for id_y in range(data.shape[2]):
            if numpy.isnan(data[:, id_x, id_y]).any():
                continue
            number_of_data_points += 1
            current_time_series = data[:, id_x, id_y]
            min_error = numpy.inf
            closest_cluster = None
            all_distances = []
            for cluster_id, (subspace, mean) in subspaces.items():
                current_time_series = current_time_series - mean
                # project current time series onto subspace
                projection = subspace.T @ (subspace @ current_time_series)
                # use squared Euclidean distance
                residual = current_time_series - projection
                distance = numpy.sum(residual ** 2)
                all_distances.append(distance)
                # otherwise could use the norm
                # distance = numpy.linalg.norm(current_time_series - x_proj)
                if distance < min_error:
                    min_error = distance
                    closest_cluster = cluster_id
                average_distances_to_each_subspace[cluster_id] += distance
            sorted_distances = sorted(all_distances)
            for ind in range(len(sorted_distances)):
                all_average_distances[ind] += sorted_distances[ind]

            # insert node with its closest cluster as attribute
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

    # plot distances to subspaces

    x_points = []
    y_points = []
    for ind in range(len(all_average_distances)):
        # print(f"average distance {ind}: {all_average_distances[ind] / number_of_data_points}")
        x_points.append(ind)
        y_points.append(all_average_distances[ind] / number_of_data_points)

    plt.plot(x_points, y_points, color="blue", marker='o')
    plt.savefig(f"{OUT_DIR}/average_distances_to_subspace_sorted_by_closest.png")
    plt.close()
    x_points = []
    y_points = []
    for cluster_id in average_distances_to_each_subspace.keys():
        # print(
        #     f"average distance to subspace {cluster_id}: {average_distances_to_each_subspace[cluster_id] / number_of_data_points}")
        x_points.append(cluster_id)
        y_points.append(average_distances_to_each_subspace[cluster_id] / number_of_data_points)
    plt.plot(x_points, y_points, color="green", marker='o')
    plt.savefig(f"{OUT_DIR}/average_distances_to_subspace_ordered_by_cluster.png")
    plt.close()
    # for each point in the cluster determine the distance to the closest subspace
    # project each time series onto each subspace
    # compute the reconstruction error
    # return a graph where each point is marked with its closest subspace
    return assignment_graph, grid_point_assignment


def start_subspace_clustering(sea_level_anomaly_data: xarray.Dataset, clustering: xarray.Dataset, out_dir: str,
                              components: []):
    """
    Start the subspace clustering
    :param components:
    :param out_dir:
    :param sea_level_anomaly_data:
    :param clustering:
    :return:
    """
    global OUT_DIR
    global EXPLAINED_VARIANCE
    EXPLAINED_VARIANCE = {}
    # adjust resolution, such that it is the same for the sea level anomaly data as it is for the clustering
    resolution = clustering.latitude.values[1] - clustering.latitude.values[0]
    min_lat = clustering.latitude.values[0]
    min_lon = clustering.longitude.values[0]
    # interpolate the sea level anomaly data to the resolution of the clustering
    sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=clustering.latitude.values,
                                                           longitude=clustering.longitude.values)
    # plot first time step
    plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, "sla", name="input_data")
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
    for number_of_components in components:
        current_out_dir = f"{out_dir}/components_{number_of_components}/"
        OUT_DIR = current_out_dir
        EXPLAINED_VARIANCE[number_of_components] = []

        # how many nan values are there?
        nan_count = numpy.isnan(cluster_data).sum()

        # 2D array of size lat x lon that contains only the lat or lon values at each point
        extended_lats = numpy.tile(clustering["latitude"].values[:, numpy.newaxis],
                                   (1, clustering["longitude"].values.shape[0]))
        extended_lons = numpy.tile(clustering["longitude"].values[numpy.newaxis, :],
                                   (clustering["latitude"].values.shape[0], 1))
        cluster_dict = {}
        cluster_id_dict = {}
        for cluster_id in unique_numbers:
            if cluster_id is numpy.nan:
                continue
            # find lat/lon pairs for each cluster
            current_cluster_mask = cluster_data == cluster_id
            filtered_lats = extended_lats[current_cluster_mask]
            filtered_lons = extended_lons[current_cluster_mask]
            lat_lon_pairs = list(zip(filtered_lats, filtered_lons))
            cluster_dict[cluster_id] = lat_lon_pairs
            cluster_id_dict[cluster_id] = []
            for lat_lon_pair in lat_lon_pairs:
                id_x, id_y = helper.lat_lon_to_index(lat_lon_pair[0], lat_lon_pair[1], min_lat, min_lon, resolution)
                cluster_id_dict[cluster_id].append((id_x, id_y))

        plot_clustering(cluster_dict, current_out_dir, resolution, name="initial_clustering")
        logger.info("determine subspaces")
        subspaces = {}  # contains the subspaces for each cluster
        for cluster in cluster_id_dict.keys():
            if not len(cluster_id_dict[cluster]) <= number_of_components * 2:
                current_subspace, mean = determine_subspace(cluster_id_dict[cluster], sla_data, number_of_components)
                if current_subspace is not None:
                    subspaces[cluster] = (current_subspace, mean)
        # determine similarity of subspaces
        similarities = {}  # key: (subspace1, subspace2), value: similarity
        for cluster_1 in subspaces.keys():
            for cluster_2 in subspaces.keys():
                if cluster_1 == cluster_2:
                    continue
                if similarities.get((cluster_1, cluster_2)) is None and similarities.get(
                        (cluster_2, cluster_1)) is None:
                    subspace_1 = subspaces[cluster_1][0]
                    subspace_2 = subspaces[cluster_2][0]
                    # orthonormalize the rows using QR decomposition
                    Q_A = np.linalg.qr(subspace_1.T)
                    Q_B = np.linalg.qr(subspace_2.T)
                    # compute singular values of Q_A^T * Q_B
                    M = Q_A[0].T @ Q_B[0]
                    U, Sigma, Vt = np.linalg.svd(M)
                    # compute principal angles
                    principal_angles = np.arccos(np.clip(Sigma, -1, 1))
                    principal_angles = np.degrees(principal_angles)
                    similarities[(cluster_1, cluster_2)] = principal_angles
                    # calculate cosine similarity of principle angles
                    cosine_similarity = np.cos(principal_angles)
                    # print("Average cosine similarity: ", np.mean(cosine_similarity))
                    # print(f" principal angle: {sorted(principal_angles)}")
                    # TODO: plot this
                    # determine projection frobenius norm
                    U = orth(subspace_1)
                    V = orth(subspace_2)
                    # compute projection matrix
                    P_U = U @ U.T
                    P_V = V @ V.T
                    # compute frobenius norm of difference
                    frob_norm = np.linalg.norm(P_U - P_V, 'fro')
                    # normalize frobenius norm
                    n = U.shape[0]
                    similarity = 1 - (frob_norm / np.sqrt(n))
                    # print(f"frobenius norm: {similarity}")
        assignment_graph, grid_point_assignment = determine_closest_subspace(sla_data, subspaces)
        grid_point_assignment_lat_lon = {}
        for cluster_id in grid_point_assignment.keys():
            grid_point_assignment_lat_lon[cluster_id] = []
            for grid_point in grid_point_assignment[cluster_id]:
                lat, lon = helper.index_to_lat_lon(grid_point[0], grid_point[1], min_lat, min_lon, resolution)
                grid_point_assignment_lat_lon[cluster_id].append((lat, lon))

        plot_clustering(grid_point_assignment_lat_lon, current_out_dir, resolution,
                        name=f"grid_point_assignment{number_of_components}")
    # plot all number of components and variance together
    colors = plotting.random_color_generator(len(EXPLAINED_VARIANCE.keys()) + 1)
    current_color = 0
    for number_of_components in EXPLAINED_VARIANCE.keys():
        explained_variance = EXPLAINED_VARIANCE[number_of_components]
        plt.plot(explained_variance, color=colors[current_color], marker='o',
                 label=f"components: {number_of_components}")
        current_color += 1
    plt.ylabel("explained variance")
    plt.xlabel("cluster_id")
    plt.legend()
    plt.savefig(f"{out_dir}/explained_variance.png")
    plt.close()
