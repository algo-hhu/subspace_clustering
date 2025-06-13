import os

import matplotlib.pyplot as plt
import numpy as np
import xarray
from loguru import logger
from sklearn.decomposition import PCA
from tqdm import tqdm

from src import helper, plotting
from src.clustering.connect_clusters import reestablish_connectivity
from src.connectivity.gauss_filter_grid_point_assignment import SphericalGaussFilterClustering
from src.distance import subspace_timeseries_distance_calculation
from src.plotting import plot_clustering, assign_color_to_cluster

OUT_DIR = None
EXPLAINED_VARIANCE = {}
AVG_DIST_TO_SUBSPACE = {}
AVG_DIFF_BETWEEN_SUBSPACES = {}


def start_subspace_clustering(sea_level_anomaly_data: xarray.Dataset, clustering: xarray.Dataset, out_dir: str,
                              components: []):
    """
    Start the subspace clustering
    TODO: plot how the distances to the subspaces change every round
    :param components:
    :param out_dir:
    :param sea_level_anomaly_data:
    :param clustering:
    :return:
    """
    # # use profiler to find bottlenecks
    # profiler = cProfile.Profile()
    # profiler.enable()
    establish_connectivity_afterwards = False
    filter_grid_point_assignment = True
    make_connected_every_round = False

    global OUT_DIR
    global EXPLAINED_VARIANCE
    global AVG_DIFF_BETWEEN_SUBSPACES
    global AVG_DIST_TO_SUBSPACE
    AVG_DIST_TO_SUBSPACE = {}
    AVG_DIFF_BETWEEN_SUBSPACES = {}
    EXPLAINED_VARIANCE = {}
    # adjust resolution, such that it is the same for the sea level anomaly data as it is for the clustering
    min_lat, min_lon, resolution, sea_level_anomaly_data = adjust_resolution(clustering, sea_level_anomaly_data)
    # plot first time step
    plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, "sla", name="input_data")
    sla_data = sea_level_anomaly_data["sla"].values

    # nan mask
    non_nan_mask = ~np.isnan(sla_data).any(axis=0)
    # apply nan mask to clustering data
    cluster_data = clustering["__xarray_dataarray_variable__"].values
    # assign nans where there are all nans in the cluster data
    cluster_data[~non_nan_mask] = np.nan
    unique_numbers, counts = np.unique(cluster_data, return_counts=True)

    unique_numbers = unique_numbers[~np.isnan(unique_numbers)]
    print(unique_numbers)
    # perform the subspace clustering for each wanted number of components
    for number_of_components in components:
        # get start clustering dictionary from initial clustering netcdf data and plot
        cluster_dict, cluster_to_grid_point_ids_dict = extract_original_clusters(cluster_data, clustering, min_lat,
                                                                                 min_lon,
                                                                                 resolution, unique_numbers)
        cluster_id_to_color = assign_color_to_cluster(cluster_to_grid_point_ids_dict)
        logger.info(f"assigning subspaces for {number_of_components} components")
        current_out_dir = f"{out_dir}/components_{number_of_components}/"
        OUT_DIR = current_out_dir
        EXPLAINED_VARIANCE[number_of_components] = []
        # evaluate startclustering
        initial_clustering_outdir = f"{out_dir}/components_{number_of_components}/initial_clustering"
        name = f"initial_clustering_{number_of_components}"
        determine_final_distances_to_subspaces(cluster_to_grid_point_ids_dict, sla_data, number_of_components,
                                               initial_clustering_outdir, name)
        name = f"initial_clustering_{number_of_components}"
        plot_clustering(cluster_dict, current_out_dir, resolution, name, cluster_id_to_color)
        change = True
        counter = 0
        # grid_point_assignment = cluster_to_grid_point_ids_dict.copy()
        while change:
            # for each cluster, determine its subspace
            subspaces = calculate_subspaces_for_clusters(cluster_to_grid_point_ids_dict, number_of_components, sla_data)
            # calculate how similar/different the subspaces are
            similarities = calculate_principal_angles(subspaces)
            AVG_DIFF_BETWEEN_SUBSPACES[number_of_components] = similarities
            # assign each grid point to its closest subspace
            cluster_to_grid_point_ids_dict, change = determine_closest_subspace(sla_data, subspaces,
                                                                                number_of_components,
                                                                                cluster_to_grid_point_ids_dict)
            # map the grid point assignment to the lat/lon coordinates and plot
            grid_point_assignment_lat_lon = convert_idx_idy_to_lat_lon(cluster_to_grid_point_ids_dict, min_lat, min_lon,
                                                                       resolution)
            name = f"{counter}"
            plot_clustering(grid_point_assignment_lat_lon, current_out_dir, resolution,
                            name, cluster_id_to_color)

            # create map containing the cluster id for each grid point
            cluster_map = np.full(cluster_data.shape, np.nan)
            for cluster_id in cluster_to_grid_point_ids_dict.keys():
                for (id_x, id_y) in cluster_to_grid_point_ids_dict[cluster_id]:
                    cluster_map[id_x, id_y] = cluster_id

            if filter_grid_point_assignment:
                half_width = 200  # in km
                current_filter = SphericalGaussFilterClustering(clustering.latitude.values,
                                                                clustering.longitude.values, half_width)
                cluster_to_grid_point_ids_dict = current_filter.parallelized_filter(cluster_map)
                cluster_to_lat_lon = convert_idx_idy_to_lat_lon(cluster_to_grid_point_ids_dict, min_lat, min_lon,
                                                                resolution)
                name = f"filtered_{counter}"
                plot_clustering(cluster_to_lat_lon, current_out_dir, resolution, name, cluster_id_to_color)
            if make_connected_every_round:
                cluster_to_grid_point_ids_dict = reestablish_connectivity(sea_level_anomaly_data,
                                                                          grid_point_assignment_lat_lon,
                                                                          cluster_map, subspaces,
                                                                          counter, OUT_DIR, cluster_id_to_color)

                # plot filtered data
                cluster_to_lat_lon = convert_idx_idy_to_lat_lon(cluster_to_grid_point_ids_dict, min_lat, min_lon,
                                                                resolution)
                name = f"reconnected_{counter}"
                plot_clustering(cluster_to_lat_lon, current_out_dir, resolution, name, cluster_id_to_color)
            # for cluster in cluster_to_grid_point_ids_dict.keys():
            #     print(f"cluster {cluster}: {len(cluster_to_grid_point_ids_dict[cluster])}")

            counter += 1
            if counter >= 50:
                break
        counter += 1
        if establish_connectivity_afterwards == True:
            cluster_to_grid_point_ids_dict = reestablish_connectivity(sea_level_anomaly_data,
                                                                      grid_point_assignment_lat_lon,
                                                                      cluster_map, subspaces,
                                                                      counter, OUT_DIR, cluster_id_to_color)
            cluster_to_lat_lon = convert_idx_idy_to_lat_lon(cluster_to_grid_point_ids_dict, min_lat, min_lon,
                                                            resolution)
        # calculate resulting value
        name = f"final_clustering_{number_of_components}"
        determine_final_distances_to_subspaces(cluster_to_grid_point_ids_dict, sla_data, number_of_components, out_dir,
                                               name)

    # plot the explained variance, difference between subspaces and distance between points and subspaces for each number of components
    # plot_explained_variance_and_distances(out_dir)


def adjust_resolution(clustering: xarray.Dataset, sea_level_anomaly_data: xarray.Dataset):
    """
    Adjust the resolution of the sea level anomaly data to the resolution of the clustering
    :param clustering:
    :param sea_level_anomaly_data:
    :return:
    """
    resolution = clustering.latitude.values[1] - clustering.latitude.values[0]
    min_lat = clustering.latitude.values[0]
    min_lon = clustering.longitude.values[0]
    # interpolate the sea level anomaly data to the resolution of the clustering
    sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=clustering.latitude.values,
                                                           longitude=clustering.longitude.values)
    return min_lat, min_lon, resolution, sea_level_anomaly_data


def determine_final_distances_to_subspaces(cluster_to_grid_point_ids_dict, sla_data, number_of_components, out_dir,
                                           name):
    """

    :param out_dir:
    :param number_of_components:
    :param sla_data:
    :param cluster_to_grid_point_ids_dict: {int: [(int, int)]}
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sum_of_distances = 0
    number_of_grid_points = 0
    for cluster_id in cluster_to_grid_point_ids_dict.keys():
        if len(cluster_to_grid_point_ids_dict[cluster_id]) <= number_of_components:
            logger.warning(
                f"Cluster {cluster_id} has less than {number_of_components} grid points. No subspace can be created.")
            continue
        subspace, mean = determine_subspace_per_cluster(cluster_to_grid_point_ids_dict[cluster_id], sla_data,
                                                        number_of_components)
        if subspace is None:
            logger.warning(
                f"Cluster {cluster_id} has less than {number_of_components} grid points. No subspace can be created.")
            continue
        # calculate the distance to the subspace for each grid point in the cluster
        for (id_x, id_y) in cluster_to_grid_point_ids_dict[cluster_id]:
            current_time_series = sla_data[:, id_x, id_y]
            distance = subspace_timeseries_distance_calculation([], current_time_series, mean, subspace)
            sum_of_distances += distance
            number_of_grid_points += 1
    logger.info(
        f"Average distance to subspace for each cluster components: {sum_of_distances / len(cluster_to_grid_point_ids_dict)}")
    logger.info(f"average distance to subspace per grid point: {sum_of_distances / number_of_grid_points}")
    logger.info(f"total sum of distances to subspace: {sum_of_distances}")
    with open(f"{out_dir}/{name}_average_distance_to_subspace.txt", "w") as f:
        f.write(
            f"Average distance to subspace for each cluster components: {sum_of_distances / len(cluster_to_grid_point_ids_dict)}\n")
        f.write(f"average distance to subspace per grid point: {sum_of_distances / number_of_grid_points}\n")
        f.write(f"total sum of distances to subspace: {sum_of_distances}\n")


def determine_subspace_per_cluster(cluster_grid_point_ids: [(int, int)], data: np.ndarray,
                                   number_of_components: int):
    """
    Generate the subspace of a cluster
    :param cluster_grid_point_ids:
    :param number_of_components:
    :param data:
    :return:
    """
    global EXPLAINED_VARIANCE
    # check if the number of grid points is less than the number of components (otherwise the subspace with number_of_components dimensions can not be created)
    if len(cluster_grid_point_ids) <= number_of_components:
        return None
    # create data matrix X = N x T where T is the number of time steps and N is the number of grid points
    time_length = data.shape[0]
    data_matrix = np.zeros((len(cluster_grid_point_ids), time_length))
    counter = 0
    for grid_point_id_x, grid_point_id_y in cluster_grid_point_ids:
        time_series = data[:, grid_point_id_x, grid_point_id_y]
        data_matrix[counter] = time_series
        counter += 1
    # remove nans
    data_matrix = data_matrix[~np.isnan(data_matrix).any(axis=1)]
    # perform PCA
    pca = PCA(number_of_components)
    try:
        pca.fit(data_matrix)
    except:
        logger.error(f"Could not fit PCA for cluster with {len(cluster_grid_point_ids)} grid points. "
                     f"Number of components: {number_of_components}.")
        print(data_matrix)
        exit(1)
    explained_variance = pca.explained_variance_ratio_
    # save sum of explained variance for current subspace
    EXPLAINED_VARIANCE[number_of_components].append(np.sum(explained_variance))
    components = pca.components_
    mean = pca.mean_
    return components, mean


def determine_closest_subspace(data: np.ndarray, subspaces: {int: np.array}, number_of_components: int,
                               previous_grid_point_assignment: {int: [(int, int)]}):
    """
    For each point in the cluster determine the distance to the closest subspace
    Project each time series onto each subspace
    Compute the reconstruction error
    Return a graph where each point is marked with its closest subspace
    :param previous_grid_point_assignment:
    :param number_of_components:
    :param data:
    :param subspaces:
    :return:
    """
    change = False  # check if the subspaces have changed
    grid_point_assignment = {cluster_id: [] for cluster_id in subspaces}
    # create a dictionary for each cluster id, to store the average distance to the subspace for plotting
    all_average_distances = [0] * len(subspaces)
    average_distances_to_each_subspace = {cluster_id: 0 for cluster_id in subspaces}
    number_of_data_points = 0
    # iterate over grid points to find its closest subspace
    for id_x in tqdm(range(data.shape[1])):
        for id_y in range(data.shape[2]):
            if np.isnan(data[:, id_x, id_y]).any():
                continue
            number_of_data_points += 1
            current_time_series = data[:, id_x, id_y]
            all_distances, closest_cluster = compare_distances_to_subspaces(average_distances_to_each_subspace,
                                                                            current_time_series, subspaces)
            # add distance to average distance for plotting
            sorted_distances = sorted(all_distances)
            for ind in range(len(sorted_distances)):
                all_average_distances[ind] += sorted_distances[ind]
            grid_point_assignment[closest_cluster].append((id_x, id_y))
            # check if the cluster id has changed
            if not (id_x, id_y) in previous_grid_point_assignment[closest_cluster]:
                change = True
    # plotting
    plot_distances_to_subspaces(all_average_distances, average_distances_to_each_subspace, number_of_components,
                                number_of_data_points)
    return grid_point_assignment, change


def plot_distances_to_subspaces(all_average_distances, average_distances_to_each_subspace, number_of_components,
                                number_of_data_points):
    """
    Plot the distances to subspaces once sorted by distance to subspace and once sorted by cluster id
    :param all_average_distances:
    :param average_distances_to_each_subspace:
    :param number_of_components:
    :param number_of_data_points:
    :return:
    """
    # save average distance to chosen subspace in global
    global AVG_DIST_TO_SUBSPACE
    AVG_DIST_TO_SUBSPACE[number_of_components] = all_average_distances[0] / number_of_data_points
    # plot average distance to subspace, sorted by distance to subspace
    x_points = []
    y_points = []
    for ind in range(len(all_average_distances)):
        x_points.append(ind)
        y_points.append(all_average_distances[ind] / number_of_data_points)
    plt.plot(x_points, y_points, color="blue", marker='o')
    plt.savefig(f"{OUT_DIR}/average_distances_to_subspace_sorted_by_closest.png")
    plt.close()
    # plot average distance to subspace, sorted by cluster id
    x_points = []
    y_points = []
    for cluster_id in average_distances_to_each_subspace.keys():
        x_points.append(cluster_id)
        y_points.append(average_distances_to_each_subspace[cluster_id] / number_of_data_points)
    plt.plot(x_points, y_points, color="green", marker='o')
    plt.savefig(f"{OUT_DIR}/average_distances_to_subspace_ordered_by_cluster.png")
    plt.close()
    return


def compare_distances_to_subspaces(average_distances_to_each_subspace: {int: int}, current_time_series: np.array,
                                   subspaces: {int: (np.array, np.array)}):
    """
    Compare the distances to each subspace
    :param average_distances_to_each_subspace:
    :param current_time_series:
    :param subspaces:
    :return:
    """
    min_error = np.inf
    closest_cluster = None
    all_distances = []
    # iterate over all subspaces and calculate the distance to the current time series
    for cluster_id, (subspace, mean) in subspaces.items():
        distance = subspace_timeseries_distance_calculation(all_distances, current_time_series, mean, subspace)
        if distance < min_error:
            min_error = distance
            closest_cluster = cluster_id
        average_distances_to_each_subspace[cluster_id] += distance
    return all_distances, closest_cluster


def calculate_subspaces_for_clusters(cluster_id_dict, number_of_components, sla_data):
    """
    Calculate the subspaces for each cluster
    :param cluster_id_dict:
    :param number_of_components:
    :param sla_data:
    :return:
    """
    subspaces = {}  # contains the subspaces and mean for each cluster
    # iterate over all clusters and determine the subspace
    for cluster in cluster_id_dict.keys():
        if not len(cluster_id_dict[cluster]) <= number_of_components:
            # print(f"current cluster for subspace clustering: {cluster}")
            current_subspace, mean = determine_subspace_per_cluster(cluster_id_dict[cluster], sla_data,
                                                                    number_of_components)
            if current_subspace is not None:
                subspaces[cluster] = (current_subspace, mean)
    return subspaces


def convert_idx_idy_to_lat_lon(grid_point_assignment, min_lat, min_lon, resolution):
    """
    Convert the grid point assignment from index to lat/lon
    :param grid_point_assignment:
    :param min_lat:
    :param min_lon:
    :param resolution:
    :return:
    """
    grid_point_assignment_lat_lon = {}
    for cluster_id in grid_point_assignment.keys():
        grid_point_assignment_lat_lon[cluster_id] = []
        for grid_point in grid_point_assignment[cluster_id]:
            lat, lon = helper.index_to_lat_lon(grid_point[0], grid_point[1], min_lat, min_lon, resolution)
            grid_point_assignment_lat_lon[cluster_id].append((lat, lon))
    return grid_point_assignment_lat_lon


def extract_original_clusters(cluster_data, clustering, min_lat, min_lon, resolution, unique_numbers):
    """
    Extract the original clusters from the clustering data
    :param cluster_data:
    :param clustering:
    :param min_lat:
    :param min_lon:
    :param resolution:
    :param unique_numbers:
    :return:
    """
    # 2D array of size lat x lon that contains only the lat or lon values at each point
    extended_lats = np.tile(clustering["latitude"].values[:, np.newaxis],
                            (1, clustering["longitude"].values.shape[0]))
    extended_lons = np.tile(clustering["longitude"].values[np.newaxis, :],
                            (clustering["latitude"].values.shape[0], 1))
    cluster_dict = {}
    cluster_id_dict = {}
    for cluster_id in unique_numbers:
        print(cluster_id)
        if np.isnan(cluster_id):
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
    return cluster_dict, cluster_id_dict


def calculate_principal_angles(subspaces):
    """
    Calculate the principal angles between the subspaces
    :param subspaces:
    :return:
    """
    # determine similarity of subspaces
    similarities = {}  # key: (subspace1, subspace2), value: similarity
    for cluster_1 in subspaces.keys():
        for cluster_2 in subspaces.keys():
            if cluster_1 == cluster_2:
                continue
            if similarities.get(f"{cluster_1}_{cluster_2}") is None and similarities.get(
                    f"{cluster_2}_{cluster_1}") is None:
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
                similarities[f"{cluster_1}_{cluster_2}"] = principal_angles
    return similarities


def plot_explained_variance_and_distances(out_dir):
    """
    Plot the explained variance for each number of components
    :param out_dir:
    :return:
    """
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
    current_color = 0
    avg_dist_to_chosen_subspace_keys = AVG_DIST_TO_SUBSPACE.keys()
    avg_dist_to_chosen_subspace_values = AVG_DIST_TO_SUBSPACE.values()
    plt.plot(avg_dist_to_chosen_subspace_keys, avg_dist_to_chosen_subspace_values, color="gray",
             linestyle="-")
    plt.scatter(avg_dist_to_chosen_subspace_keys, avg_dist_to_chosen_subspace_values, color=colors, marker='o',
                zorder=3)
    plt.ylabel("average distance to chosen subspaces")
    plt.xlabel("number of components")
    plt.savefig(f"{out_dir}/average_distance_to_chosen_subspaces.png")
    plt.close()
    avg_dist_between_subspaces_keys = []
    avg_dist_between_subspaces_values = []
    for number_of_components in AVG_DIFF_BETWEEN_SUBSPACES.keys():
        current_diff = AVG_DIFF_BETWEEN_SUBSPACES[number_of_components]
        avg_dist_between_subspaces_keys.append(number_of_components)
        avg_dist_between_subspaces_values.append(np.mean(list(current_diff.values())))
    plt.plot(avg_dist_between_subspaces_keys, avg_dist_between_subspaces_values, color="gray", linestyle="-")
    plt.scatter(avg_dist_between_subspaces_keys, avg_dist_between_subspaces_values, color=colors, marker='o',
                zorder=3)
    plt.ylabel(f"avg principle angles")
    plt.xlabel("number of components")
    plt.savefig(f"{out_dir}/principle_angles.png")
    plt.close()
    # min principle angle between subspaces
    min_dist_between_subspaces_keys = []
    min_dist_between_subspaces_values = []
    for number_of_components in AVG_DIFF_BETWEEN_SUBSPACES.keys():
        current_diff = AVG_DIFF_BETWEEN_SUBSPACES[number_of_components]
        min_dist_between_subspaces_keys.append(number_of_components)
        min_dist_between_subspaces_values.append(np.min(list(current_diff.values())))
    plt.plot(min_dist_between_subspaces_keys, min_dist_between_subspaces_values, color="gray", linestyle="-")
    plt.scatter(min_dist_between_subspaces_keys, min_dist_between_subspaces_values, color=colors,
                marker='o',
                zorder=3)
    plt.ylabel(f"min principle angles")
    plt.xlabel("number of components")
    plt.savefig(f"{out_dir}/min_dist_between_subspaces.png")
    plt.close()
    # max principle angle between subspaces
    max_dist_between_subspaces_keys = []
    max_dist_between_subspaces_values = []
    for number_of_components in AVG_DIFF_BETWEEN_SUBSPACES.keys():
        current_diff = AVG_DIFF_BETWEEN_SUBSPACES[number_of_components]
        max_dist_between_subspaces_keys.append(number_of_components)
        max_dist_between_subspaces_values.append(np.max(list(current_diff.values())))
    plt.plot(max_dist_between_subspaces_keys, max_dist_between_subspaces_values, color="gray", linestyle="-")
    plt.scatter(max_dist_between_subspaces_keys, max_dist_between_subspaces_values, color=colors,
                marker='o', zorder=3)
    plt.ylabel(f"max principle angles")
    plt.xlabel("number of components")
    plt.savefig(f"{out_dir}/max_dist_between_subspaces.png")
    plt.close()
