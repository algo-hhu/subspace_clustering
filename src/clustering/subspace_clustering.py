import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import xarray
import xarray as xr
from loguru import logger
from sklearn.decomposition import PCA

from src import helper, plotting
from src.clustering.connect_clusters import reestablish_connectivity
from src.connectivity.gauss_filter_grid_point_assignment import SphericalGaussFilterClustering
from src.distance import subspace_timeseries_distance_calculation
from src.helper import extract_clusters_from_xarray_dataset, save_clustering
from src.plotting import plot_clustering, assign_color_to_cluster

OUT_DIR = None
# Seed for PCA's randomized SVD solver; set from GlobalSettings.random_seed in main() for reproducibility.
PCA_RANDOM_STATE = 42


def plot_explained_variance_per_iteration(explained_variance_per_iteration: dict[int: dict[int: float]],
                                          current_out_dir: str):
    """
    Plot the explained variance per iteration
    :param explained_variance_per_iteration:
    :param current_out_dir:
    :return:
    """
    # plot one line for each cluster, where the x-axis is the iteration number and the y-axis is the explained variance
    explained_variance_per_cluster = {}
    for iteration in explained_variance_per_iteration.keys():
        for cluster_id in explained_variance_per_iteration[iteration].keys():
            if explained_variance_per_cluster.get(cluster_id) is None:
                explained_variance_per_cluster[cluster_id] = ([], [])
            explained_variance_per_cluster[cluster_id][0].append(iteration)
            explained_variance_per_cluster[cluster_id][1].append(
                explained_variance_per_iteration[iteration][cluster_id])
    plt.figure()
    for cluster_id in explained_variance_per_cluster.keys():
        plt.plot(explained_variance_per_cluster[cluster_id][0],
                 explained_variance_per_cluster[cluster_id][1], label=f"Cluster {cluster_id}")
    plt.xlabel("Iteration")
    plt.ylabel("Explained Variance")
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    plt.title("Explained Variance per Iteration")
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.savefig(f"{current_out_dir}/explained_variance_per_iteration.png", bbox_inches='tight')
    plt.close()
    return


def start_subspace_clustering(sea_level_anomaly_data: xarray.Dataset, clustering_dataset: xarray.Dataset,
                              original_out_dir: str,
                              components: list[int], number_of_clusters: int, collect_output_file_path: str):
    """
    Start the subspace clustering
    :param collect_output_file_path:
    :param clustering_dataset:
    :param number_of_clusters:
    :param original_out_dir:
    :param components:
    :param sea_level_anomaly_data:
    :return:
    """
    # # use profiler to find bottlenecks
    # profiler = cProfile.Profile()
    # profiler.enable()

    global OUT_DIR
    print(f"number of components: {components}")
    min_lat = clustering_dataset.latitude.values.min()
    min_lon = clustering_dataset.longitude.values.min()
    resolution = float(clustering_dataset.latitude.values[1]) - float(clustering_dataset.latitude.values[0])
    # plot first time step
    # plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, original_out_dir, "sla", name="input_data")
    sla_data = sea_level_anomaly_data["sla"].values
    cluster_data = clustering_dataset.__xarray_dataarray_variable__.values
    # perform the subspace clustering for each set number of components three times, once with filtering, once with
    # connectivity after the clustering and once with connectivity every iteration
    establish_connectivity_afterwards, filter_grid_point_assignment, make_connected_every_round = False, False, False
    for i in range(3):
        # set the output directory according to the settings
        if i == 0:
            (establish_connectivity_afterwards, filter_grid_point_assignment, make_connected_every_round,
             out_dir) = settings_for_connectivity_every_iteration(
                original_out_dir)
            with open(collect_output_file_path, "a") as f:
                f.write(
                    f"merging every iteration: \n"
                )

        elif i == 1:
            (establish_connectivity_afterwards, filter_grid_point_assignment, make_connected_every_round,
             out_dir) = settings_for_connectivity_once(
                original_out_dir)
            with open(collect_output_file_path, "a") as f:
                f.write(f"merging once after clustering: \n")
        elif i == 2:
            (establish_connectivity_afterwards, filter_grid_point_assignment, make_connected_every_round,
             out_dir) = settings_for_filtering_for_connectivity(
                original_out_dir)
            with open(collect_output_file_path, "a") as f:
                f.write(f"filtering every iteration, merging once after clustering: \n")

        for number_of_components in components:
            # check if this has already been done

            if os.path.exists(f"{out_dir}/components_{number_of_components}/"):
                logger.info(f"clustering for {number_of_components} components already exists at {out_dir}")
                continue
            logger.info(f"assigning subspaces for {number_of_components} components")

            # get start clustering dictionary from initial clustering netcdf data and plot
            cluster_dict_lat_lon, cluster_to_grid_point_ids_dict = extract_clusters_from_xarray_dataset(
                clustering_dataset, min_lat, min_lon, resolution, sla_data)
            cluster_id_to_color = assign_color_to_cluster(cluster_to_grid_point_ids_dict, number_of_clusters)

            # create output directory for current number of components
            current_out_dir = f"{out_dir}/components_{number_of_components}/"
            OUT_DIR = current_out_dir

            # evaluate start-clustering
            start_sum_of_distances, start_explained_variance_per_cluster = evaluate_distances_to_subspaces(
                cluster_to_grid_point_ids_dict, sla_data, number_of_components,
                current_out_dir)
            if i == 0:
                with open(collect_output_file_path, "a") as f:
                    f.write(f"initial; {number_of_components}; {round(start_sum_of_distances, 5)} \\\\ \n")
            name = f"initial_clustering_{number_of_components}"
            with open(f"{current_out_dir}/{name}.txt", "w") as f:
                f.write(f"Start sum of distances to subspaces: {start_sum_of_distances}\n")

            # plot_clustering(cluster_dict_lat_lon, current_out_dir, resolution, name, cluster_id_to_color)
            sum_distances_to_subspaces = {}

            # iteratively determine the subspaces for each cluster and assign grid points to the closest subspace
            change = True
            counter = 0
            # save the sum of the distances from grid points to the subspaces in each iteration
            sum_distances_to_subspaces[counter] = start_sum_of_distances
            counter += 1
            # grid_point_assignment = cluster_to_grid_point_ids_dict.copy()
            best_solution = None
            best_grid_point_assignment = None  # grid-point form of best_solution, for subspace similarity
            best_distances_to_subspaces = start_sum_of_distances
            best_iteration = 0
            explained_variance_per_iteration = {}
            while change:
                print(".", end="")
                # for each cluster, determine its subspace
                subspaces, explained_variance_per_cluster = calculate_subspaces_for_clusters(
                    cluster_to_grid_point_ids_dict, number_of_components,
                    sla_data)
                explained_variance_per_iteration[counter] = explained_variance_per_cluster
                # subspace similarity (principal angles) is computed once on the final
                # clustering below, after the loop, rather than every iteration
                # assign each grid point to its closest subspace
                cluster_to_grid_point_ids_dict, change, summed_distances = determine_closest_subspace(sla_data,
                                                                                                      subspaces,
                                                                                                      number_of_components,
                                                                                                      cluster_to_grid_point_ids_dict)
                # map the grid point assignment to the lat/lon coordinates and plot
                grid_point_assignment_lat_lon = convert_idx_idy_to_lat_lon(cluster_to_grid_point_ids_dict, min_lat,
                                                                           min_lon,
                                                                           resolution)
                name = f"{counter}"
                # plot_clustering(grid_point_assignment_lat_lon, current_out_dir, resolution,
                #                 name, cluster_id_to_color)

                cluster_map = create_cluster_map(cluster_data, cluster_to_grid_point_ids_dict)

                if filter_grid_point_assignment:
                    half_width = 200  # in km
                    current_filter = SphericalGaussFilterClustering(clustering_dataset.latitude.values,
                                                                    clustering_dataset.longitude.values, half_width)
                    cluster_to_grid_point_ids_dict = current_filter.parallelized_filter(cluster_map)
                    cluster_to_lat_lon = convert_idx_idy_to_lat_lon(cluster_to_grid_point_ids_dict, min_lat, min_lon,
                                                                    resolution)
                    name = f"filtered_{counter}"
                    # plot_clustering(cluster_to_lat_lon, current_out_dir, resolution, name, cluster_id_to_color)
                elif make_connected_every_round:
                    cluster_to_grid_point_ids_dict = reestablish_connectivity(sea_level_anomaly_data,
                                                                              grid_point_assignment_lat_lon,
                                                                              subspaces,
                                                                              counter, OUT_DIR, cluster_id_to_color,
                                                                              number_of_clusters)

                    # plot filtered data
                    cluster_to_lat_lon = convert_idx_idy_to_lat_lon(cluster_to_grid_point_ids_dict, min_lat, min_lon,
                                                                    resolution)
                    name = f"reconnected_{counter}"
                    # plot_clustering(cluster_to_lat_lon, current_out_dir, resolution, name, cluster_id_to_color)
                else:
                    cluster_to_lat_lon = grid_point_assignment_lat_lon
                # evaluate the distances to the subspaces after the current clustering
                sum_of_distances_after_conn, explained_variance_after_conn = evaluate_distances_to_subspaces(
                    cluster_to_grid_point_ids_dict, sla_data, number_of_components, current_out_dir)
                # save the sum of the distances to the subspaces in each iteration
                sum_distances_to_subspaces[counter] = sum_of_distances_after_conn
                if sum_of_distances_after_conn < best_distances_to_subspaces:
                    best_distances_to_subspaces = sum_of_distances_after_conn
                    best_solution = cluster_to_lat_lon
                    best_grid_point_assignment = cluster_to_grid_point_ids_dict.copy()
                    best_iteration = counter
                counter += 1
                if counter >= 50:
                    break
            counter += 1

            if best_solution is None:
                logger.error(f"No best solution found for {number_of_components} components. Skipping.")
                with open(collect_output_file_path, "a") as f:
                    # A & B \\ in latex table inside table
                    f.write(f" ;- ; - \\\\ \n")
                continue
            # if connectivity has not been established in every iteration, do it now.
            # reconnect the best clustering (the one that gets saved), using its own subspaces,
            # and treat the reconnected result as the new best clustering.
            if establish_connectivity_afterwards == True or filter_grid_point_assignment == True:
                best_subspaces, _ = calculate_subspaces_for_clusters(
                    best_grid_point_assignment, number_of_components, sla_data)
                best_grid_point_assignment = reestablish_connectivity(sea_level_anomaly_data,
                                                                      best_solution,
                                                                      best_subspaces,
                                                                      counter, OUT_DIR, cluster_id_to_color,
                                                                      number_of_clusters)
                best_solution = convert_idx_idy_to_lat_lon(best_grid_point_assignment, min_lat, min_lon,
                                                           resolution)
                name = f"final_clustering_{number_of_components}"
                plot_clustering(best_solution, current_out_dir, resolution, name, cluster_id_to_color)
                # calculate resulting value
                summed_distances, explained_variance = evaluate_distances_to_subspaces(best_grid_point_assignment,
                                                                                       sla_data,
                                                                                       number_of_components,
                                                                                       current_out_dir)
                best_distances_to_subspaces = summed_distances
                sum_distances_to_subspaces[counter] = summed_distances
            # calculate how similar/different the subspaces of the returned (best) clustering are
            final_subspaces, _ = calculate_subspaces_for_clusters(
                best_grid_point_assignment, number_of_components, sla_data)
            principal_angles = calculate_principal_angles(final_subspaces)
            ordered_ids, largest_angle, chordal_distance, mean_angle = summarize_principal_angles(
                principal_angles, final_subspaces.keys())
            similarity_file = os.path.join(current_out_dir, f"subspace_similarity_{number_of_components}.txt")
            write_subspace_similarity(similarity_file, ordered_ids, largest_angle, chordal_distance, mean_angle)
            logger.info(f"wrote subspace similarity matrices to {similarity_file}")
            # plot the summed distances to the subspaces
            plotting.plot_summed_distances_to_subspaces(sum_distances_to_subspaces, current_out_dir,
                                                        number_of_components, best_iteration,
                                                        best_distances_to_subspaces)
            # save summed distances to subspaces to a file
            with open(f"{current_out_dir}/sum_distances_to_subspaces.pkl", "wb") as outfile:
                pickle.dump(sum_distances_to_subspaces, outfile)
            # plot the explained variance
            # save best clustering and start / end distances
            final_results_name = f"final_results_{number_of_components}.txt"
            with open(os.path.join(current_out_dir, final_results_name), "w") as f:
                f.write(
                    f"Best solution found after {best_iteration} iterations with {number_of_components} components.\n")
                f.write(f"Total number of iterations: {counter}\n")
                f.write(f"Best sum of distances to subspaces: {best_distances_to_subspaces}\n")
                f.write(f"Sum of distances to subspace before clustering {start_sum_of_distances}\n")
            name = f"clustering_{number_of_clusters}"
            save_clustering(best_solution, current_out_dir, sea_level_anomaly_data, name)
            # plot the explained variance
            plot_explained_variance_per_iteration(explained_variance_per_iteration, current_out_dir)
            # plotting.plot_average_explained_variance(explained_variance_per_iteration, current_out_dir)
            # write the explained variance to a json file
            with open(f"{current_out_dir}/explained_variance_per_iteration.json", "w") as outfile:
                json.dump(explained_variance_per_iteration, outfile)
            # save the final distances to subspaces in the file:
            with open(collect_output_file_path, "a") as f:
                # A & B \\ in latex table inside table
                f.write(f"; {number_of_components:} ; {round(best_distances_to_subspaces, 5)} \\\\ \n")
    return


def settings_for_filtering_for_connectivity(original_out_dir):
    """
    Settings for filtering every round and establishing connectivity once after clustering
    :param original_out_dir:
    :return:
    """
    filter_grid_point_assignment = True
    make_connected_every_round = False
    establish_connectivity_afterwards = False
    out_dir = f"{original_out_dir}/filter_every_round_connectivity_once/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info("Filtering every round and establishing connectivity once after clustering")
    return establish_connectivity_afterwards, filter_grid_point_assignment, make_connected_every_round, out_dir


def settings_for_connectivity_once(original_out_dir):
    """
    Settings for establishing connectivity once after clustering
    :param original_out_dir:
    :return:
    """
    filter_grid_point_assignment = False
    make_connected_every_round = False
    establish_connectivity_afterwards = True
    out_dir = f"{original_out_dir}/establish_connectivity_once/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info("Establishing connectivity once after clustering")
    return establish_connectivity_afterwards, filter_grid_point_assignment, make_connected_every_round, out_dir


def settings_for_connectivity_every_iteration(original_out_dir):
    """
    Settings for establishing connectivity every iteration
    :param original_out_dir:
    :return:
    """
    filter_grid_point_assignment = False
    make_connected_every_round = True
    establish_connectivity_afterwards = False
    out_dir = f"{original_out_dir}/establish_connectivity_every_iteration/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logger.info("Establishing connectivity every iteration")
    return establish_connectivity_afterwards, filter_grid_point_assignment, make_connected_every_round, out_dir


def create_cluster_map(cluster_data, cluster_to_grid_point_ids_dict):
    """
    Create a map containing the cluster id for each grid point
    :param cluster_data:
    :param cluster_to_grid_point_ids_dict:
    :return:
    """
    # create map containing the cluster id for each grid point
    cluster_map = np.full(cluster_data.shape, np.nan)
    for cluster_id in cluster_to_grid_point_ids_dict.keys():
        for (id_x, id_y) in cluster_to_grid_point_ids_dict[cluster_id]:
            cluster_map[id_x, id_y] = cluster_id
    return cluster_map


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


def evaluate_distances_to_subspaces(cluster_to_grid_point_ids_dict: dict[int, list[tuple[float, float]]],
                                    sla_data: np.ndarray, number_of_components: int, out_dir: str):
    """

    :param out_dir:
    :param number_of_components:
    :param sla_data:
    :param cluster_to_grid_point_ids_dict: {int: [(int, int)]}
    :return:
    """
    explained_variance_per_cluster = {}
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sum_of_distances = 0
    number_of_grid_points = 0
    for cluster_id in cluster_to_grid_point_ids_dict.keys():
        if len(cluster_to_grid_point_ids_dict[cluster_id]) <= number_of_components:
            logger.warning(
                f"Cluster {cluster_id} has less than {number_of_components} grid points. No subspace can be created.")
            continue
        subspace, mean, explained_variance = determine_subspace_per_cluster(cluster_to_grid_point_ids_dict[cluster_id],
                                                                            sla_data,
                                                                            number_of_components)
        # save explained variance for current subspace
        explained_variance_per_cluster[cluster_id] = explained_variance
        if subspace is None:
            logger.warning(
                f"Cluster {cluster_id} has less than {number_of_components} grid points. No subspace can be created.")
            continue
        # calculate the distance to the subspace for each grid point in the cluster
        for (id_x, id_y) in cluster_to_grid_point_ids_dict[cluster_id]:
            current_time_series = sla_data[:, id_x, id_y]
            distance = subspace_timeseries_distance_calculation(current_time_series, mean, subspace)
            sum_of_distances += distance
            number_of_grid_points += 1

    return sum_of_distances, explained_variance_per_cluster


def determine_subspace_per_cluster(cluster_grid_point_ids: [(int, int)], data: np.ndarray,
                                   number_of_components: int) -> tuple[np.array, np.array, float]:
    """
    Generate the subspace of a cluster
    :param cluster_grid_point_ids:
    :param number_of_components:
    :param data:
    :return:
    """
    # check if the number of grid points is less than the number of components (otherwise the subspace with
    # number_of_components dimensions can not be created)
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
    pca = PCA(number_of_components, random_state=PCA_RANDOM_STATE)
    try:
        pca.fit(data_matrix)
    except ValueError as e:
        raise RuntimeError(
            f"PCA failed for cluster with {len(cluster_grid_point_ids)} grid points and "
            f"{number_of_components} components") from e
    explained_variance = pca.explained_variance_ratio_
    # print(explained_variance)
    # save sum of explained variance for current subspace
    components = pca.components_
    mean = pca.mean_
    return components, mean, np.sum(explained_variance)


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
    summed_distances = 0
    grid_point_assignment = {cluster_id: [] for cluster_id in subspaces}
    # create a dictionary for each cluster id, to store the average distance to the subspace for plotting
    all_average_distances = [0] * len(subspaces)
    average_distances_to_each_subspace = {cluster_id: 0 for cluster_id in subspaces}
    number_of_data_points = 0
    # iterate over grid points to find its closest subspace
    for id_x in (range(data.shape[1])):
        for id_y in range(data.shape[2]):
            if np.isnan(data[:, id_x, id_y]).any():
                continue
            number_of_data_points += 1
            current_time_series = data[:, id_x, id_y]
            all_distances, closest_cluster, best_distance = compare_distances_to_subspaces(
                average_distances_to_each_subspace, current_time_series, subspaces)
            # add distance to average distance for plotting
            sorted_distances = sorted(all_distances)
            for ind in range(len(sorted_distances)):
                all_average_distances[ind] += sorted_distances[ind]
            grid_point_assignment[closest_cluster].append((id_x, id_y))
            summed_distances += best_distance
            # check if the cluster id has changed
            if not (id_x, id_y) in previous_grid_point_assignment[closest_cluster]:
                change = True
    return grid_point_assignment, change, summed_distances


def compare_distances_to_subspaces(average_distances_to_each_subspace: dict[int, float], current_time_series: np.array, subspaces: dict[int: (np.array, np.array)]):
    """
    Compare the distances to each subspace
    :param average_distances_to_each_subspace:
    :param current_time_series:
    :param subspaces:
    :return:
    """
    min_error = np.inf
    closest_cluster = None
    best_distance = np.inf
    all_distances = []
    # iterate over all subspaces and calculate the distance to the current time series
    for cluster_id, (subspace, mean) in subspaces.items():
        distance = subspace_timeseries_distance_calculation(current_time_series, mean, subspace)
        all_distances.append(distance)
        if distance < min_error:
            min_error = distance
            closest_cluster = cluster_id
            best_distance = distance
        average_distances_to_each_subspace[cluster_id] += distance
    return all_distances, closest_cluster, best_distance


def calculate_subspaces_for_clusters(cluster_id_dict, number_of_components, sla_data) -> tuple[dict, dict]:
    """
    Calculate the subspaces for each cluster
    :param cluster_id_dict:
    :param number_of_components:
    :param sla_data:
    :return:
    """
    subspaces = {}  # contains the subspaces and mean for each cluster
    explained_variance_per_cluster = {}  # contains the explained variance for each cluster
    # iterate over all clusters and determine the subspace
    for cluster in cluster_id_dict.keys():
        if not len(cluster_id_dict[cluster]) <= number_of_components:
            # print(f"current cluster for subspace clustering: {cluster}")
            current_subspace, mean, explained_variance = determine_subspace_per_cluster(cluster_id_dict[cluster],
                                                                                        sla_data,
                                                                                        number_of_components)
            explained_variance_per_cluster[cluster] = explained_variance
            if current_subspace is not None:
                subspaces[cluster] = (current_subspace, mean)
    return subspaces, explained_variance_per_cluster


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


def calculate_principal_angles(subspaces):
    """
    Calculate the principal angles between the subspaces
    :param subspaces:
    :return:
    """
    # determine similarity of subspaces
    similarities = {}  # key: (cluster_1, cluster_2), value: array of principal angles (degrees)
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
    return similarities


def summarize_principal_angles(similarities, cluster_ids):
    """
    Reduce the pairwise principal angles to cluster-by-cluster similarity matrices.

    For each cluster pair the array of principal angles (in degrees, as returned by
    calculate_principal_angles) is collapsed into three scalar measures:
      - largest principal angle (degrees): most conservative; small means one subspace
        is nearly contained in the other (subspaces are similar).
      - chordal distance: sqrt(sum_i sin^2(theta_i)), a metric on the Grassmannian;
        0 = identical subspaces, larger = more different.
      - mean principal angle (degrees).

    :param similarities: dict {(cluster_1, cluster_2): array of angles in degrees}
    :param cluster_ids: iterable of the cluster ids present in the clustering
    :return: (ordered_ids, largest_angle, chordal_distance, mean_angle) where the three
             values are symmetric (n x n) numpy matrices aligned with ordered_ids.
             Self-comparisons on the diagonal are 0 (identical subspace).
    """
    ordered_ids = sorted(cluster_ids)
    index = {cluster_id: i for i, cluster_id in enumerate(ordered_ids)}
    n = len(ordered_ids)
    largest_angle = np.full((n, n), np.nan)
    chordal_distance = np.full((n, n), np.nan)
    mean_angle = np.full((n, n), np.nan)
    # a subspace is identical to itself
    np.fill_diagonal(largest_angle, 0.0)
    np.fill_diagonal(chordal_distance, 0.0)
    np.fill_diagonal(mean_angle, 0.0)
    for (cluster_1, cluster_2), angles_deg in similarities.items():
        i, j = index[cluster_1], index[cluster_2]
        angles_rad = np.radians(angles_deg)
        largest = float(np.max(angles_deg))
        chordal = float(np.sqrt(np.sum(np.sin(angles_rad) ** 2)))
        mean = float(np.mean(angles_deg))
        # principal angles are symmetric between two subspaces
        largest_angle[i, j] = largest_angle[j, i] = largest
        chordal_distance[i, j] = chordal_distance[j, i] = chordal
        mean_angle[i, j] = mean_angle[j, i] = mean
    return ordered_ids, largest_angle, chordal_distance, mean_angle


def write_subspace_similarity(file_path, ordered_ids, largest_angle, chordal_distance, mean_angle):
    """
    Write the three cluster-by-cluster subspace-similarity matrices to a text file.
    """

    def format_matrix(matrix):
        header = "          " + "".join(f"{cluster_id:>12}" for cluster_id in ordered_ids)
        rows = [header]
        for i, cluster_id in enumerate(ordered_ids):
            cells = "".join(
                "         nan" if np.isnan(value) else f"{value:>12.4f}"
                for value in matrix[i]
            )
            rows.append(f"{cluster_id:>10}" + cells)
        return "\n".join(rows)

    with open(file_path, "w") as f:
        f.write("Cluster-by-cluster subspace similarity of the final clustering\n")
        f.write(f"Clusters: {[float(cluster_id) for cluster_id in ordered_ids]}\n\n")
        f.write("Largest principal angle (degrees; small = similar subspaces)\n")
        f.write(format_matrix(largest_angle) + "\n\n")
        f.write("Chordal distance (0 = identical subspaces, larger = more different)\n")
        f.write(format_matrix(chordal_distance) + "\n\n")
        f.write("Mean principal angle (degrees)\n")
        f.write(format_matrix(mean_angle) + "\n")


def modify_clustering_with_subspaces(cluster_to_grid_point_ids_dict: {int: (int, int)}, sla_data: np.array,
                                     subspaces: {int: (np.array, np.array)}, cluster_data: np.array):
    """
Modify the clustering with subspaces by checking if the grid points are closer to a different subspace
    :param cluster_data:
    :param cluster_to_grid_point_ids_dict:
    :param sla_data:
    :param subspaces:
    :return:
    """
    change = False
    summed_distances = 0
    new_cluster_to_grid_point_ids_dict = {cluster_id: [] for cluster_id in cluster_to_grid_point_ids_dict.keys()}
    cluster_map = create_cluster_map(cluster_data, cluster_to_grid_point_ids_dict)

    # iterate over all grid points and check if they have a neighbor of a different color, if yes, check if subspace
    # is closer
    for id_x in range(sla_data.shape[1]):
        for id_y in range(sla_data.shape[2]):
            if np.isnan(sla_data[:, id_x, id_y]).any():
                continue
            current_time_series = sla_data[:, id_x, id_y]
            current_cluster = cluster_map[id_x, id_y]
            if current_cluster not in subspaces.keys():
                continue
            possible_clusters = []
            # find all clusters that are neighbors of the current cluster
            # check the 4 neighbors (up, down, left, right)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_x = id_x + dx
                neighbor_y = id_y + dy
                if 0 <= neighbor_x < sla_data.shape[1] and 0 <= neighbor_y < sla_data.shape[2]:
                    neighbor_cluster = cluster_map[neighbor_x, neighbor_y]
                    if np.isnan(neighbor_cluster):
                        continue
                    if neighbor_cluster != current_cluster and neighbor_cluster not in possible_clusters:
                        possible_clusters.append(neighbor_cluster)
            # if there are no neighbors, continue
            if len(possible_clusters) == 0:
                new_cluster_to_grid_point_ids_dict[current_cluster].append((id_x, id_y))
                distance = subspace_timeseries_distance_calculation([], current_time_series,
                                                                    subspaces[current_cluster][1],
                                                                    subspaces[current_cluster][0])
                summed_distances += distance
                continue
            # calculate the distance to the subspace for each neighbor cluster
            min_distance = np.inf
            closest_cluster = current_cluster
            closest_distance = np.inf
            for neighbor_cluster in possible_clusters:
                if neighbor_cluster not in subspaces.keys():
                    continue
                subspace, mean = subspaces[int(neighbor_cluster)]
                distance = subspace_timeseries_distance_calculation([], current_time_series, mean, subspace)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = neighbor_cluster
                    closest_distance = distance
            # if the distance to the closest neighbor cluster is smaller than the distance to the current cluster,
            # change the cluster assignment
            if closest_distance < subspace_timeseries_distance_calculation([], current_time_series,
                                                                           subspaces[current_cluster][1],
                                                                           subspaces[current_cluster][0]):
                change = True
                new_cluster_to_grid_point_ids_dict[closest_cluster].append((id_x, id_y))
                summed_distances += closest_distance
            else:
                new_cluster_to_grid_point_ids_dict[current_cluster].append((id_x, id_y))
                distance = subspace_timeseries_distance_calculation([], current_time_series,
                                                                    subspaces[current_cluster][1],
                                                                    subspaces[current_cluster][0])
                summed_distances += distance
    return new_cluster_to_grid_point_ids_dict, change, summed_distances


def start_subspace_clustering_with_integrated_connectivity(sea_level_anomaly_data: xarray.Dataset,
                                                           initial_clustering: xarray.Dataset, out_dir: str,
                                                           number_of_components: list, resolution: float,
                                                           number_of_clusters: int, collect_output_file_path: str):
    """
    Start the subspace clustering with integrated connectivity
    :param resolution:
    :param number_of_clusters:
    :param out_dir:
    :param sea_level_anomaly_data:
    :param initial_clustering:
    :param number_of_components:
    :return:
    """

    global OUT_DIR
    OUT_DIR = out_dir
    logger.info("Starting subspace clustering with integrated connectivity")
    # adjust resolution, such that it is the same for the sea level anomaly data as it is for the clustering
    min_lat = sea_level_anomaly_data.latitude.values[0]
    min_lon = sea_level_anomaly_data.longitude.values[0]
    # plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, "sla", name="input_data")
    sla_data = sea_level_anomaly_data["sla"].values
    cluster_data = initial_clustering["__xarray_dataarray_variable__"].values
    with open(collect_output_file_path, "a") as f:
        f.write(f"subspace_clustering_integrated_connectivity\n")
    for current_number_of_components in number_of_components:
        logger.info(
            f"Starting subspace clustering with integrated connectivity with {current_number_of_components} components")
        current_out_dir = f"{out_dir}/components_{current_number_of_components}/"
        if not os.path.exists(current_out_dir):
            os.makedirs(current_out_dir)
        else:
            logger.info(
                f"Since the directory {current_out_dir} already exists, the subspace clustering with integrated "
                f"connectivity with {current_number_of_components} components is skipped")
            return
        summed_distances_to_subspaces = {}
        iteration_counter = 0
        cluster_dict, cluster_to_grid_point_ids_dict = extract_clusters_from_xarray_dataset(initial_clustering, min_lat,
                                                                                            min_lon,
                                                                                            resolution, sla_data)
        cluster_id_to_color = assign_color_to_cluster(cluster_to_grid_point_ids_dict, number_of_clusters)
        name = f"initial_clustering_{current_number_of_components}"

        start_summed_distances, start_explained_variance = evaluate_distances_to_subspaces(
            cluster_to_grid_point_ids_dict, sla_data, current_number_of_components, out_dir)
        plot_clustering(cluster_dict, current_out_dir, resolution, name, cluster_id_to_color)
        summed_distances_to_subspaces[iteration_counter] = start_summed_distances
        change = True
        logger.info(f"Subspace clustering with {current_number_of_components} components started")
        while change:
            change = False
            print(".", end=" ")
            iteration_counter += 1
            subspaces, explained_variance_per_cluster = calculate_subspaces_for_clusters(cluster_to_grid_point_ids_dict,
                                                                                         current_number_of_components,
                                                                                         sla_data)
            # change the clustering on the basis of the subspaces
            cluster_to_grid_point_ids_dict, change, summed_distances = modify_clustering_with_subspaces(
                cluster_to_grid_point_ids_dict, sla_data, subspaces, cluster_data)
            grid_point_assignment_lat_lon = convert_idx_idy_to_lat_lon(cluster_to_grid_point_ids_dict, min_lat, min_lon,
                                                                       resolution)
            summed_distances_to_subspaces[iteration_counter] = summed_distances
            name = f"{iteration_counter}"
            # plot_clustering(grid_point_assignment_lat_lon, current_out_dir, resolution, name, cluster_id_to_color)
            if iteration_counter >= 100:
                break

        # reestablish connectivity after the clustering is done
        iteration_counter += 1
        cluster_to_grid_point_ids_dict = reestablish_connectivity(sea_level_anomaly_data,
                                                                  grid_point_assignment_lat_lon, subspaces,
                                                                  iteration_counter, current_out_dir,
                                                                  cluster_id_to_color, number_of_clusters)
        grid_point_assignment_lat_lon = convert_idx_idy_to_lat_lon(cluster_to_grid_point_ids_dict, min_lat, min_lon,
                                                                   resolution)
        name = f"final"
        plot_clustering(grid_point_assignment_lat_lon, current_out_dir, resolution, name, cluster_id_to_color)
        summed_distances, explained_variance = evaluate_distances_to_subspaces(cluster_to_grid_point_ids_dict, sla_data,
                                                                               current_number_of_components,
                                                                               current_out_dir)
        with open(f"{current_out_dir}/final.txt", "w") as f:
            f.write(f"Summed distances to subspaces: {summed_distances}\n")
            f.write(f"Start summed distances to subspaces: {start_summed_distances}\n")

        summed_distances_to_subspaces[iteration_counter] = summed_distances
        with open(f"{current_out_dir}/sum_distances_to_subspaces.pkl", "wb") as outfile:
            pickle.dump(summed_distances_to_subspaces, outfile)

        plotting.plot_summed_distances_to_subspaces(summed_distances_to_subspaces, current_out_dir,
                                                    current_number_of_components, None, None)
        # save clustering as netcdf
        name = f"clustering_{number_of_clusters}"
        save_clustering(grid_point_assignment_lat_lon, current_out_dir, sea_level_anomaly_data, name)
        with open(collect_output_file_path, "a") as f:
            f.write(
                f";{current_number_of_components} ; {round(summed_distances, 5)} \\\\ \n"
            )

    return


def calculate_subspace_clustering(global_settings, out_dir: str,
                                  subspace_clustering_settings,
                                  unfiltered_sea_level_anomaly_data: xr.Dataset, initial_clustering: xarray.Dataset,
                                  collect_output_file_path: str):
    """
    Calculate subspace clustering
    :param collect_output_file_path:
    :param initial_clustering:
    :param out_dir:
    :param global_settings:
    :param subspace_clustering_settings:
    :param unfiltered_sea_level_anomaly_data:
    :return:
    """

    # plot underlying data
    # plotting.plot_sla_for_point_in_time(unfiltered_sea_level_anomaly_data, out_dir, "sla", name="input_data")
    if subspace_clustering_settings.do_subspace_clustering:
        current_out_dir = f"{out_dir}/subspace_clustering_{subspace_clustering_settings.number_of_clusters}"
        if not os.path.exists(current_out_dir):
            os.makedirs(current_out_dir)
        print(f"output directory: {current_out_dir}")

        start_subspace_clustering(unfiltered_sea_level_anomaly_data, initial_clustering,
                                  f"{current_out_dir}",
                                  subspace_clustering_settings.number_of_components,
                                  subspace_clustering_settings.number_of_clusters, collect_output_file_path)

    if subspace_clustering_settings.integrated_connectivity:
        if not subspace_clustering_settings.do_subspace_clustering:
            current_out_dir = f"{out_dir}/subspace_clustering_{subspace_clustering_settings.number_of_clusters}"
        current_out_dir = f"{current_out_dir}/integrated_connectivity"
        print(f"output directory: {current_out_dir}")
        if not os.path.exists(current_out_dir):
            os.makedirs(current_out_dir)
        start_subspace_clustering_with_integrated_connectivity(
            unfiltered_sea_level_anomaly_data,
            initial_clustering,
            current_out_dir,
            subspace_clustering_settings.number_of_components,
            global_settings.resolution, subspace_clustering_settings.number_of_clusters, collect_output_file_path)
