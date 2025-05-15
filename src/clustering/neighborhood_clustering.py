import cProfile
import multiprocessing.shared_memory as shm
import pstats

import numpy
import numpy as np
import xarray
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from src import helper, plotting
from src.clustering.connectivity_helper import find_first_last_longitude, ensure_bidirectional_neighbors

MIN_LATITUDE = None
MIN_LONGITUDE = None
RESOLUTION = None


def find_neighbors(sea_level_anomaly_data: xarray.Dataset, distance_function, lat_lon_to_clusters: {}, nan_mask) -> (
        dict, set):
    """
    Find neighbors for each grid point
    :param lat_lon_to_clusters:
    :param distance_function:
    :param sea_level_anomaly_data:
    :return:
    """
    neighbors = {}  # {cluster: {neighbor_cluster1, neighbor_cluster2, ...}}
    lat_range = sea_level_anomaly_data["latitude"].shape[0]
    latitudes = sea_level_anomaly_data.latitude.values
    long_range = sea_level_anomaly_data["longitude"].shape[0]
    longitudes = sea_level_anomaly_data.longitude.values
    data = sea_level_anomaly_data["sla"].values
    unique_pairs = set()
    first_longitude, last_longitude, lat_for_first_longitude, lat_for_last_longitude = find_first_last_longitude(
        lat_range, long_range, nan_mask)

    print(f"first longitude: {first_longitude}, last longitude: {last_longitude}")
    print(
        f"first long {latitudes[lat_for_first_longitude], longitudes[first_longitude]}, last long {latitudes[lat_for_last_longitude], longitudes[last_longitude]}")

    # extract all unique pairs of grid points that are neighbors with their time series
    unique_pairs_with_time_series = []
    neighbors = iteratively_find_neighbors(data, distance_function, first_longitude, last_longitude,
                                           lat_lon_to_clusters, lat_range, latitudes, long_range, longitudes, nan_mask,
                                           neighbors, unique_pairs, unique_pairs_with_time_series)
    return neighbors, unique_pairs_with_time_series


def iteratively_find_neighbors(data, distance_function, first_longitude, last_longitude, lat_lon_to_clusters, lat_range,
                               latitudes, long_range, longitudes, nan_mask, neighbors, unique_pairs,
                               unique_pairs_with_time_series):
    """
    Find neighbors for each grid point
    :param data:
    :param distance_function:
    :param first_longitude:
    :param last_longitude:
    :param lat_lon_to_clusters:
    :param lat_range:
    :param latitudes:
    :param long_range:
    :param longitudes:
    :param nan_mask:
    :param neighbors:
    :param unique_pairs:
    :param unique_pairs_with_time_series:
    :return:
    """
    # iterate through latitudes and longitudes and find neighbors for each grid point
    for i in tqdm(range(lat_range)):
        for j in (range(long_range)):
            if nan_mask[i, j]:
                continue
            neighbors[lat_lon_to_clusters[latitudes[i], longitudes[j]]] = set()
            # direct neighbors
            neighbor_positions = [
                (i - 1, j),  # North
                (i + 1, j),  # South
                (i, (j - 1) % long_range),  # West (wraps around)
                (i, (j + 1) % long_range),  # East (wraps around)
            ]
            # check if the grid point is on the edge of the grid, because of the interpolation there might be nan values at the edges
            if j == last_longitude:
                neighbor_positions.extend([(i, first_longitude), (i - 1, first_longitude), (i + 1, first_longitude)])
            if j == first_longitude:
                neighbor_positions.extend([(i, last_longitude), (i - 1, last_longitude), (i + 1, last_longitude)])
            # diagonal neighbors
            neighbor_positions.extend([
                ((i - 1), (j - 1) % long_range),  # Northwest
                ((i - 1), (j + 1) % long_range),  # Northeast
                ((i + 1), (j - 1) % long_range),  # Southwest
                ((i + 1), (j + 1) % long_range),  # Southeast
            ])
            # Handle out-of-bounds positions
            neighbor_positions_without_out_of_bounds = [
                (pos[0], pos[1]) if 0 <= pos[0] < lat_range else None
                for pos in neighbor_positions
            ]
            valid_neighbor_positions = [(pos[0], pos[1]) for pos in neighbor_positions_without_out_of_bounds if
                                        not nan_mask[pos[0], pos[1]]]
            # add valid neighbors to the set of neighbors
            cluster1 = lat_lon_to_clusters[latitudes[i], longitudes[j]]
            for pos in valid_neighbor_positions:
                if pos is not None and not nan_mask[pos[0], pos[1]]:
                    cluster2 = lat_lon_to_clusters[latitudes[pos[0]], longitudes[pos[1]]]
                    neighbors[cluster1].add(cluster2)

                    if not ((latitudes[i], longitudes[j]), (latitudes[pos[0]], longitudes[pos[1]])) in unique_pairs or (
                            (latitudes[pos[0]], longitudes[pos[1]]), (latitudes[i], longitudes[j])) in unique_pairs:
                        unique_pairs_with_time_series.append(
                            (distance_function, (latitudes[i], longitudes[j]), data[:, i, j],
                             (latitudes[pos[0]], longitudes[pos[1]]),
                             data[:, pos[0], pos[1]]))
                        unique_pairs.add(((latitudes[i], longitudes[j]), (latitudes[pos[0]], longitudes[pos[1]])))
    neighbors = ensure_bidirectional_neighbors(neighbors)
    return neighbors


def wrap_distance_function(args):
    """
    Wrap the distance function to calculate the distance between two points and return the points and the distance
    :param args: distance_function, lat1, lon1, time_series1, lat2, lon2, time_series2
    :return:
    """
    distance_function, (lat1, lon1), time_series1, (lat2, lon2), time_series2 = args
    distance = distance_function(lat1, lon1, time_series1, lat2, lon2, time_series2)
    return (lat1, lon1), (lat2, lon2), distance


def calculate_distances(unique_pairs_with_time_series, lat_lon_to_clusters: {(float, float): int}) -> dict:
    """
    Calculate distances between each neighbor_pair of grid points
    :param lat_lon_to_clusters:
    :param unique_pairs_with_time_series:
    :return:
    """
    logger.info(f"Calculating distances for {len(unique_pairs_with_time_series)} pairs of grid points")
    distances = {}
    results = Parallel(n_jobs=-2, verbose=1)(
        delayed(wrap_distance_function)(args) for args in unique_pairs_with_time_series)
    for (lat1, lon1), (lat2, lon2), distance in results:
        if distance is not numpy.nan:
            distances[(lat_lon_to_clusters[lat1, lon1], lat_lon_to_clusters[lat2, lon2])] = distance
            distances[(lat_lon_to_clusters[lat2, lon2], lat_lon_to_clusters[lat1, lon1])] = distance
    return distances


def recalculate_distance(cluster2_grid_points: set, neighbor_grid_points: set,
                         distance_function,
                         distance_to_cluster1: float, cluster1_grid_points: set, sla_shm_name, shared_sla_data_shape,
                         shared_sla_data_type, min_lat, min_lon, resolution) -> float:
    """
    Recalculate the distance between a cluster and a neighbor
    :param resolution:
    :param min_lon:
    :param min_lat:
    :param shared_sla_data_type:
    :param shared_sla_data_shape:
    :param sla_shm_name:
    :param cluster1_grid_points:
    :param distance_to_cluster1:
    :param neighbor_grid_points:
    :param cluster2_grid_points:
    :param distance_function:
    :return:
    """
    # attach to shared memory
    sla_shm = shm.SharedMemory(name=sla_shm_name)
    # Create a NumPy array backed by shared memory
    shared_sla_data = np.ndarray(shared_sla_data_shape, dtype=shared_sla_data_type, buffer=sla_shm.buf)
    current_distances = []
    for lat_long_cluster2 in cluster2_grid_points:
        id_x1, id_y1 = lat_lon_to_index(lat_long_cluster2[0], lat_long_cluster2[1], min_lat, min_lon, resolution)
        for lat_long_neighbor in neighbor_grid_points:
            # distance_function, lat1, lon1, time_series1, lat2, lon2, time_series2
            id_x2, id_y2 = lat_lon_to_index(lat_long_neighbor[0], lat_long_neighbor[1], min_lat, min_lon, resolution)
            current_distances.append(
                distance_function(lat_long_cluster2[0], lat_long_cluster2[1], shared_sla_data[:, id_x1, id_y1],
                                  lat_long_neighbor[0], lat_long_neighbor[1], shared_sla_data[:, id_x2, id_y2]))
    summed_distances = np.sum(current_distances)
    new_distance = summed_distances / len(current_distances)
    new_distance = (distance_to_cluster1 * len(cluster1_grid_points) + new_distance * len(cluster2_grid_points)) / (
            len(cluster2_grid_points) + len(cluster1_grid_points))
    sla_shm.close()
    return new_distance


def clustering(clustering_results, sea_level_anomaly_data, k, neighbors, distances,
               distance_function, output_dir) -> dict:
    """
    Hierarchical clustering of neighboring grid points
    :param output_dir:
    :param distance_function:
    :param clustering_results:
    :param sea_level_anomaly_data:
    :param k:
    :param neighbors:
    :param distances:
    :return:
    """
    solutions_for_k = {}
    number_grid_points = len(clustering_results.keys())
    for _ in tqdm(range(number_grid_points)):
        old_len_clustering_results = len(clustering_results.keys())
        if len(clustering_results.keys()) <= 1:
            logger.warning("Clustering continued until only one cluster was left")
            break
        if len(distances.keys()) <= 1:
            logger.warning("Distances empty")
            logger.warning(f"number of clusters left {len(clustering_results.keys())}")
            logger.info(f"number of neighbors left {len(neighbors)}")
            # average number of neighbors per cluster
            logger.info(
                f"average number of neighbors per cluster {np.mean([len(value) for value in neighbors.values()])}")
            return {len(clustering_results.keys()): clustering_results}
        min_distance_pair = min(distances, key=distances.get)
        clustering_results, neighbors, distances = merge_clusters(clustering_results, distance_function, distances,
                                                                  min_distance_pair, neighbors,
                                                                  sea_level_anomaly_data)
        if len(clustering_results.keys()) in k:
            solutions_for_k[len(clustering_results.keys())] = clustering_results.copy()
            if len(clustering_results.keys()) == min(k):
                break
            continue

    return solutions_for_k


def distance_recalculation(args):
    """
    Recalculate the distances between a new cluster and its neighbors
    :param args: (neighbor, {grid_point for grid_point in clustering_results[neighbor]}, grid_points1, grid_points2,
             distance_function, distances.get((cluster1, neighbor)),
             distances.get((cluster2, neighbor)), sla_shm_name, shared_sla_data_shape, shared_sla_data_type,
             MIN_LATITUDE, MIN_LONGITUDE, RESOLUTION)
    :return:
    """

    (neighbor_id, neighbor_grid_points, grid_points1, grid_points2, distance_function, distance_to_cluster1,
     distance_to_cluster2, sla_shm_name, shared_sla_data_shape, shared_sla_data_type, min_lat, min_lon,
     resolution) = args

    new_distance = 0.0
    if distance_to_cluster1 and distance_to_cluster2:
        new_distance = (distance_to_cluster1 * len(grid_points1) + (
                distance_to_cluster2 * len(
            grid_points2))) / (len(grid_points1) + len(grid_points2))

    # (cluster2_grid_points: set, neighbor_grid_points: set,
    #                          distance_function,
    #                          distance_to_cluster1: float, cluster1_grid_points: set, sla_shm_name, shared_sla_data_shape,
    #                          shared_sla_data_type, min_lat, min_lon, resolution)
    elif distance_to_cluster1 and not distance_to_cluster2:
        new_distance = recalculate_distance(grid_points2, neighbor_grid_points, distance_function,
                                            distance_to_cluster1, grid_points1, sla_shm_name, shared_sla_data_shape,
                                            shared_sla_data_type, min_lat, min_lon, resolution)
    elif distance_to_cluster2 and not distance_to_cluster1:
        new_distance = recalculate_distance(grid_points1, neighbor_grid_points, distance_function,
                                            distance_to_cluster2, grid_points2, sla_shm_name, shared_sla_data_shape,
                                            shared_sla_data_type, min_lat, min_lon, resolution)

    return [neighbor_id, new_distance]


def new_recalculate_distance_function(sla_data: numpy.ndarray, cluster1: int, cluster2: int, distance_function,
                                      neighbor: int, clustering_results: {int: (float, float)}, distance_cluster1,
                                      distance_cluster2) -> float:
    """
    Recalculate the distance between a cluster and a neighbor
    :param distance_cluster1:
    :param distance_cluster2:
    :param sla_data:
    :param cluster1:
    :param cluster2:
    :param distance_function:
    :param neighbor:
    :param clustering_results:
    :return:
    """
    new_distance = None
    grid_points1 = clustering_results.get(cluster1)
    grid_points2 = clustering_results.get(cluster2)
    grid_points_neighbor = clustering_results.get(neighbor)
    grid_points_neighbor_with_timeseries = [(lat, lon, sla_data[:,
                                                       helper.lat_lon_to_index(lat, lon, MIN_LATITUDE, MIN_LONGITUDE,
                                                                               RESOLUTION)[0],
                                                       helper.lat_lon_to_index(lat, lon, MIN_LATITUDE, MIN_LONGITUDE,
                                                                               RESOLUTION)[1]]) for lat, lon in
                                            grid_points_neighbor]
    if distance_cluster1 is not None and distance_cluster2 is not None:
        new_distance = (distance_cluster1 * len(grid_points1) + (
                distance_cluster2 * len(
            grid_points2))) / (len(grid_points1) + len(grid_points2))
    if distance_cluster1 is not None and distance_cluster2 is None:
        grid_points2_with_timeseries = [(lat, lon, sla_data[:,
                                                   helper.lat_lon_to_index(lat, lon, MIN_LATITUDE, MIN_LONGITUDE,
                                                                           RESOLUTION)[0],
                                                   helper.lat_lon_to_index(lat, lon, MIN_LATITUDE, MIN_LONGITUDE,
                                                                           RESOLUTION)[1]]) for lat, lon in
                                        grid_points2]
        grid_point_pairs_with_timeseries = [
            (distance_function, (grid_points_neighbor[0], grid_points_neighbor[1]), grid_points_neighbor[2],
             (grid_point2[0], grid_point2[1]), grid_point2[2]) for
            grid_points_neighbor, grid_point2 in
            zip(grid_points_neighbor_with_timeseries, grid_points2_with_timeseries)]
        # recalculate distances in parallel using joblib
        results = Parallel(n_jobs=-2)(
            delayed(wrap_distance_function)(args) for args in grid_point_pairs_with_timeseries)
        summed_distances = np.sum([distance for _, _, distance in results])
        new_distance = (distance_cluster1 * len(grid_points1) + summed_distances * len(
            grid_points2)) / (len(grid_points1) + len(grid_points2))
    if distance_cluster2 is not None and distance_cluster1 is None:
        grid_points1_with_timeseries = [(lat, lon, sla_data[:,
                                                   helper.lat_lon_to_index(lat, lon, MIN_LATITUDE, MIN_LONGITUDE,
                                                                           RESOLUTION)[0],
                                                   helper.lat_lon_to_index(lat, lon, MIN_LATITUDE, MIN_LONGITUDE,
                                                                           RESOLUTION)[1]]) for lat, lon in
                                        grid_points1]
        grid_point_pairs_with_timeseries = [
            (distance_function, (grid_points_neighbor[0], grid_points_neighbor[1]), grid_points_neighbor[2],
             (grid_point1[0], grid_point1[1]), grid_point1[2]) for
            grid_points_neighbor, grid_point1 in
            zip(grid_points_neighbor_with_timeseries, grid_points1_with_timeseries)]
        # recalculate distances in parallel using joblib
        results = Parallel(n_jobs=-2)(
            delayed(wrap_distance_function)(args) for args in grid_point_pairs_with_timeseries)
        summed_distances = np.sum([distance for _, _, distance in results])
        new_distance = (distance_cluster2 * len(grid_points2) + summed_distances * len(
            grid_points1)) / (len(grid_points1) + len(grid_points2))
    return new_distance


def merge_clusters(clustering_results, distance_function, distances, min_distance_pair, neighbors,
                   sla_data: numpy.ndarray):
    """
    merge two clusters
    :param sla_data:
    :param clustering_results:
    :param distance_function:
    :param distances:
    :param min_distance_pair:
    :param neighbors:
    :return:
    """
    cluster1, cluster2 = min_distance_pair

    # get unique neighbors from cluster 1 and cluster 2
    new_neighbors = neighbors[cluster1].union(neighbors[cluster2])
    new_neighbors.remove(cluster1)
    new_neighbors.remove(cluster2)
    # recalculate distances to all neighbors
    distances.pop((cluster1, cluster2))
    distances.pop((cluster2, cluster1))

    results = []
    for neighbor in new_neighbors:
        distance_cluster1 = distances.get((cluster1, neighbor))
        distance_cluster2 = distances.get((cluster2, neighbor))
        new_distance = new_recalculate_distance_function(sla_data, cluster1, cluster2, distance_function, neighbor,
                                                         clustering_results, distance_cluster1, distance_cluster2)
        distances[(cluster1, neighbor)] = new_distance
        distances[(neighbor, cluster1)] = new_distance
        if neighbor in neighbors[cluster2]:
            distances.pop((cluster2, neighbor))
            distances.pop((neighbor, cluster2))
            neighbors[neighbor].remove(cluster2)
    neighbors.pop(cluster2)
    neighbors[cluster1] = new_neighbors

    clustering_results[cluster1].extend(clustering_results[cluster2])
    clustering_results.pop(cluster2)
    for neighbor in new_neighbors:
        # ensure bidirectionality
        if cluster1 not in neighbors[neighbor]:
            neighbors[neighbor].add(cluster1)
    return clustering_results, neighbors, distances


def index_to_lat_lon(x, y, lat_min, lon_min, resolution) -> (float, float):
    """
    Convert an index to a latitude and longitude
    :param x:
    :param y:
    :param lat_min:
    :param lon_min:
    :param resolution:
    :return:
    """
    lat = lat_min + x * resolution
    lon = lon_min + y * resolution
    return lat, lon


def lat_lon_to_index(lat, lon, lat_min, lon_min, resolution) -> (int, int):
    """
    Convert a latitude and longitude to an index
    :param lat:
    :param lon:
    :param lat_min:
    :param lon_min:
    :param resolution:
    :return:
    """
    x = int((lat - lat_min) / resolution)
    y = int((lon - lon_min) / resolution)
    return x, y


def start_clustering(sea_level_anomaly_data: xarray.Dataset, k: [int], distance_function, out_dir: str):
    """
    start hierarchical neighborhood clustering
    :param out_dir:
    :param distance_function:
    :param sea_level_anomaly_data:
    :param k:
    :return:
    """
    global MIN_LATITUDE
    global MIN_LONGITUDE
    global RESOLUTION
    MIN_LATITUDE = sea_level_anomaly_data.latitude.min().values
    MIN_LONGITUDE = sea_level_anomaly_data.longitude.min().values
    RESOLUTION = sea_level_anomaly_data.latitude.values[1] - sea_level_anomaly_data.latitude.values[0]
    # use profiler to find bottlenecks
    profiler = cProfile.Profile()
    profiler.enable()
    k = sorted(k)
    data = sea_level_anomaly_data["sla"].values
    # find grid points with NaN values
    # nan_mask = sea_level_anomaly_data["sla"].isnull().values
    nan_mask = numpy.isnan(data).any(axis=0)

    lat_lon_to_idx = {(lat, lon): (i, j) for i, lat in enumerate(sea_level_anomaly_data.latitude.values) for j, lon in
                      enumerate(sea_level_anomaly_data.longitude.values)}
    # in the beginning each grid point is its own clusters
    clusters = {}
    counter = 0
    for lat in sea_level_anomaly_data.latitude.values:
        for lon in sea_level_anomaly_data.longitude.values:
            if nan_mask[lat_lon_to_idx[lat, lon]]:
                continue
            else:
                clusters[counter] = [(lat, lon)]
                counter += 1
    lat_lon_to_clusters = {value[0]: key for key, value in clusters.items()}
    # for each cluster find out which clusters are neighbors (direct and diagonal)
    neighbors, unique_pairs_with_timeseries = find_neighbors(sea_level_anomaly_data, distance_function,
                                                             lat_lon_to_clusters, nan_mask)
    print(f"len grid points: {len(neighbors)}")
    print(f"avg number of neighbors: {np.mean([len(value) for value in neighbors.values()])}")
    # calculate initial distances between neighbors
    distances = calculate_distances(unique_pairs_with_timeseries, lat_lon_to_clusters)
    print(f"number of distances: {len(distances)}")
    counter = 0

    for key, dist in distances.items():
        if np.isnan(dist):
            counter += 1
    print(f"number of nan distances: {counter}")
    logger.info(
        f"min distance: {min(distances.items(), key=lambda x: x[1])}, max distance: {max(distances.items(), key=lambda x: x[1])}, number of grid points: {len(clusters.keys())}")
    # hierarchical neighbor clustering
    clusters = clustering(clusters, data, k, neighbors, distances, distance_function, out_dir)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)
    for current_k in clusters.keys():
        # plot results
        plotting.plot_clustering(clusters[current_k], out_dir, RESOLUTION, name=f"clustering_{current_k}")

        # save as netcdf file
        cluster_data = numpy.zeros((sea_level_anomaly_data.latitude.size, sea_level_anomaly_data.longitude.size))
        cluster_number = 0
        clustering_dict = clusters[current_k]
        for cluster in clustering_dict.keys():
            for grid_point in clustering_dict[cluster]:
                # get index of lat long in sea_level_anomaly_data
                lat_index = numpy.where(sea_level_anomaly_data.latitude.values == grid_point[0])[0][0]
                long_index = numpy.where(sea_level_anomaly_data.longitude.values == grid_point[1])[0][0]
                cluster_data[lat_index, long_index] = cluster_number
            cluster_number += 1
        cluster_data = xarray.DataArray(cluster_data, dims=["latitude", "longitude"])
        cluster_data = cluster_data.assign_coords(latitude=sea_level_anomaly_data.latitude,
                                                  longitude=sea_level_anomaly_data.longitude)
        cluster_data.to_netcdf(f"{out_dir}/clusters_{len(clustering_dict.keys())}.nc")
    return
