import cProfile
import multiprocessing.shared_memory as shm
import os
import pickle
import pstats
from multiprocessing import Pool

import dill
import numpy as np
import xarray
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

MIN_LATITUDE = None
MIN_LONGITUDE = None
RESOLUTION = None


def find_neighbors(sea_level_anomaly_data: xarray.Dataset, distance_function, lat_lon_to_clusters: {}) -> (dict, set):
    """
    Find neighbors for each grid point
    :param lat_lon_to_clusters:
    :param distance_function:
    :param sea_level_anomaly_data:
    :return:
    """
    neighbors = {}
    lat_ids = sea_level_anomaly_data["latitude"].shape[0]
    latitudes = sea_level_anomaly_data.latitude.values
    long_ids = sea_level_anomaly_data["longitude"].shape[0]
    longitudes = sea_level_anomaly_data.longitude.values
    data = sea_level_anomaly_data["sla"].values
    unique_pairs = set()
    unique_pairs_with_time_series = []
    for i in tqdm(range(lat_ids)):
        for j in (range(long_ids)):
            if np.isnan(data[:, i, j]).any():
                continue
            neighbors[lat_lon_to_clusters[latitudes[i], longitudes[j]]] = set()
            # direct neighbors
            neighbor_positions = [
                (i - 1, j),  # North
                (i + 1, j),  # South
                (i, (j - 1) % long_ids),  # West (wraps around)
                (i, (j + 1) % long_ids),  # East (wraps around)
            ]
            # diagonal neighbors
            neighbor_positions.extend([
                (i - 1, (j - 1) % long_ids),  # Northwest
                (i - 1, (j + 1) % long_ids),  # Northeast
                (i + 1, (j - 1) % long_ids),  # Southwest
                (i + 1, (j + 1) % long_ids),  # Southeast
            ])
            # Handle out-of-bounds positions
            neighbor_positions = [
                (pos[0], pos[1]) if 0 <= pos[0] < lat_ids else None
                for pos in neighbor_positions
            ]
            cluster1 = lat_lon_to_clusters[latitudes[i], longitudes[j]]
            for pos in neighbor_positions:
                if pos is not None and not np.isnan(data[:, pos[0], pos[1]]).any():
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
    return neighbors, unique_pairs_with_time_series


def ensure_bidirectional_neighbors(neighbors: {}):
    """
    Ensure that neighbor-relationships are bidirectional
    :param neighbors:
    :return:
    """
    for key, value in neighbors.items():
        for neighbor in value:
            if key not in neighbors[neighbor]:
                neighbors[neighbor].append(key)
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
               distance_function) -> dict:
    """
    Hierarchical clustering of neighboring grid points
    :param distance_function:
    :param clustering_results:
    :param sea_level_anomaly_data:
    :param k:
    :param neighbors:
    :param distances:
    :return:
    """
    # shared memory block for sla data
    sla_shm = shm.SharedMemory(create=True, size=sea_level_anomaly_data.nbytes)
    # Create a NumPy array backed by shared memory
    shared_sla_data = np.ndarray(sea_level_anomaly_data.shape, dtype=sea_level_anomaly_data.dtype, buffer=sla_shm.buf)
    # Copy the original data into shared memory
    np.copyto(shared_sla_data, sea_level_anomaly_data)
    try:
        number_grid_points = len(clustering_results.keys())
        for _ in tqdm(range(number_grid_points)):
            old_len_clustering_results = len(clustering_results.keys())
            if len(clustering_results.keys()) <= 1:
                logger.warning("Clustering continued until only one cluster was left")
                break
            if len(distances.keys()) <= 1:
                logger.warning("Distances empty")
                logger.warning(f"number of clusters left {len(clustering_results.keys())}")
                break
            min_distance_pair = min(distances, key=distances.get)
            merge_clusters(clustering_results, distance_function, distances, min_distance_pair, neighbors,
                           sla_shm.name, shared_sla_data.shape, shared_sla_data.dtype)
            if len(clustering_results.keys()) in k:
                # save clustering results using pickle
                with open(f"../output/clustering_results_{len(clustering_results.keys())}.pkl", "wb") as f:
                    pickle.dump(clustering_results, f)
                if len(clustering_results.keys()) == min(k):
                    break
                continue
    finally:
        sla_shm.close()
        sla_shm.unlink()
    return clustering_results


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


def merge_clusters(clustering_results, distance_function, distances, min_distance_pair, neighbors,
                   sla_shm_name, shared_sla_data_shape, shared_sla_data_type):
    """
    merge two clusters
    :param shared_sla_data_type:
    :param shared_sla_data_shape:
    :param sla_shm_name:
    :param clustering_results:
    :param distance_function:
    :param distances:
    :param min_distance_pair:
    :param neighbors:
    :return:
    """
    cluster1, cluster2 = min_distance_pair
    grid_points1 = clustering_results[cluster1]
    grid_points2 = clustering_results[cluster2]

    # get unique neighbors from cluster 1 and cluster 2
    new_neighbors = neighbors[cluster1].union(neighbors[cluster2])
    new_neighbors.remove(cluster1)
    new_neighbors.remove(cluster2)
    # recalculate distances to all neighbors
    distances.pop((cluster1, cluster2))
    distances.pop((cluster2, cluster1))

    # the function needs to know the grid_points of the neighbors, cluster1 and cluster2 and the distance function, additionally the sla at each grid point
    # args: neighbor_id, {neighbor_grid_points: sla}, {cluster1_grid_points: sla}, {cluster2_grid_points: sla}, distance_function, distance_to_cluster1, distance_to_cluster2
    args = [(neighbor, {grid_point for grid_point in clustering_results[neighbor]}, grid_points1, grid_points2,
             distance_function, distances.get((cluster1, neighbor)),
             distances.get((cluster2, neighbor)), sla_shm_name, shared_sla_data_shape, shared_sla_data_type,
             MIN_LATITUDE, MIN_LONGITUDE, RESOLUTION) for neighbor in new_neighbors]

    n_jobs = os.cpu_count() - 1
    with Pool(processes=n_jobs if n_jobs > 0 else None, initializer=lambda: setattr(dill, '_dill', 'pool')) as pool:
        results = pool.map(distance_recalculation, args)

    for neighbor, new_distance in results:
        distances[(cluster1, neighbor)] = new_distance
        distances[(neighbor, cluster1)] = new_distance
        if neighbor in neighbors[cluster2]:
            distances.pop((cluster2, neighbor))
            distances.pop((neighbor, cluster2))
            neighbors[neighbor].remove(cluster2)
    neighbors.pop(cluster2)
    neighbors[cluster1] = new_neighbors

    # for neighbor in new_neighbors:
    #     if neighbor == cluster1 or neighbor == cluster2:
    #         continue
    #     # calculate new distance
    #     new_distance = 0.0
    #
    #     if neighbor in neighbors[cluster1] and neighbor in neighbors[cluster2]:
    #         try:
    #             new_distance = (distances[(cluster1, neighbor)] * len(grid_points1) + (
    #                     distances[(cluster2, neighbor)] * len(
    #                 grid_points2))) / (len(grid_points1) + len(grid_points2))
    #         except KeyError:
    #             logger.warning(f"KeyError: {cluster1, neighbor}, {cluster2, neighbor}")
    #             exit()
    #         distances.pop((cluster1, neighbor))
    #         distances.pop((neighbor, cluster1))
    #         distances.pop((cluster2, neighbor))
    #         distances.pop((neighbor, cluster2))
    #         neighbors[neighbor].remove(cluster2)
    #     elif neighbor in neighbors[cluster1] and not neighbor in neighbors[cluster2]:
    #         new_distance = recalculate_distance(cluster2, neighbor, clustering_results, distance_function,
    #                                             distances[(cluster1, neighbor)], cluster1, lat_lon_to_idx,
    #                                             sea_level_anomaly_data)
    #         distances.pop((cluster1, neighbor))
    #         distances.pop((neighbor, cluster1))
    #     elif neighbor in neighbors[cluster2] and not neighbor in neighbors[cluster1]:
    #         new_distance = recalculate_distance(cluster1, neighbor, clustering_results, distance_function,
    #                                             distances[cluster2, neighbor], cluster2, lat_lon_to_idx,
    #                                             sea_level_anomaly_data)
    #         distances.pop((cluster2, neighbor))
    #         distances.pop((neighbor, cluster2))
    #         neighbors[neighbor].remove(cluster2)
    #         neighbors[neighbor].append(cluster1)
    #     else:
    #         logger.warning("Neighbor not in cluster 1 or cluster 2")
    #
    #     distances[(cluster1, neighbor)] = new_distance
    #     distances[(neighbor, cluster1)] = new_distance
    # neighbors.pop(cluster2)
    # neighbors[cluster1] = new_neighbors
    # update cluster in clustering_results
    clustering_results[cluster1].extend(clustering_results[cluster2])
    clustering_results.pop(cluster2)
    for neighbor in new_neighbors:
        # ensure bidirectionality
        if cluster1 not in neighbors[neighbor]:
            neighbors[neighbor].add(cluster1)


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
    profiler = cProfile.Profile()
    profiler.enable()
    k = sorted(k)
    data = sea_level_anomaly_data["sla"].values
    nan_mask = sea_level_anomaly_data["sla"].isnull().values
    nan_mask = nan_mask[0, :, :]
    clusters = np.full((sea_level_anomaly_data.latitude.size, sea_level_anomaly_data.longitude.size), fill_value=-1,
                       dtype=int)
    lat_lon_to_idx = {(lat, lon): (i, j) for i, lat in enumerate(sea_level_anomaly_data.latitude.values) for j, lon in
                      enumerate(sea_level_anomaly_data.longitude.values)}
    idx_to_lat_lon = {(i, j): (lat, lon) for (lat, lon), (i, j) in lat_lon_to_idx.items()}
    clusters = {current_id: [(lat, lon)] for current_id, (lat, lon) in enumerate(lat_lon_to_idx.keys())}
    lat_lon_to_clusters = {value[0]: key for key, value in clusters.items()}
    neighbors, unique_pairs_with_timeseries = find_neighbors(sea_level_anomaly_data, distance_function,
                                                             lat_lon_to_clusters)
    distances = calculate_distances(unique_pairs_with_timeseries, lat_lon_to_clusters)
    logger.info(f"min distance: {min(distances.keys())}, max distance: {max(distances.keys())}")
    # hierarchical neighbor clustering
    clusters = clustering(clusters, data, k, neighbors, distances, distance_function)
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumulative").print_stats(20)
    exit()
