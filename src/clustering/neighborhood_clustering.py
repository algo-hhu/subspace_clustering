import os
from dataclasses import dataclass

import numpy
import numpy as np
import xarray
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from src import helper, plotting
from src.clustering.cluster_entities.initial_clustering import InitialClustering
from src.clustering.connectivity_helper import find_first_last_longitude, ensure_bidirectional_neighbors
from src.helper import save_clustering


@dataclass
class NeighborhoodClustering(InitialClustering):
    sea_level_anomaly_data: xarray.Dataset
    number_of_clusters: list[int]
    distance_function: callable
    out_dir: str
    data_array: numpy.ndarray = None
    min_lat: float = None
    min_lon: float = None
    resolution: float = None

    def start_initial_clustering(self):
        """
        start hierarchical neighborhood clustering
        :return:
        """
        self.min_lat = self.sea_level_anomaly_data.latitude.min().values
        self.min_lon = self.sea_level_anomaly_data.longitude.min().values
        self.resolution = self.sea_level_anomaly_data.latitude.values[1] - self.sea_level_anomaly_data.latitude.values[
            0]
        # # use profiler to find bottlenecks
        # profiler = cProfile.Profile()
        # profiler.enable()
        self.data_array = self.sea_level_anomaly_data["sla"].values
        # find grid points with NaN values
        # nan_mask = sea_level_anomaly_data["sla"].isnull().values
        nan_mask = numpy.isnan(self.data_array).any(axis=0)

        lat_lon_to_idx = {(lat, lon): (i, j) for i, lat in enumerate(self.sea_level_anomaly_data.latitude.values) for
                          j, lon
                          in
                          enumerate(self.sea_level_anomaly_data.longitude.values)}
        # in the beginning, each grid point is its own clusters
        clusters = {}
        counter = 0
        for lat in self.sea_level_anomaly_data.latitude.values:
            for lon in self.sea_level_anomaly_data.longitude.values:
                if nan_mask[lat_lon_to_idx[lat, lon]]:
                    continue
                else:
                    clusters[counter] = [(lat, lon)]
                    counter += 1
        lat_lon_to_clusters = {value[0]: key for key, value in clusters.items()}
        # for each cluster, find out which clusters are neighbors (direct and diagonal)
        neighbors, unique_pairs_with_timeseries = self.find_neighbors(lat_lon_to_clusters, nan_mask)
        print(f"len grid points: {len(neighbors)}")
        print(f"avg number of neighbors: {np.mean([len(value) for value in neighbors.values()])}")
        # calculate initial distances between neighbors
        distances = self.calculate_initial_distances(unique_pairs_with_timeseries, lat_lon_to_clusters)
        print(f"number of distances: {len(distances)}")
        counter = 0

        for key, dist in distances.items():
            if np.isnan(dist):
                counter += 1
        print(f"number of nan distances: {counter}")
        logger.info(
            f"min distance: {min(distances.items(), key=lambda x: x[1])}, max distance: {max(distances.items(), key=lambda x: x[1])}, number of grid points: {len(clusters.keys())}")
        # hierarchical neighbor clustering
        clustering_results = self.clustering(clusters, neighbors, distances)

        for current_k in clustering_results.keys():
            # change cluster ids to start from 0 to k
            clustering_results[current_k] = {i: clustering_results[current_k][cluster_id] for i, cluster_id in
                                             enumerate(clustering_results[current_k].keys())}
            # plot results
            name = f"clustering_{current_k}"
            plotting.plot_clustering_without_preassigned_colors(clustering_results[current_k], self.out_dir,
                                                                self.resolution,
                                                                name)
            current_out_dir = os.path.join(self.out_dir, f"{current_k}")
            if not os.path.exists(current_out_dir):
                os.makedirs(current_out_dir)
            for cluster_id, grid_points in clustering_results[current_k].items():
                name = f"cluster_{cluster_id}"
                current_cluster = {cluster_id: grid_points}
                plotting.plot_clustering_without_preassigned_colors(current_cluster, current_out_dir, self.resolution,
                                                                    name)

            # save as a netcdf file
            clustering_dict = clustering_results[current_k]
            filename = f"clustering_{current_k}"
            save_clustering(clustering_dict, self.out_dir, self.sea_level_anomaly_data, filename)
        return

    def clustering(self, all_clusters: dict[int, tuple[float, float]], neighbors: dict[int, set[int]],
                   distances: dict[tuple[int, int], float]) -> dict:
        """
        Hierarchical clustering of neighboring grid points
        :param all_clusters:
        :param neighbors:
        :param distances:
        :return:
        """
        solutions_for_k = {}
        number_grid_points = len(all_clusters.keys())
        for _ in tqdm(range(number_grid_points)):
            # check if all clusters are merged
            if len(all_clusters.keys()) <= 1:
                logger.warning("Clustering continued until only one cluster was left")
                break
            # check if there are neighbors left, if this is not the case, there are isolated patches on the map
            if len(distances.keys()) <= 1:
                self.warn_missing_neighbors(all_clusters, neighbors)
                return {len(all_clusters.keys()): all_clusters}
            # get the pair of clusters with the smallest distance
            min_distance_pair = min(distances, key=distances.get)
            # merge the two clusters that have the smallest distance
            all_clusters, neighbors, distances = self.merge_clusters(all_clusters, distances,
                                                                     min_distance_pair, neighbors)
            # check if one of the desired k values is reached
            if len(all_clusters.keys()) in self.number_of_clusters:
                solutions_for_k[len(all_clusters.keys())] = {key: value[:] for key, value in
                                                             all_clusters.items()}
                # stop if all desired k values are reached
                if len(all_clusters.keys()) == min(self.number_of_clusters):
                    break
                continue

        return solutions_for_k

    @staticmethod
    def warn_missing_neighbors(all_clusters, neighbors):
        logger.warning("Distances empty")
        logger.warning(f"number of clusters left {len(all_clusters.keys())}")
        logger.info(f"number of neighbors left {len(neighbors)}")
        # average number of neighbors per cluster
        logger.info(
            f"average number of neighbors per cluster {np.mean([len(value) for value in neighbors.values()])}")

    def merge_clusters(self, all_clusters: dict[int, list[tuple[float, float]]],
                       distances: dict[tuple[int, int], float], min_distance_pair: tuple[int, int],
                       neighbors: dict[int, set[int]]) -> tuple[
        dict[int, list[tuple[float, float]]], dict[int, set[int]], dict[tuple[int, int], float]]:
        """
        Merge two clusters and update the distances, neighbors and all_clusters
        :param all_clusters:
        :param distances:
        :param min_distance_pair:
        :param neighbors:
        :return all_clusters, neighbors, distances:
        """
        cluster1, cluster2 = min_distance_pair  # the two clusters to merge

        # get unique neighbors from cluster 1 and cluster 2
        new_neighbors = set(neighbors[cluster1].union(neighbors[cluster2]))
        new_neighbors.remove(cluster1)
        new_neighbors.remove(cluster2)
        # recalculate distances to all neighbors
        distances.pop((cluster1, cluster2))
        distances.pop((cluster2, cluster1))

        for neighbor in new_neighbors:
            distance_cluster1 = distances.get((cluster1, neighbor))
            distance_cluster2 = distances.get((cluster2, neighbor))
            new_distance = self.new_recalculate_distance_function(cluster1, cluster2, neighbor, all_clusters,
                                                                  distance_cluster1, distance_cluster2)
            distances[(cluster1, neighbor)] = new_distance
            distances[(neighbor, cluster1)] = new_distance
            if neighbor in neighbors[cluster2]:
                distances.pop((cluster2, neighbor))
                distances.pop((neighbor, cluster2))
                neighbors[neighbor].remove(cluster2)
        neighbors.pop(cluster2)
        neighbors[cluster1] = new_neighbors

        all_clusters[cluster1].extend(all_clusters[cluster2])
        all_clusters.pop(cluster2)
        for neighbor in new_neighbors:
            # ensure bidirectionality
            if cluster1 not in neighbors[neighbor]:
                neighbors[neighbor].add(cluster1)
        return all_clusters, neighbors, distances

    def new_recalculate_distance_function(self, cluster1: int, cluster2: int, neighbor: int,
                                          clustering_results: {int: (float, float)}, distance_cluster1,
                                          distance_cluster2) -> float:
        """
        Recalculate the distance between a cluster and a neighbor
        :param distance_cluster1:
        :param distance_cluster2:
        :param cluster1:
        :param cluster2:
        :param neighbor:
        :param clustering_results:
        :return:
        """
        new_distance = None
        grid_points1 = clustering_results.get(cluster1)
        grid_points2 = clustering_results.get(cluster2)
        grid_points_neighbor = clustering_results.get(neighbor)
        grid_points_neighbor_with_timeseries = [(lat, lon, self.data_array[:,
                                                           helper.lat_lon_to_index(lat, lon, self.min_lat,
                                                                                   self.min_lon,
                                                                                   self.resolution)[0],
                                                           helper.lat_lon_to_index(lat, lon, self.min_lat,
                                                                                   self.min_lon,
                                                                                   self.resolution)[1]]) for lat, lon in
                                                grid_points_neighbor]
        if distance_cluster1 is not None and distance_cluster2 is not None:
            new_distance = (distance_cluster1 * len(grid_points1) + (
                    distance_cluster2 * len(
                grid_points2))) / (len(grid_points1) + len(grid_points2))
        if distance_cluster1 is not None and distance_cluster2 is None:
            grid_points2_with_timeseries = [(lat, lon, self.data_array[:,
                                                       helper.lat_lon_to_index(lat, lon, self.min_lat, self.min_lon,
                                                                               self.resolution)[0],
                                                       helper.lat_lon_to_index(lat, lon, self.min_lat, self.min_lon,
                                                                               self.resolution)[1]]) for lat, lon in
                                            grid_points2]
            grid_point_pairs_with_timeseries = [
                (self.distance_function, (grid_points_neighbor[0], grid_points_neighbor[1]), grid_points_neighbor[2],
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
            grid_points1_with_timeseries = [(lat, lon, self.data_array[:,
                                                       helper.lat_lon_to_index(lat, lon, self.min_lat, self.min_lon,
                                                                               self.resolution)[0],
                                                       helper.lat_lon_to_index(lat, lon, self.min_lat, self.min_lon,
                                                                               self.resolution)[1]]) for lat, lon in
                                            grid_points1]
            grid_point_pairs_with_timeseries = [
                (self.distance_function, (grid_points_neighbor[0], grid_points_neighbor[1]), grid_points_neighbor[2],
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

    def find_neighbors(self, lat_lon_to_clusters: dict[tuple[float, float], int], nan_mask: np.ndarray) -> (
            dict, set):
        """
        Find neighbors for each grid point
        :param nan_mask:
        :param lat_lon_to_clusters:
        :return:
        """
        neighbors = {}  # {cluster: {neighbor_cluster1, neighbor_cluster2, ...}}
        lat_range = self.sea_level_anomaly_data["latitude"].shape[0]
        latitudes = self.sea_level_anomaly_data.latitude.values
        long_range = self.sea_level_anomaly_data["longitude"].shape[0]
        longitudes = self.sea_level_anomaly_data.longitude.values
        unique_pairs = set()
        first_longitude, last_longitude, lat_for_first_longitude, lat_for_last_longitude = find_first_last_longitude(
            lat_range, long_range, nan_mask)

        print(f"first longitude: {first_longitude}, last longitude: {last_longitude}")
        print(
            f"first long {latitudes[lat_for_first_longitude], longitudes[first_longitude]}, last long {latitudes[lat_for_last_longitude], longitudes[last_longitude]}")

        # extract all unique pairs of grid points that are neighbors with their time series
        unique_pairs_with_time_series = []
        neighbors = self.iteratively_find_neighbors(first_longitude, last_longitude,
                                                    lat_lon_to_clusters, lat_range, latitudes, long_range, longitudes,
                                                    nan_mask,
                                                    neighbors, unique_pairs, unique_pairs_with_time_series)
        return neighbors, unique_pairs_with_time_series

    def iteratively_find_neighbors(self, first_longitude, last_longitude, lat_lon_to_clusters,
                                   lat_range, latitudes, long_range, longitudes, nan_mask, neighbors, unique_pairs,
                                   unique_pairs_with_time_series):
        """
        Find neighbors for each grid point
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
                    neighbor_positions.extend(
                        [(i, first_longitude), (i - 1, first_longitude), (i + 1, first_longitude)])
                if j == first_longitude:
                    neighbor_positions.extend([(i, last_longitude), (i - 1, last_longitude), (i + 1, last_longitude)])
                # # diagonal neighbors
                # neighbor_positions.extend([
                #     ((i - 1), (j - 1) % long_range),  # Northwest
                #     ((i - 1), (j + 1) % long_range),  # Northeast
                #     ((i + 1), (j - 1) % long_range),  # Southwest
                #     ((i + 1), (j + 1) % long_range),  # Southeast
                # ])
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

                        if not ((latitudes[i], longitudes[j]),
                                (latitudes[pos[0]], longitudes[pos[1]])) in unique_pairs or (
                                (latitudes[pos[0]], longitudes[pos[1]]), (latitudes[i], longitudes[j])) in unique_pairs:
                            unique_pairs_with_time_series.append(
                                (self.distance_function, (latitudes[i], longitudes[j]), self.data_array[:, i, j],
                                 (latitudes[pos[0]], longitudes[pos[1]]),
                                 self.data_array[:, pos[0], pos[1]]))
                            unique_pairs.add(((latitudes[i], longitudes[j]), (latitudes[pos[0]], longitudes[pos[1]])))
        neighbors = ensure_bidirectional_neighbors(neighbors)
        return neighbors

    @staticmethod
    def calculate_initial_distances(unique_pairs_with_time_series, lat_lon_to_clusters: {(float, float): int}) -> dict:
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

    @staticmethod
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


def wrap_distance_function(args):
    """
    Wrap the distance function to calculate the distance between two points and return the points and the distance
    :param args: distance_function, lat1, lon1, time_series1, lat2, lon2, time_series2
    :return:
    """
    distance_function, (lat1, lon1), time_series1, (lat2, lon2), time_series2 = args
    distance = distance_function(lat1, lon1, time_series1, lat2, lon2, time_series2)
    return (lat1, lon1), (lat2, lon2), distance
