import os
from dataclasses import dataclass

import numpy
import numpy as np
import xarray
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from src.clustering.cluster_entities.initial_clustering import InitialClustering


@dataclass
class GridPoint:
    id: tuple[int, int]
    latitude: float
    longitude: float
    time_series: numpy.ndarray
    cluster_id: int = None


@dataclass
class Cluster:
    id: int
    centroid: numpy.ndarray
    members: list[GridPoint]
    distances_of_members_to_centroid: int
    neighbors: set[int]


def assign_neighbors_to_clusters(all_clusters, all_grid_points, potential_neighbors_for_grid_points):
    """
    Assign neighbors to clusters based on the potential neighbors for each grid point.
    :param all_clusters:
    :param all_grid_points:
    :param potential_neighbors_for_grid_points:
    :return:
    """
    # assign the neighbors to the clusters
    for grid_point_id, grid_point in all_grid_points.items():
        neighbors = []
        for neighbor_id in potential_neighbors_for_grid_points[grid_point_id]:
            if neighbor_id in all_grid_points:
                neighbor = all_grid_points[neighbor_id]
                neighbors.append(neighbor.cluster_id)
        for neighbor_cluster_id in neighbors:
            all_clusters[grid_point.cluster_id].neighbors.add(neighbor_cluster_id)
    # print average number of neighbors per cluster
    neighbor_counts = [len(cluster.neighbors) for cluster in all_clusters.values()]
    avg_neighbors = sum(neighbor_counts) / len(all_clusters)
    logger.info(f"Average number of neighbors per cluster: {avg_neighbors}")
    return all_clusters


def recalculate_distance(len_cluster1, len_cluster2, len_neighbor, distance_cluster1_to_neighbor,
                         distance_cluster2_to_neighbor, distance_between_min_clusters, cluster1, cluster2, neighbor):
    """
    Recalculate the distance between the new cluster and its neighbors after merging two clusters.
    :param neighbor:
    :param cluster2:
    :param cluster1:
    :param len_cluster1:
    :param len_cluster2:
    :param len_neighbor:
    :param distance_cluster1_to_neighbor:
    :param distance_cluster2_to_neighbor:
    :param distance_between_min_clusters:
    :return:
    """
    new_distance = None
    if distance_cluster1_to_neighbor is not None:
        weighted_distance_cluster1_to_neighbor = ((len_cluster1 + len_neighbor) / (
                len_cluster1 + len_cluster2 + len_neighbor)) * distance_cluster1_to_neighbor
    else:  # recalculate distance based on the centroid of cluster 1 and the neighbor
        weighted_distance_cluster1_to_neighbor = ((len_cluster1 + len_neighbor) / (
                len_cluster1 + len_cluster2 + len_neighbor)) * np.sum((cluster1.centroid - neighbor.centroid) ** 2)
    if distance_cluster2_to_neighbor is not None:
        weighted_distance_cluster2_to_neighbor = ((len_cluster2 + len_neighbor) / (
                len_cluster1 + len_cluster2 + len_neighbor)) * distance_cluster2_to_neighbor
    else:  # recalculate distance based on the centroid of cluster 2 and the neighbor
        weighted_distance_cluster2_to_neighbor = ((len_cluster2 + len_neighbor) / (
                len_cluster1 + len_cluster2 + len_neighbor)) * np.sum((cluster2.centroid - neighbor.centroid) ** 2)
    if distance_between_min_clusters is None:
        logger.error(f"Distance between min clusters is None for pair {cluster1.id} and {cluster2.id}.")
    weighted_distance_between_min_clusters = (len_neighbor / (
            len_cluster1 + len_cluster2 + len_neighbor)) * distance_between_min_clusters
    # calculate the new distance based on the weighted distances
    new_distance = weighted_distance_cluster1_to_neighbor + weighted_distance_cluster2_to_neighbor - weighted_distance_between_min_clusters
    return new_distance


@dataclass
class WardsMethodConnectedNew(InitialClustering):
    sea_level_anomaly_data: xarray.Dataset
    number_of_clusters: list[int]
    distance_function: callable
    out_dir: str
    data_array: numpy.ndarray = None
    min_lat: float = None
    min_lon: float = None
    resolution: float = None
    nan_mask: numpy.ndarray = None

    def start_initial_clustering(self):
        """
        start hierarchical neighborhood clustering
        :return:
        """
        self.min_lat = self.sea_level_anomaly_data.latitude.min().values
        self.min_lon = self.sea_level_anomaly_data.longitude.min().values
        self.resolution = self.sea_level_anomaly_data.latitude.values[1] - self.sea_level_anomaly_data.latitude.values[
            0]
        # in the beginning each cluster is a single point
        self.data_array = self.sea_level_anomaly_data["sla"].values
        # make 2d nan mask, that is True for all points that have at least one NaN in the time series
        self.nan_mask = numpy.isnan(self.data_array).any(axis=0)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        logger.info(f"Starting initial clustering with method {self.__class__.__name__} and distance function ")
        logger.info(f"Begin cluster creation...")
        all_clusters = self.initially_create_clusters()
        logger.info("Calculating initial distances between clusters and their neighbors...")
        # calculate initial distances between clusters and their neighbors
        distances = self.calculate_initial_distances(all_clusters)

        # start the clustering process
        clustering_results = self.perform_clustering(all_clusters, distances)
        # plot and save the clustering results
        self.plot_and_save_clustering_results(clustering_results)
        exit()

    def initially_create_clusters(self):
        """
        Create initial clusters where each point is its own cluster.
        :return:
        """
        all_clusters = {}
        counter = 0
        potential_neighbors_for_grid_points = {}
        all_grid_points = {}
        # iterate over all spatial points in the data array (time, latitude, longitude)
        time, latitudes, longitudes = self.data_array.shape
        for i in range(latitudes):
            for j in range(longitudes):
                if self.nan_mask[i, j]:
                    continue
                time_series = self.data_array[:, i, j]
                # each grid point has its position in the data array as id
                grid_point = GridPoint((i, j), self.sea_level_anomaly_data.latitude[i],
                                       self.sea_level_anomaly_data.longitude[j], time_series)
                all_grid_points[grid_point.id] = grid_point
                # each grid point has a list of potential neighbors, which are the eight points around it
                potential_neighbors_for_grid_points[grid_point.id] = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
                                                                      (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1),
                                                                      (i + 1, j + 1)]
                # check validity of the neighbors with the nan mask
                potential_neighbors_for_grid_points[grid_point.id] = [
                    (x, y) for x, y in potential_neighbors_for_grid_points[grid_point.id]
                    if 0 <= x < latitudes and 0 <= y < longitudes and not self.nan_mask[x, y]]
                # create a cluster for each point, in the beginning, the centroid is the time series itself
                all_clusters[counter] = Cluster(counter, time_series, [grid_point], 0, set())
                grid_point.cluster_id = counter
                counter += 1
        all_clusters = assign_neighbors_to_clusters(all_clusters, all_grid_points, potential_neighbors_for_grid_points)

        return all_clusters

    def perform_clustering(self, all_clusters: dict[int, Cluster], distances: dict[tuple[int, int], float]):
        """
        Perform the clustering process using the Wards method.
        :param distances:
        :param all_clusters:
        :return:
        """
        clustering_results = {}
        total_number_of_clusters = len(all_clusters)
        logger.info(f"Total number of clusters: {total_number_of_clusters}")
        logger.info(f"Begin clustering...")
        for current_number_of_clusters in tqdm(range(total_number_of_clusters)):
            # merge the two clusters with the smallest distance
            try:
                min_pair, min_value = min(distances.items(), key=lambda item: item[1])
            except:
                logger.warning(f"Error in calculating min pair and value.")
                print(distances)
                exit()

            all_clusters, distances = self.merge_min_clusters(all_clusters, min_pair, distances, min_value)
            if len(all_clusters) <= min(self.number_of_clusters):
                exit()
        return clustering_results

    def plot_and_save_clustering_results(self, clustering_results):
        pass

    def calculate_initial_distances(self, all_clusters):
        """
        Calculate the initial distances between clusters and their neighbors.
        :param all_clusters:
        :return:
        """
        distances = {}
        cluster_pairs = set()
        # create a list of all cluster pairs that are neighbors of each other
        for cluster_id, cluster in all_clusters.items():
            for neighbor_cluster_id in cluster.neighbors:
                current_neighbor_cluster = all_clusters[neighbor_cluster_id]
                cluster_pair = tuple(sorted([cluster.id, current_neighbor_cluster.id]))
                cluster_pairs.add(cluster_pair)
        # calculate distances for each pair of clusters in parallel using joblib
        results = Parallel(n_jobs=-1, verbose=1)([delayed(wrap_distance_function)(all_clusters[cluster1_id],
                                                                                  all_clusters[cluster2_id],
                                                                                  self.distance_function) for
                                                  cluster1_id, cluster2_id in cluster_pairs])

        for result in results:
            cluster1, cluster2, distance = result
            distances[tuple(sorted([cluster1.id, cluster2.id]))] = distance
        max_distance = max(distances.values())
        min_distance = min(distances.values())
        logger.info(
            f"Max distance: {max_distance}, Min distance: {min_distance}, Avg distance: {sum(distances.values()) / len(distances)}")
        return distances

    def merge_min_clusters(self, all_clusters, min_pair, distances, distance_between_min_clusters: float):
        """
        Merge the two clusters with the smallest distance and update the distances and the clusters as well as their neighbors.
        :param distance_between_min_clusters:
        :param all_clusters:
        :param min_pair:
        :param distances:
        :return:
        """
        cluster1_id, cluster2_id = min_pair
        cluster1 = all_clusters[cluster1_id]
        cluster2 = all_clusters[cluster2_id]
        # recalculate the centroid of the new cluster
        len_cluster1 = len(cluster1.members)
        len_cluster2 = len(cluster2.members)
        new_centroid = (len_cluster1 * cluster1.centroid + len_cluster2 * cluster2.centroid) / (
                len_cluster1 + len_cluster2)
        # update cluster members
        new_members = cluster1.members + cluster2.members
        # merge neighbor-sets of the two clusters and remove cluster 1 and 2 from the set
        new_neighbors = cluster1.neighbors.union(cluster2.neighbors)
        new_neighbors.discard(cluster1_id)
        new_neighbors.discard(cluster2_id)

        # # Filter out any neighbors that no longer exist in all_clusters
        # new_neighbors = {neighbor_id for neighbor_id in new_neighbors if neighbor_id in all_clusters}

        # recalculate the distance between the new cluster and all neighboring clusters
        for neighbor_id in new_neighbors:
            len_neighbor = len(all_clusters[neighbor_id].members)
            neighbor_cluster = all_clusters[neighbor_id]
            distance_cluster1_to_neighbor = distances.get(tuple(sorted([cluster1_id, neighbor_id])), None)
            distance_cluster2_to_neighbor = distances.get(tuple(sorted([cluster2_id, neighbor_id])), None)
            new_distance = recalculate_distance(len_cluster1, len_cluster2, len_neighbor, distance_cluster1_to_neighbor,
                                                distance_cluster2_to_neighbor, distance_between_min_clusters, cluster1,
                                                cluster2, neighbor_cluster)
            # add the new distance to the distances dictionary
            if new_distance is None:
                logger.error(f"New distance is None for pair {min_pair} and neighbor {neighbor_id}. ")
            distances[tuple(sorted([cluster1_id, neighbor_id]))] = new_distance
            # update the neighbor-set of the neighbor cluster and remove cluster 2 from it
            try:
                neighbor_cluster.neighbors.remove(cluster2_id)
            except KeyError:
                pass
            try:
                neighbor_cluster.neighbors.remove(cluster1_id)
            except KeyError:
                pass
            # remove the distance between the two clusters that were merged
            # remove the distance between the two clusters and the neighbor cluster
            distances.pop(tuple(sorted([cluster2_id, neighbor_id])), None)
        distances.pop(tuple(sorted([cluster1_id, cluster2_id])))
        # remove the old clusters from the all_clusters dictionary
        del all_clusters[cluster1_id]
        del all_clusters[cluster2_id]
        # calculate distances of members to the new centroid

        # create a new cluster with the new centroid and members
        new_cluster_id = cluster1_id
        all_clusters[new_cluster_id] = Cluster(new_cluster_id, new_centroid, new_members, 0, new_neighbors)

        # update the neighbor sets of the new neighbors of the new cluster
        for neighbor_id in new_neighbors:
            neighbor_cluster = all_clusters[neighbor_id]
            # add the new cluster to the neighbor set of the neighbor cluster
            neighbor_cluster.neighbors.add(new_cluster_id)
            # distances are symmetric and have already been calculated for the new cluster
        return all_clusters, distances


def wrap_distance_function(cluster1: Cluster, cluster2: Cluster, distance_function: callable):
    """
    Wrap the distance function to calculate the distance between two clusters.
    :param cluster1:
    :param cluster2:
    :param distance_function:
    :return:
    """
    # calculate the distance between the centroids of the two clusters
    return cluster1, cluster2, distance_function(cluster1.centroid, cluster2.centroid)
