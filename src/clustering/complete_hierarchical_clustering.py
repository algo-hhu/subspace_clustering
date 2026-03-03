import os
import uuid
from dataclasses import dataclass
from typing import List

import numpy
import numpy as np
import xarray
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

from src import plotting
from src.clustering.cluster_entities.initial_clustering import InitialClustering


@dataclass()
class GridPoint:
    id: uuid.UUID
    latitude: float
    longitude: float
    timeseries: np.ndarray


@dataclass
class Cluster:
    id: int
    grid_points: List[GridPoint]


@dataclass
class CompleteHierarchicalClustering(InitialClustering):
    sea_level_anomaly_data: xarray.Dataset
    number_of_clusters: list[int]
    distance_function: callable
    out_dir: str
    resolution: float = None

    def start_initial_clustering(self):
        """
        Start hierarchical clustering given a 5-degree grid with values for sea level anomaly
        Save each clustering that has a size in k
        :return:
        """
        logger.info(f"Start hierarchical clustering")
        logger.info(f"Calculating distances")
        distances, number_of_clusters, clusters = self.precalculate_distances()
        # save the distances for a given resolution

        logger.info(f"Distances calculated")
        all_clusters = self.clustering(distances, clusters)
        self.resolution = float(self.sea_level_anomaly_data.latitude[1]) - float(
            self.sea_level_anomaly_data.latitude[0])
        for clustering in all_clusters.values():
            # save as a netcdf file
            cluster_data = numpy.zeros(
                (self.sea_level_anomaly_data.latitude.size, self.sea_level_anomaly_data.longitude.size))
            cluster_number = 0

            for cluster in clustering.values():
                for grid_point in cluster.grid_points:
                    lat_idx = np.where(self.sea_level_anomaly_data.latitude.values == grid_point.latitude)[0][0]
                    long_idx = np.where(self.sea_level_anomaly_data.longitude.values == grid_point.longitude)[0][0]
                    cluster_data[lat_idx, long_idx] = cluster_number
                cluster_number += 1
            cluster_data = xarray.DataArray(cluster_data, dims=["latitude", "longitude"])
            cluster_data = cluster_data.assign_coords(latitude=self.sea_level_anomaly_data.latitude,
                                                      longitude=self.sea_level_anomaly_data.longitude)
            cluster_data.to_netcdf(f"{self.out_dir}/clustering_{len(clustering.values())}.nc")
            clustering_dict_for_plotting = {}
            for cluster in clustering.values():
                clustering_dict_for_plotting[cluster.id] = []
                for grid_point in cluster.grid_points:
                    clustering_dict_for_plotting[cluster.id].append(
                        (float(grid_point.latitude), float(grid_point.longitude)))
            plotting.plot_clustering_without_preassigned_colors(clustering_dict_for_plotting, self.out_dir,
                                                                self.resolution,
                                                                f"clustering_{len(clustering.values())}")
            # plot individual clusters for each k
            current_out_dir = f"{self.out_dir}/{len(clustering.values())}"
            if not os.path.exists(current_out_dir):
                os.makedirs(current_out_dir)
            for cluster_id, grid_points in clustering_dict_for_plotting.items():
                name = f"cluster_{cluster_id}"
                current_cluster = {cluster_id: grid_points}
                plotting.plot_clustering_without_preassigned_colors(current_cluster, current_out_dir, self.resolution,
                                                                    name)
        logger.info(f"Clustering done")

    def clustering(self, distances: np.ndarray, clusters: dict[int, Cluster]) -> dict[int, dict[int, Cluster]]:
        """
        Perform hierarchical clustering
        :param clusters:
        :param distances:
        :return:
        """
        all_clusters = {}
        total_number_of_clusters = len(clusters)
        logger.info(f"Starting with {total_number_of_clusters} clusters")
        while total_number_of_clusters > min(self.number_of_clusters):
            if total_number_of_clusters % 500 == 0:
                logger.info(f"Number of clusters: {total_number_of_clusters}")
            # find smallest difference in distances, ignore NaN values
            row_idx, col_idx = np.unravel_index(np.nanargmin(distances), distances.shape)
            # print(f"Row index: {row_idx}, col index: {col_idx}")
            # merge the two clusters
            try:
                cluster1 = clusters[int(row_idx)]
                cluster2 = clusters[int(col_idx)]
            except KeyError:
                logger.error(f"Key error: {row_idx}, {col_idx}")
                logger.error(f"Number of clusters: {total_number_of_clusters}")
                exit()
            # remove the two clusters from the dict of clusters use the smaller id as the id of the merged cluster
            new_cluster = Cluster(id=min(cluster1.id, cluster2.id),
                                  grid_points=cluster1.grid_points + cluster2.grid_points)
            del clusters[cluster1.id]
            del clusters[cluster2.id]
            # set diff between the two old clusters to nan
            distances[cluster1.id, cluster2.id] = np.nan
            distances[cluster2.id, cluster1.id] = np.nan
            clusters[new_cluster.id] = new_cluster
            # 1) calculate distance between merged cluster and all other clusters.
            # 2) take the two rows of the old clusters and calculate the distances of the new cluster to all other
            # clusters by using the values from the old clusters and weighting them with the number of grid points in
            # each cluster.
            # 3) update the distances matrix, update the number of clusters, update the clusters dict, check if the number of
            # clusters is in k, if yes, save the clustering
            current_row = distances[cluster1.id, :]
            for i in range(current_row.size):
                if i == cluster1.id or i == cluster2.id:
                    continue
                if np.isnan(distances[cluster1.id, i]):
                    continue
                distance_current_cluster1 = distances[cluster1.id, i]
                distance_current_cluster2 = distances[cluster2.id, i]
                number_of_grid_points_cluster1 = len(cluster1.grid_points)
                number_of_grid_points_cluster2 = len(cluster2.grid_points)
                number_of_grid_points_current_cluster = number_of_grid_points_cluster1 + number_of_grid_points_cluster2
                new_distance = (
                                       distance_current_cluster1 * number_of_grid_points_cluster1 +
                                       distance_current_cluster2 * number_of_grid_points_cluster2) / number_of_grid_points_current_cluster
                # delete old distances
                distances[cluster1.id, i] = np.nan
                distances[i, cluster1.id] = np.nan
                distances[cluster2.id, i] = np.nan
                distances[i, cluster2.id] = np.nan
                # include new distance
                distances[new_cluster.id, i] = new_distance
                distances[i, new_cluster.id] = new_distance
            total_number_of_clusters -= 1
            if total_number_of_clusters in self.number_of_clusters:
                all_clusters[total_number_of_clusters] = clusters.copy()

        return all_clusters

    def precalculate_distances(self) -> tuple[np.ndarray, int, dict[int, Cluster]]:
        """
        Calculate the distances between each pair of grid points.
        This function does the following steps:
        1. Create a Cluster for each grid point in the sea level anomaly data.
        2. Calculate the distances between each pair of clusters.
        3. Save the distances in a matrix with the cluster ids as indices.

        :return: A tuple of the distance matrix, the number of clusters, and a dictionary of clusters with their ids as keys.
        """
        # in the beginning, each cluster is a single grid point
        logger.info(f"precalculating distances with {self.distance_function.__name__}")
        clusters = {}
        grid_points = []
        cluster_id = 0
        for i in tqdm(range(self.sea_level_anomaly_data.latitude.size), desc="creating clusters"):
            for j in range(self.sea_level_anomaly_data.longitude.size):
                if self.sea_level_anomaly_data["sla"][:, i, j].isnull().values.any():
                    continue
                current_grid_point = GridPoint(id=uuid.uuid4(), latitude=self.sea_level_anomaly_data.latitude.values[i],
                                               longitude=self.sea_level_anomaly_data.longitude.values[j],
                                               timeseries=self.sea_level_anomaly_data.sla.values[:, i, j])
                grid_points.append(current_grid_point)
                current_cluster = Cluster(id=cluster_id, grid_points=[current_grid_point])
                cluster_id += 1
                clusters[current_cluster.id] = current_cluster
        print(f"len grid points: {len(grid_points)}")

        # calculate distances between all pairs of clusters
        # save the distances in matrix form - the cluster ids are the indices
        distances = np.full((cluster_id, cluster_id), np.nan, dtype=np.float32)
        # set of tuples of the form (cluster1, cluster2)
        cluster_pairs = [(x, y) for i, x in enumerate(clusters.values()) for y in list(clusters.values())[i + 1:]]
        logger.info(f"Calculating distances between {len(cluster_pairs)} pairs of clusters")
        distances = self.distances_between_all_pairs(cluster_pairs, distances)
        return distances, cluster_id, clusters

    def distances_between_all_pairs(self, cluster_pairs: list[tuple[Cluster, Cluster]],
                                    distances: np.ndarray) -> np.ndarray:
        """
        Calculate the distances between all pairs of clusters.

        :param cluster_pairs: Pairs of clusters to calculate the distance between. Each pair is a tuple of two Cluster objects.
        :param distances: Matrix of distances between the clusters. The indices of the matrix are the cluster ids.
        :return: The distance matrix.
        """

        # # calculate distances between all pairs of clusters in parallel
        results = Parallel(n_jobs=-2, backend='multiprocessing', verbose=1)(
            delayed(calculate_distance)(*args, self.distance_function) for args in cluster_pairs)

        for i, j, distance in results:
            distances[i, j] = distance
            distances[j, i] = distance

        return distances


def calculate_distance(cluster1: Cluster, cluster2: Cluster, distance_function: callable):
    """
    Calculate the distance between two clusters
    :param cluster1:
    :param cluster2:
    :param distance_function:
    :return:
    """
    grid_point1 = cluster1.grid_points[0]
    grid_point2 = cluster2.grid_points[0]
    lat1 = grid_point1.latitude
    long1 = grid_point1.longitude
    timeseries1 = grid_point1.timeseries
    lat2 = grid_point2.latitude
    long2 = grid_point2.longitude
    timeseries2 = grid_point2.timeseries
    current_distance = distance_function(lat1, long1, timeseries1, lat2, long2,
                                         timeseries2)
    return cluster1.id, cluster2.id, current_distance

# def save_clustering(clusters: {int: Cluster}, number_of_clusters: int, out_dir: str):
#     """
#     Save the clustering
#     :param out_dir:
#     :param clusters:
#     :param number_of_clusters:
#     :return:
#     """
#     # plot and save clustering
#     # create gdf
#     colors = plotting.random_color_generator(number_of_clusters + 1)
#     cluster_ids = []
#     counter = 0
#     grid_point_area = 2
#     polygons = []
#     for cluster in clusters.values():
#         # create a polygon from all grid points in the current cluster
#         cluster_squares = []
#         cluster_ids.append(counter)
#         counter += 1
#
#         for grid_point in cluster.grid_points:
#             square = shapely.Polygon([
#                 (grid_point.longitude + grid_point_area, grid_point.latitude + grid_point_area),
#                 (grid_point.longitude + grid_point_area, grid_point.latitude - grid_point_area),
#                 (grid_point.longitude - grid_point_area, grid_point.latitude - grid_point_area),
#                 (grid_point.longitude - grid_point_area, grid_point.latitude + grid_point_area)
#             ])
#             cluster_squares.append(square)
#
#         # Merge all squares in the cluster using unary_union
#         merged_polygon = unary_union(cluster_squares)
#         polygons.append(merged_polygon)
#
#     # Create a GeoDataFrame with the merged polygons and colors
#     cluster_gdf = geopandas.GeoDataFrame(
#         {'cluster_id': cluster_ids, 'color': colors, 'geometry': polygons}
#         # ,crs="EPSG:4326"  # WGS 84 coordinate system
#     )
#     # save clustering
#     plotting.plot_regions(geopandas.read_file("../data/ne_10m_land/ne_10m_land.shp"),
#                           f"{out_dir}", cluster_gdf,
#                           f"clustering_{number_of_clusters}")
