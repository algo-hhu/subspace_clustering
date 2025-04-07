import uuid
from dataclasses import dataclass
from typing import List

import geopandas
import numpy
import numpy as np
import shapely
import xarray
from loguru import logger
from shapely.ops import unary_union
from tqdm import tqdm

import src.distance
from src import plotting


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


def precalculate_distances(sea_level_anomaly_data: xarray.Dataset):
    """
    Calculate the distances between each pair of grid points
    :return:
    """
    # in the beginning each cluster is a single grid point
    clusters = {}
    grid_points = []
    number_of_clusters = 0
    for i in tqdm(range(sea_level_anomaly_data.latitude.size)):
        for j in range(sea_level_anomaly_data.longitude.size):
            if sea_level_anomaly_data["sla"][:, i, j].isnull().values.any():
                continue
            current_grid_point = GridPoint(id=uuid.uuid4(), latitude=sea_level_anomaly_data.latitude.values[i],
                                           longitude=sea_level_anomaly_data.longitude.values[j],
                                           timeseries=sea_level_anomaly_data.sla.values[:, i, j])
            grid_points.append(current_grid_point)
            current_cluster = Cluster(id=number_of_clusters, grid_points=[current_grid_point])
            number_of_clusters += 1
            clusters[current_cluster.id] = current_cluster

    # calculate distances between all pairs of clusters
    # save the distances in matrix form - the cluster ids are the indices
    distances = np.full((number_of_clusters, number_of_clusters), np.nan)
    cluster_pairs = [(x, y) for x in clusters.values() for y in clusters.values() if x != y]
    logger.info(f"Calculating distances between {len(cluster_pairs)} pairs of clusters")
    distances_between_all_pairs(cluster_pairs, distances)
    return distances, number_of_clusters, clusters


def distances_between_all_pairs(cluster_pairs, distances):
    """
    Calculate the distances between all pairs of clusters
    :param cluster_pairs:
    :param distances:
    :return:
    """
    for cluster_pair in cluster_pairs:
        cluster1 = cluster_pair[0]
        cluster2 = cluster_pair[1]
        grid_point1 = cluster1.grid_points[0]
        grid_point2 = cluster2.grid_points[0]
        lat1 = grid_point1.latitude
        long1 = grid_point1.longitude
        timeseries1 = grid_point1.timeseries
        lat2 = grid_point2.latitude
        long2 = grid_point2.longitude
        timeseries2 = grid_point2.timeseries
        current_distance = src.distance.distance_function(lat1, long1, timeseries1, lat2, long2,
                                                          timeseries2)
        distances[cluster1.id, cluster2.id] = current_distance
        distances[cluster2.id, cluster1.id] = current_distance
    return distances


def hierarchical_clustering(distances: np.array, sea_level_anomaly_data: xarray.Dataset,
                            number_of_clusters: int, k: [int], clusters: {int: Cluster}):
    """
    Perform hierarchical clustering
    :param clusters:
    :param k:
    :param number_of_clusters:
    :param distances:
    :param sea_level_anomaly_data:
    :return:
    """
    all_clusters = {}
    logger.info(f"Starting with {number_of_clusters} clusters")
    while number_of_clusters > min(k):
        if number_of_clusters % 500 == 0:
            logger.info(f"Number of clusters: {number_of_clusters}")
        # find smallest difference in distances, ignore NaN values
        row_idx, col_idx = np.unravel_index(np.nanargmin(distances), distances.shape)
        # print(f"Row index: {row_idx}, col index: {col_idx}")
        # merge the two clusters
        try:
            cluster1 = clusters[int(row_idx)]
            cluster2 = clusters[int(col_idx)]
        except KeyError:
            logger.error(f"Key error: {row_idx}, {col_idx}")
            logger.error(f"Number of clusters: {number_of_clusters}")
            exit()
        # remove the two clusters from the dict of clusters use the smaller id as the id of the merged cluster
        new_cluster = Cluster(id=min(cluster1.id, cluster2.id), grid_points=cluster1.grid_points + cluster2.grid_points)
        del clusters[cluster1.id]
        del clusters[cluster2.id]
        # set diff between the two old clusters to nan
        distances[cluster1.id, cluster2.id] = np.nan
        distances[cluster2.id, cluster1.id] = np.nan
        clusters[new_cluster.id] = new_cluster
        # calculate distance between merged cluster and all other clusters
        # take the two rows of the old clusters and calculate the distances of the new cluster to all other clusters by
        # using the values from the old clusters and weighting them with the number of grid points in each cluster
        # update the distances matrix, update the number of clusters, update the clusters dict, check if the number of
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
        number_of_clusters -= 1
        if number_of_clusters in k:
            all_clusters[number_of_clusters] = clusters.copy()

    return all_clusters


def save_clustering(clusters: {int: Cluster}, number_of_clusters: int, out_dir: str):
    """
    Save the clustering
    :param out_dir:
    :param clusters:
    :param number_of_clusters:
    :return:
    """
    # plot and save clustering
    # create gdf
    colors = plotting.random_color_generator(number_of_clusters + 1)
    cluster_ids = []
    counter = 0
    grid_point_area = 2.5
    polygons = []
    for cluster in clusters.values():
        # create a polygon from all grid points in the current cluster
        cluster_squares = []
        cluster_ids.append(counter)
        counter += 1

        for grid_point in cluster.grid_points:
            square = shapely.Polygon([
                (grid_point.longitude + grid_point_area, grid_point.latitude + grid_point_area),
                (grid_point.longitude + grid_point_area, grid_point.latitude - grid_point_area),
                (grid_point.longitude - grid_point_area, grid_point.latitude - grid_point_area),
                (grid_point.longitude - grid_point_area, grid_point.latitude + grid_point_area)
            ])
            cluster_squares.append(square)

        # Merge all squares in the cluster using unary_union
        merged_polygon = unary_union(cluster_squares)
        polygons.append(merged_polygon)
    cluster_gdf = geopandas.GeoDataFrame(
        {'cluster_id': cluster_ids, 'color': colors, 'geometry': polygons}
        # ,crs="EPSG:4326"  # WGS 84 coordinate system
    )
    # save clustering
    plotting.plot_regions(geopandas.read_file("../data/ne_10m_land/ne_10m_land.shp"),
                          f"{out_dir}", cluster_gdf,
                          f"clustering_{number_of_clusters}")


def start_clustering(k, sea_level_anomaly_data: xarray.Dataset, out_dir):
    """
    Start hierarchical clustering given a 5 degree grid with values for sea level anomaly
    Save each clustering that has a size in k
    :return:
    """
    logger.info(f"Start hierarchical clustering")
    logger.info(f"Calculating distances")
    distances, number_of_clusters, clusters = precalculate_distances(sea_level_anomaly_data)
    logger.info(f"Distances calculated")
    all_clusters = hierarchical_clustering(distances, sea_level_anomaly_data, number_of_clusters, k, clusters)
    for clustering in all_clusters.values():
        # save clustering
        # copy xarray dataset
        # save as netcdf file
        cluster_data = numpy.zeros((sea_level_anomaly_data.latitude.size, sea_level_anomaly_data.longitude.size))
        cluster_number = 0

        for cluster in clustering.values():
            for grid_point in cluster.grid_points:
                lat_idx = np.where(sea_level_anomaly_data.latitude.values == grid_point.latitude)[0][0]
                long_idx = np.where(sea_level_anomaly_data.longitude.values == grid_point.longitude)[0][0]
                cluster_data[lat_idx, long_idx] = cluster_number
            cluster_number += 1
        # save clustering as netcdf file
        cluster_data = xarray.DataArray(cluster_data, dims=["latitude", "longitude"])
        cluster_data = cluster_data.assign_coords(latitude=sea_level_anomaly_data.latitude,
                                                  longitude=sea_level_anomaly_data.longitude)
        cluster_data.to_netcdf(f"{out_dir}/clusters_{len(clustering.values())}.nc")
        save_clustering(clustering, len(clustering), out_dir)
    logger.info(f"Clustering done")
