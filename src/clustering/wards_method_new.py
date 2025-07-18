import os
from dataclasses import dataclass

import numpy
import xarray

from src.clustering.cluster_entities.initial_clustering import InitialClustering


@dataclass
class GridPoint:
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
    neighbors: list[int]


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
        self.nan_mask = numpy.isnan(self.data_array)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        clusters = self.initially_create_clusters()
        # find the neighboring clusters for each cluster
        clusters = self.find_neighbors(clusters)
        # start the clustering process
        clustering_results = self.perform_clustering(clusters)
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
        # iterate over all spatial points in the data array (time, latitude, longitude)
        time, latitudes, longitudes = self.data_array.shape
        for i in range(latitudes):
            for j in range(longitudes):
                if self.nan_mask[:, i, j].any():
                    continue
                time_series = self.data_array[:, i, j]
                grid_point = GridPoint(self.sea_level_anomaly_data.latitude[i],
                                       self.sea_level_anomaly_data.longitude[j], time_series)
                # create a cluster for each point, in the beginning, the centroid is the time series itself
                all_clusters[counter] = Cluster(counter, time_series, [grid_point], 0, [])
                grid_point.cluster_id = counter
                counter += 1
        return all_clusters

    def find_neighbors(self, clusters):
        pass

    def perform_clustering(self, clusters):
        pass

    def plot_and_save_clustering_results(self, clustering_results):
        pass
