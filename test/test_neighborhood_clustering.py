from unittest import TestCase

import numpy as np

from src.clustering import neighborhood_clustering
from src.clustering.neighborhood_clustering import clustering
from src.distance import test_distance_function


class Test(TestCase):
    def test_clustering(self):
        sea_level_anomaly_data = np.array([
            [[1, 1], [1, 1]],  # Time step 1
            [[2, 2], [2, 2]],  # Time step 2
            [[3, 3], [3, 3]]  # Time step 3
        ])

        neighborhood_clustering.DISTANCE_FUNCTION = test_distance_function
        lat_long_to_idx = {(0, 0): (0, 0), (0, 1): (0, 1), (1, 0): (1, 0), (1, 1): (1, 1)}
        k = [1]
        clustering_results = {0: [(0, 0)], 1: [(0, 1)], 2: [(1, 0)], 3: [(1, 1)]}
        neighbors = {0: {1, 2, 3}, 1: {0, 2, 3}, 2: {0, 1, 3}, 3: {0, 1, 2}}
        distances = {}
        for cluster1 in clustering_results.keys():
            for cluster2 in clustering_results.keys():
                if cluster1 == cluster2:
                    continue
                lat1, long1 = clustering_results[cluster1][0]
                lat2, long2 = clustering_results[cluster2][0]
                time_series1 = sea_level_anomaly_data[lat_long_to_idx[(lat1, long1)]]
                time_series2 = sea_level_anomaly_data[lat_long_to_idx[(lat2, long2)]]
                distance = test_distance_function(lat1, long1, time_series1, lat2, long2, time_series2)
                distances[cluster1, cluster2] = distance
                distances[cluster2, cluster1] = distance
        neighborhood_clustering.MIN_LATITUDE = 0
        neighborhood_clustering.MIN_LONGITUDE = 0
        neighborhood_clustering.RESOLUTION = 1
        clustering_result = clustering(clustering_results, sea_level_anomaly_data, k, neighbors,
                                       distances, test_distance_function)
        assert len(
            clustering_result.keys()) == 1, f"The number of clusters is incorrect, expected 1, got {len(clustering_result.keys())}"

    def test_clustering2(self):
        sea_level_anomaly_data = np.array([
            [[1, 1], [1, 1]],  # Time step 1
            [[2, 2], [2, 2]],  # Time step 2
            [[3, 3], [3, 3]]  # Time step 3
        ])

        neighborhood_clustering.DISTANCE_FUNCTION = test_distance_function
        lat_long_to_idx = {(0, 0): (0, 0), (0, 1): (0, 1), (1, 0): (1, 0), (1, 1): (1, 1)}
        k = [1]
        clustering_results = {0: [(0, 0)], 1: [(0, 1)], 2: [(1, 0)], 3: [(1, 1)]}
        neighbors = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
        distances = {}
        for cluster1 in clustering_results.keys():
            for cluster2 in neighbors[cluster1]:
                if cluster1 == cluster2:
                    continue
                lat1, long1 = clustering_results[cluster1][0]
                lat2, long2 = clustering_results[cluster2][0]
                time_series1 = sea_level_anomaly_data[lat_long_to_idx[(lat1, long1)]]
                time_series2 = sea_level_anomaly_data[lat_long_to_idx[(lat2, long2)]]
                distance = test_distance_function(lat1, long1, time_series1, lat2, long2, time_series2)
                distances[cluster1, cluster2] = distance
                distances[cluster2, cluster1] = distance
        neighborhood_clustering.MIN_LATITUDE = 0
        neighborhood_clustering.MIN_LONGITUDE = 0
        neighborhood_clustering.RESOLUTION = 1
        clustering_result = clustering(clustering_results, sea_level_anomaly_data, k, neighbors,
                                       distances, test_distance_function)
        assert len(
            clustering_result.keys()) == 1, f"The number of clusters is incorrect, expected 1, got {len(clustering_result.keys())}"

    def test_index_to_lat_lon(self):
        lat = 89.875
        lon = 179.875
        id_x = 719
        id_y = 1439
        lat_min = -89.875
        lon_min = -179.875
        lat_result, lon_result = neighborhood_clustering.index_to_lat_lon(id_x, id_y, lat_min, lon_min, 0.25)
        assert lat_result == lat, f"Expected {lat}, got {lat_result}"
        assert lon_result == lon, f"Expected {lon}, got {lon_result}"

    def test_index_to_lat_lon2(self):
        lat = -89.875
        lon = -179.875
        id_x = 0
        id_y = 0
        lat_min = -89.875
        lon_min = -179.875
        lat_result, lon_result = neighborhood_clustering.index_to_lat_lon(id_x, id_y, lat_min, lon_min, 0.25)
        assert lat_result == lat, f"Expected {lat}, got {lat_result}"
        assert lon_result == lon, f"Expected {lon}, got {lon_result}"

    def test_index_to_lat_lon3(self):
        lat = 0.125
        lon = 0.125
        id_x = 360
        id_y = 720
        lat_min = -89.875
        lon_min = -179.875
        lat_result, lon_result = neighborhood_clustering.index_to_lat_lon(id_x, id_y, lat_min, lon_min, 0.25)
        assert lat_result == lat, f"Expected {lat}, got {lat_result}"
        assert lon_result == lon, f"Expected {lon}, got {lon_result}"

    def test_lat_lon_to_index(self):
        lat = 89.875
        lon = 179.875
        id_x = 719
        id_y = 1439
        lat_min = -89.875
        lon_min = -179.875
        id_x_result, id_y_result = neighborhood_clustering.lat_lon_to_index(lat, lon, lat_min, lon_min, 0.25)
        assert id_x_result == id_x, f"Expected {id_x}, got {id_x_result}"
        assert id_y_result == id_y, f"Expected {id_y}, got {id_y_result}"

    def test_lat_lon_to_index2(self):
        lat = -89.875
        lon = -179.875
        id_x = 0
        id_y = 0
        lat_min = -89.875
        lon_min = -179.875
        id_x_result, id_y_result = neighborhood_clustering.lat_lon_to_index(lat, lon, lat_min, lon_min, 0.25)
        assert id_x_result == id_x, f"Expected {id_x}, got {id_x_result}"
        assert id_y_result == id_y, f"Expected {id_y}, got {id_y_result}"

    def test_lat_lon_to_index3(self):
        lat = 0.125
        lon = 0.125
        id_x = 360
        id_y = 720
        lat_min = -89.875
        lon_min = -179.875
        id_x_result, id_y_result = neighborhood_clustering.lat_lon_to_index(lat, lon, lat_min, lon_min, 0.25)
        assert id_x_result == id_x, f"Expected {id_x}, got {id_x_result}"
        assert id_y_result == id_y, f"Expected {id_y}, got {id_y_result}"
