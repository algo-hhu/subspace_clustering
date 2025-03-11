from unittest import TestCase

import numpy as np

from src.clustering import neighborhood_clustering
from src.clustering.neighborhood_clustering import clustering


class Test(TestCase):
    def test_clustering(self):
        sea_level_anomaly_data = np.array([
            [[1, 1], [1, 1]],  # Time step 1
            [[2, 2], [2, 2]],  # Time step 2
            [[3, 3], [3, 3]]  # Time step 3
        ])

        def distance_function(lat1, long1, time_series1, lat2, long2, time_series2):
            return abs(sum(time_series1) - sum(time_series2))

        neighborhood_clustering.DISTANCE_FUNCTION = distance_function
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
                distance = distance_function(lat1, long1, time_series1, lat2, long2, time_series2)
                distances[cluster1, cluster2] = distance
                distances[cluster2, cluster1] = distance
        clustering_result = clustering(clustering_results, sea_level_anomaly_data, k, neighbors,
                                       distances,
                                       lat_long_to_idx, distance_function)
        assert len(
            clustering_result.keys()) == 1, f"The number of clusters is incorrect, expected 1, got {len(clustering_result.keys())}"

    def test_clustering2(self):
        sea_level_anomaly_data = np.array([
            [[1, 1], [1, 1]],  # Time step 1
            [[2, 2], [2, 2]],  # Time step 2
            [[3, 3], [3, 3]]  # Time step 3
        ])

        def distance_function(lat1, long1, time_series1, lat2, long2, time_series2):
            return abs(sum(time_series1) - sum(time_series2))

        neighborhood_clustering.DISTANCE_FUNCTION = distance_function
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
                distance = distance_function(lat1, long1, time_series1, lat2, long2, time_series2)
                distances[cluster1, cluster2] = distance
                distances[cluster2, cluster1] = distance
        clustering_result = clustering(clustering_results, sea_level_anomaly_data, k, neighbors,
                                       distances,
                                       lat_long_to_idx, distance_function)
        assert len(
            clustering_result.keys()) == 1, f"The number of clusters is incorrect, expected 1, got {len(clustering_result.keys())}"
