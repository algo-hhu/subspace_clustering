from unittest import TestCase

import numpy
import numpy as np
import xarray

from src import helper
from src.clustering import neighborhood_clustering
from src.clustering.neighborhood_clustering import NeighborhoodClustering
from src.distance import distance_function_test


class Test(TestCase):
    def test_clustering(self):
        sea_level_anomaly_data = np.array([
            [[1, 1], [1, 1]],  # Time step 1
            [[2, 2], [2, 2]],  # Time step 2
            [[3, 3], [3, 3]]  # Time step 3
        ])
        xarray_dataset = xarray.Dataset(data_vars={"sla": (("time", "latitude", "longitude"), sea_level_anomaly_data)},
                                        coords={"time": [1, 2, 3], "latitude": [0, 1], "longitude": [0, 1]})

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
                distance = distance_function_test(lat1, long1, time_series1, lat2, long2, time_series2)
                distances[cluster1, cluster2] = distance
                distances[cluster2, cluster1] = distance
        neighborhood_clustering = NeighborhoodClustering(xarray_dataset, k, distance_function_test, "", sea_level_anomaly_data, 0, 0,1)
        clustering_result = neighborhood_clustering.clustering(clustering_results, neighbors, distances)
        assert len(
            clustering_result.keys()) == 1, f"The number of clusters is incorrect, expected 1, got {len(clustering_result.keys())}"

    def test_clustering2(self):
        sea_level_anomaly_data = np.array([
            [[1, 1], [1, 1]],  # Time step 1
            [[2, 2], [2, 2]],  # Time step 2
            [[3, 3], [3, 3]]  # Time step 3
        ])
        xarray_dataset = xarray.Dataset(data_vars={"sla": (("time", "latitude", "longitude"), sea_level_anomaly_data)},
                                        coords={"time": [1, 2, 3], "latitude": [0, 1], "longitude": [0, 1]})
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
                distance = distance_function_test(lat1, long1, time_series1, lat2, long2, time_series2)
                distances[cluster1, cluster2] = distance
                distances[cluster2, cluster1] = distance
        neighborhood_clustering = NeighborhoodClustering(xarray_dataset, k, distance_function_test, "", sea_level_anomaly_data, 0, 0,1)
        clustering_result = neighborhood_clustering.clustering(clustering_results, neighbors, distances)
        assert len(
            clustering_result.keys()) == 1, f"The number of clusters is incorrect, expected 1, got {len(clustering_result.keys())}"

    def test_index_to_lat_lon(self):
        lat = 89.875
        lon = 179.875
        id_x = 719
        id_y = 1439
        lat_min = -89.875
        lon_min = -179.875
        lat_result, lon_result = helper.index_to_lat_lon(id_x, id_y, lat_min, lon_min, 0.25)
        assert lat_result == lat, f"Expected {lat}, got {lat_result}"
        assert lon_result == lon, f"Expected {lon}, got {lon_result}"

    def test_index_to_lat_lon2(self):
        lat = -89.875
        lon = -179.875
        id_x = 0
        id_y = 0
        lat_min = -89.875
        lon_min = -179.875
        lat_result, lon_result = helper.index_to_lat_lon(id_x, id_y, lat_min, lon_min, 0.25)
        assert lat_result == lat, f"Expected {lat}, got {lat_result}"
        assert lon_result == lon, f"Expected {lon}, got {lon_result}"

    def test_index_to_lat_lon3(self):
        lat = 0.125
        lon = 0.125
        id_x = 360
        id_y = 720
        lat_min = -89.875
        lon_min = -179.875
        lat_result, lon_result = helper.index_to_lat_lon(id_x, id_y, lat_min, lon_min, 0.25)
        assert lat_result == lat, f"Expected {lat}, got {lat_result}"
        assert lon_result == lon, f"Expected {lon}, got {lon_result}"

    def test_lat_lon_to_index(self):
        lat = 89.875
        lon = 179.875
        id_x = 719
        id_y = 1439
        lat_min = -89.875
        lon_min = -179.875
        id_x_result, id_y_result = helper.lat_lon_to_index(lat, lon, lat_min, lon_min, 0.25)
        assert id_x_result == id_x, f"Expected {id_x}, got {id_x_result}"
        assert id_y_result == id_y, f"Expected {id_y}, got {id_y_result}"

    def test_lat_lon_to_index2(self):
        lat = -89.875
        lon = -179.875
        id_x = 0
        id_y = 0
        lat_min = -89.875
        lon_min = -179.875
        id_x_result, id_y_result = helper.lat_lon_to_index(lat, lon, lat_min, lon_min, 0.25)
        assert id_x_result == id_x, f"Expected {id_x}, got {id_x_result}"
        assert id_y_result == id_y, f"Expected {id_y}, got {id_y_result}"

    def test_lat_lon_to_index3(self):
        lat = 0.125
        lon = 0.125
        id_x = 360
        id_y = 720
        lat_min = -89.875
        lon_min = -179.875
        id_x_result, id_y_result = helper.lat_lon_to_index(lat, lon, lat_min, lon_min, 0.25)
        assert id_x_result == id_x, f"Expected {id_x}, got {id_x_result}"
        assert id_y_result == id_y, f"Expected {id_y}, got {id_y_result}"

    # def test_neighbors_across_180(self):
    #     # read data
    #     sea_level_anomaly_data = xarray.open_dataset("../data/sea_level_anomaly_data.nc")
    #     data = sea_level_anomaly_data["sla"].values
    #     distance_function = distance_function_test
    #     lat_lon_to_idx = {(lat, lon): (i, j) for i, lat in enumerate(sea_level_anomaly_data.latitude.values) for j, lon
    #                       in
    #                       enumerate(sea_level_anomaly_data.longitude.values)}
    #     nan_mask = np.array(numpy.isnan(data).any(axis=0))
    #     clusters = {}
    #     counter = 0
    #     for lat in sea_level_anomaly_data.latitude.values:
    #         for lon in sea_level_anomaly_data.longitude.values:
    #             if nan_mask[lat_lon_to_idx[lat, lon]]:
    #                 continue
    #             else:
    #                 clusters[counter] = [(lat, lon)]
    #                 counter += 1
    #     lat_lon_to_clusters = {value[0]: key for key, value in clusters.items()}
    #     matching_items = {
    #         key: value for key, value in lat_lon_to_clusters.items()
    #         if isinstance(key[0], np.floating) and key[1] == 179.875
    #     }
    #     matching_items = {
    #         key: value for key, value in lat_lon_to_clusters.items()
    #         if isinstance(key[0], np.floating) and key[1] == -179.875
    #     }
    #     cluster_180 = lat_lon_to_clusters.get((np.float32(-65.125), np.float32(179.875)))
    #     cluster_neg180 = lat_lon_to_clusters.get((np.float32(-65.125), np.float32(-179.875)))
    #     neighborhood_clustering = NeighborhoodClustering(sea_level_anomaly_data, [10], distance_function, "", data, 0, 0,1)
    #     neighbors, unique_pairs_with_timeseries = neighborhood_clustering.find_neighbors(lat_lon_to_clusters, nan_mask)
    #     assert cluster_180 in neighbors.get(cluster_neg180) and cluster_neg180 in neighbors.get(
    #         cluster_180), f"Expected {cluster_180} to be in neighbors of {cluster_neg180} and vice versa"

    # def test_neighbors_across_180_interp(self):
    #     # read data
    #     sea_level_anomaly_data = xarray.open_dataset("../data/sea_level_anomaly_data.nc")
    #     sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=range(-90, 91, 5),
    #                                                            longitude=range(-180, 180, 5))
    #     data = sea_level_anomaly_data["sla"].values

    #     distance_function = distance_function_test
    #     lat_lon_to_idx = {(lat, lon): (i, j) for i, lat in enumerate(sea_level_anomaly_data.latitude.values) for j, lon
    #                       in
    #                       enumerate(sea_level_anomaly_data.longitude.values)}
    #     nan_mask = np.array(numpy.isnan(data).any(axis=0))
    #     clusters = {}
    #     counter = 0
    #     for lat in sea_level_anomaly_data.latitude.values:
    #         for lon in sea_level_anomaly_data.longitude.values:
    #             if nan_mask[lat_lon_to_idx[lat, lon]]:
    #                 continue
    #             else:
    #                 clusters[counter] = [(lat, lon)]
    #                 counter += 1
    #     lat_lon_to_clusters = {value[0]: key for key, value in clusters.items()}
    #     cluster_180 = lat_lon_to_clusters.get((np.int64(-60), np.int64(175)))
    #     cluster_neg180 = lat_lon_to_clusters.get((np.int64(-60), np.int64(-175)))
    #     neighborhood_clustering = NeighborhoodClustering(sea_level_anomaly_data, [10], distance_function, "", data, sea_level_anomaly_data.latitude.min().values, sea_level_anomaly_data.longitude.min().values, 1)
    #     neighbors, unique_pairs_with_timeseries = neighborhood_clustering.find_neighbors(lat_lon_to_clusters, nan_mask)
    #     assert cluster_180 in neighbors.get(cluster_neg180) and cluster_neg180 in neighbors.get(
    #         cluster_180), f"Expected {cluster_180} to be in neighbors of {cluster_neg180} and vice versa"
