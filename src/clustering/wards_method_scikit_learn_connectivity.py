from dataclasses import dataclass

import numpy
import xarray
from scipy.sparse import lil_matrix
from sklearn.cluster import AgglomerativeClustering

from src import plotting
from src.clustering.cluster_entities.initial_clustering import InitialClustering


@dataclass
class WardsMethodConnected(InitialClustering):
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
        # nan mask for the data array
        nan_mask = numpy.isnan(self.data_array).any(axis=0)
        # build feature matrix and connectivity matrix
        # scikit-learn expects an adjacency matrix, where if entry (i,j) is non-zero, then i and j are connected
        # the indices i and j have to match the indices of the feature matrix
        time, indices_x, indices_y = self.data_array.shape
        counter = 0
        index_to_grid_points = {}  # here we use the index and not the lat/lon coordinates
        grid_points_to_index = {}
        potential_neighbors_for_grid_point = {}
        feature_matrix = numpy.zeros((indices_x * indices_y, time))
        for i in range(indices_x):
            for j in range(indices_y):
                if nan_mask[i, j]:
                    # if the data point is nan, skip it
                    continue
                index_to_grid_points[counter] = (i, j)
                grid_points_to_index[(i, j)] = counter
                potential_neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1), (i - 1, j - 1), (i + 1, j + 1),
                                       (i - 1, j + 1), (i + 1, j - 1)]
                potential_neighbors_for_grid_point[(i, j)] = potential_neighbors
                time_series_data = self.data_array[:, i, j]
                feature_matrix[counter, :] = time_series_data
                counter += 1
        # remove everything after countre from feature_matrix
        feature_matrix = feature_matrix[:counter, :]

        adjacency_matrix = lil_matrix((counter, counter), dtype=numpy.int8)

        for grid_point in index_to_grid_points.values():
            potential_neighbors = potential_neighbors_for_grid_point[grid_point]
            for neighbor in potential_neighbors:
                if 0 <= neighbor[0] < indices_x and 0 <= neighbor[1] < indices_y:
                    if not nan_mask[neighbor[0], neighbor[1]]:
                        # if the neighbor is not nan, add the connection
                        index1 = grid_points_to_index[grid_point]
                        index2 = grid_points_to_index[neighbor]
                        adjacency_matrix[index1, index2] = 1
                        adjacency_matrix[index2, index1] = 1

        adjacency_matrix = adjacency_matrix.tocsr()  # convert to CSR format for efficiency

        for k in self.number_of_clusters:
            # use scikit-learn's Ward's method for clustering
            model = AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=adjacency_matrix)
            labels = model.fit_predict(feature_matrix)
            # clusters are the unique labels
            clusters = numpy.unique(labels)
            cluster_dict_with_grid_points = {cluster: [] for cluster in clusters}
            cluster_dict_with_lat_lon = {cluster: [] for cluster in clusters}
            for index, label in enumerate(labels):
                grid_point = index_to_grid_points[index]
                lat = self.min_lat + grid_point[0] * self.resolution
                lon = self.min_lon + grid_point[1] * self.resolution
                cluster_dict_with_grid_points[label].append(grid_point)
                cluster_dict_with_lat_lon[label].append((lat, lon))

            plotting.plot_clustering_without_preassigned_colors(cluster_dict_with_lat_lon, self.out_dir,
                                                                self.resolution,
                                                                f"clustering_{k}")
            # save clustering result as xarray dataset
            clustering_array = numpy.full((indices_x, indices_y), numpy.nan)
            for cluster, grid_points in cluster_dict_with_grid_points.items():
                for grid_point in grid_points:
                    clustering_array[grid_point[0], grid_point[1]] = cluster
            clustering_dataset = xarray.Dataset(
                {
                    "clustering": (["latitude", "longitude"], clustering_array)
                },
                coords={
                    "latitude": self.sea_level_anomaly_data.latitude,
                    "longitude": self.sea_level_anomaly_data.longitude
                }
            )
            clustering_dataset.to_netcdf(f"{self.out_dir}/clustering_{k}.nc")
