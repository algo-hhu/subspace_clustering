from dataclasses import dataclass

import numpy
import numpy as np
import xarray
from sklearn.cluster import KMeans

from src import plotting, helper
from src.clustering import subspace_clustering
from src.clustering.cluster_entities.initial_clustering import InitialClustering
from src.clustering.subspace_clustering import create_cluster_map


@dataclass
class KMeansClustering(InitialClustering):
    sea_level_anomaly_data: xarray.Dataset
    number_of_clusters: list[int]
    distance_function: callable
    out_dir: str
    data_array: numpy.ndarray = None
    min_lat: float = None
    min_lon: float = None
    resolution: float = None

    def start_initial_clustering(self) -> None:
        """
        Start the K-Means clustering process
        :return:
        """
        self.min_lat = self.sea_level_anomaly_data.latitude.min().values
        self.min_lon = self.sea_level_anomaly_data.longitude.min().values
        self.resolution = self.sea_level_anomaly_data.latitude.values[1] - self.sea_level_anomaly_data.latitude.values[
            0]
        self.data_array = self.sea_level_anomaly_data["sla"].values
        # find grid points with NaN values
        # nan_mask = sea_level_anomaly_data["sla"].isnull().values
        nan_mask = numpy.isnan(self.data_array).any(axis=0)
        number_of_grid_points_with_valid_data = numpy.sum(~nan_mask)
        # perform k-means clustering on the time-series data
        time_series_data = numpy.full((self.data_array.shape[0], number_of_grid_points_with_valid_data), numpy.nan)
        index_to_grid_point = {}
        index_to_lat_lon = {}
        counter = 0
        lat_size = self.sea_level_anomaly_data.latitude.size
        lon_size = self.sea_level_anomaly_data.longitude.size
        for idx in range(lat_size):
            for idy in range(lon_size):
                current_time_series_data = self.data_array[:, idx, idy]
                if np.isnan(current_time_series_data).any():
                    continue
                else:
                    time_series_data[:, counter] = current_time_series_data
                    index_to_grid_point[counter] = (idx, idy)
                    index_to_lat_lon[counter] = helper.index_to_lat_lon(idx, idy, self.min_lat, self.min_lon,
                                                                        self.resolution)
                    counter += 1
        for cluster_count in self.number_of_clusters:
            kmeans = KMeans(n_clusters=cluster_count, random_state=0)
            labels = kmeans.fit_predict(time_series_data.T)  # Transpose, because KMeans expects (n_samples, n_features)
            clustering_data_array = np.full(nan_mask.shape, numpy.nan)
            cluster_id_to_lat_lon = {}
            cluster_to_grid_point_dict = {}
            for i, label in enumerate(labels):
                idx, idy = index_to_grid_point[i]
                clustering_data_array[idx, idy] = float(label)
                try:
                    cluster_id_to_lat_lon[float(label)].append(index_to_lat_lon[i])
                    cluster_to_grid_point_dict[float(label)].append((idx, idy))
                except KeyError:
                    cluster_id_to_lat_lon[float(label)] = [index_to_lat_lon[i]]
                    cluster_to_grid_point_dict[float(label)] = [(idx, idy)]
            print(len(cluster_to_grid_point_dict))
            plotting.plot_clustering_without_preassigned_colors(cluster_id_to_lat_lon, self.out_dir, self.resolution,
                                                                f"clustering_before_reestablishing_connectivity_"
                                                                f"{cluster_count}")
            clustering_data = xarray.Dataset(
                {
                    "__xarray_dataarray_variable__": (["latitude", "longitude"], clustering_data_array)
                },
                coords={
                    "latitude": self.sea_level_anomaly_data.latitude,
                    "longitude": self.sea_level_anomaly_data.longitude
                }
            )

            # save clustering data to netCDF file
            # make clustering connected using subspace_clustering.reestablish_connectivity
            subspaces, explained_variance_per_cluster = subspace_clustering.calculate_subspaces_for_clusters(
                cluster_to_grid_point_dict, 1, self.data_array)

            if subspaces.keys() != cluster_to_grid_point_dict.keys():
                print(sorted(subspaces.keys()))
                print(sorted(cluster_to_grid_point_dict.keys()))
            else:
                print("Subspaces and cluster_to_grid_point_dict keys match.")
            cluster_map = create_cluster_map(clustering_data_array, cluster_to_grid_point_dict)
            cluster_id_to_color = plotting.assign_color_to_cluster(cluster_to_grid_point_dict, self.number_of_clusters)
            cluster_id_to_grid_point_id_reconnected = subspace_clustering.reestablish_connectivity(
                self.sea_level_anomaly_data,
                cluster_id_to_lat_lon,
                subspaces, 1, self.out_dir,
                cluster_id_to_color,
                cluster_count)

            cluster_id_to_lat_lon_reconnected = subspace_clustering.convert_idx_idy_to_lat_lon(
                cluster_id_to_grid_point_id_reconnected, self.min_lat, self.min_lon, self.resolution)
            # make a new xarray dataset with the clustering data
            final_clustering_data_array = np.full(
                (self.sea_level_anomaly_data.latitude.size, self.sea_level_anomaly_data.longitude.size), np.nan)
            for cluster_id, grid_point_id in cluster_id_to_grid_point_id_reconnected.items():
                for idx, idy in grid_point_id:
                    final_clustering_data_array[idx, idy] = cluster_id
            clustering_data = xarray.Dataset(
                {
                    "__xarray_dataarray_variable__": (["latitude", "longitude"], final_clustering_data_array)
                },
                coords={
                    "latitude": self.sea_level_anomaly_data.latitude,
                    "longitude": self.sea_level_anomaly_data.longitude
                }
            )

            clustering_data.to_netcdf(f"{self.out_dir}/clustering_{cluster_count}.nc")
            plotting.plot_clustering_without_preassigned_colors(cluster_id_to_lat_lon_reconnected, self.out_dir,
                                                                self.resolution,
                                                                f"clustering_{cluster_count}")
            plotting.plot_xarray_dataset_on_map(clustering_data, self.out_dir, f"saved_clustering{cluster_count}")
