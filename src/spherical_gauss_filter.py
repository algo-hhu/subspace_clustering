import math

import numpy as np
from loguru import logger


class SphericalGaussFilter:
    def __init__(self, lat: np.ndarray, lon: np.ndarray, half_width: int):
        """
        Initialize gauss filter with given coordinate system
        :param lat: latitude values
        :param lon: longitude values
        :param half_width: half width of the Gaussian filter in km
        """
        self.R = 6371  # Radius of the Earth in km
        self.lat = lat
        self.lon = lon
        self.half_width = half_width
        self.grid_cell_size = 360 / len(self.lon)
        # Compute possible latitude offsets once
        max_lat_steps = len(lat)
        lat_offsets = np.arange(max_lat_steps + 1) * self.grid_cell_size
        self.lat_distances = self.R * lat_offsets

        # Compute base longitude distances at equator
        max_lon_steps = len(lon)
        lon_offsets = np.arange(max_lon_steps + 1) * self.grid_cell_size

        # Handle wraparound
        lon_offsets = np.minimum(lon_offsets, 360 - lon_offsets)

        # Precompute cosine factors for each latitude
        self.cos_factors = np.cos(np.deg2rad(self.lat))
        self.base_lon_distances = self.R * lon_offsets

    def get_lon_distance(self, lon1_index, lon2_index, lat_index):
        """
        Get the distance between two longitudes
        :param lon2_index:
        :param lon1_index:
        :param lat_index:
        :param lon2:
        :return:
        """
        # compute shortest distance based on precalculated base_lon_distances and cos_factors
        lon_diff = np.abs(self.base_lon_distances[lon1_index] - self.base_lon_distances[lon2_index])
        distance = np.round(lon_diff * self.cos_factors[lat_index], 2)

        return distance

    def precompute_grid_distances(self):
        """
        Precompute the distances between all points in the coordinate system
        Use the fact that the distances between latitudes are always the same, and the distances between longitude only
        have to be computed once for each latitude
        :return:
        """
        logger.info("Precomputing distances")
        # Assume, that the latitudes are equally spaced and the longitudes are equally spaced
        self.lat_distances = self.R * np.deg2rad(self.grid_cell_size)

        # Compute the distances between longitudes for each latitude
        lon_step = np.deg2rad(self.grid_cell_size)
        self.lon_distances = np.zeros(len(self.lat))
        self.lon_distances = self.R * lon_step * np.cos(np.deg2rad(self.lat))  # Vectorized version

        return self.lat_distances, self.lon_distances

    def filter(self, data):
        """
        Apply a spherical Gaussian filter to the data.
        Data has the dimensions (time, lat, lon)
        :param data:
        :return:
        """
        logger.info("Applying Gaussian filter")
        filtered_data = data.copy()
        # parallelize using joblib
        # for time_step in range(data.shape[0]):
        # results = Parallel(n_jobs=-2)(
        #     delayed(self.filter_one_time_step)(data, mask, time_step) for time_step in
        #     enumerate(data["time"].values))
        for time_step in range(data.time.shape[0]):
            filtered_data[time_step] = self.filter_one_time_step(data, time_step)
            exit(0)

    def calculate_sigma_per_latitude(self, latitude: float, grid_cell_size: float):
        """
        Given the latitude of a point, calculate the sigma for the Gaussian filter
        :param grid_cell_size:
        :param latitude:
        :return:
        """
        km_per_degree_latitude = 111
        km_per_degree_longitude = 111 * math.cos(math.radians(latitude))
        sigma_latitude = self.half_width / km_per_degree_latitude / grid_cell_size
        sigma_longitude = self.half_width / km_per_degree_longitude / grid_cell_size
        return sigma_latitude, sigma_longitude

    def filter_one_time_step(self, data, time_step):
        """
        Apply the spherical gauss filter to one time step
        :param data:
        :param time_step:
        :return:
        """
        # sea_level_anomaly_data[feature].isel(time=0)
        current_data = data["sla"].isel(time=time_step).values
        # print(f"current data: {current_data}")
        # print(f"current data shape: {current_data.shape}")
        # print("=====================================")
        current_filtered_data = np.nan * np.ones_like(current_data)
        # # how many nan values are there in the data at the current time step
        # print(f"current time step: {time_step}")
        # print(
        #     f"number of nan values in current time step {data.sla.isel(time=time_step).isnull().sum().item()}")
        # print(f"number of not NaN values in current time step {data.sla.isel(time=time_step).notnull().sum().item()}")
        # print("=====================================")
        # plotting.plot_nan_values(data, time_step)
        invalid_counter = 0
        valid_counter = 0
        for i in range(data.latitude.size):
            for j in range(data.longitude.size):
                lat = data.latitude[i].item()
                lon = data.longitude[j].item()
                if np.isnan(current_data[i, j]):
                    invalid_counter += 1
                    continue
                # compute weights for latitude and longitude
                valid_counter += 1
                if valid_counter % 1000 == 0:
                    print(f"valid counter: {valid_counter}")
                logger.info(f"compute weights")
                lat_diff = np.abs(np.deg2rad(self.lat - self.lat[i]))
                lon_diff = np.abs(np.deg2rad(self.lon - self.lon[j]))
                print(f"lat diff: {lat_diff}")
                print(f"lon diff: {lon_diff}")

                # wrap around the earth
                lon_diff = np.minimum(lon_diff, np.pi - lon_diff)
                print(f"lon diff wrap around: {lon_diff}")

                # compute distances
                logger.info(f"compute distances")
                lat_dist = self.R * lat_diff

                # for k, lat in enumerate(np.deg2rad(self.lat)):
                #     lon_dist[k] = self.R * lon_diff[k] * np.cos(lat)
                # more efficient version:
                lon_dist = self.R * lon_diff * np.cos(np.deg2rad(self.lat))[:, np.newaxis]
                sigma_lat, sigma_lon = self.calculate_sigma_per_latitude(self.lat[i], 360 / len(self.lon))
                # compute weights
                logger.info(f"compute weights")
                lat_weights = np.exp(- (lat_dist ** 2) / (2 * sigma_lat ** 2))
                lon_weights = np.exp(- (lon_dist ** 2) / (2 * sigma_lon ** 2))
                weights = lat_weights * lon_weights
                # weights[current_mask] = 0

                if weights.sum() > 0:
                    weights /= weights.sum()
                    current_filtered_data[i, j] = np.sum(
                        data[time_step] * weights)
                # return a data array with the filtered data
                return current_filtered_data
