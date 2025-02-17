import math

import numpy as np
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


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
        self.distances = self.precompute_grid_distances()

    def select_lat_lon_pairs(self, lat1, lon1, max_distance_in_degrees):
        """
        Select latitude longitude pairs that are within the half-width-distance of each other
        :param max_distance_in_degrees:
        :param lon1:
        :param lat1:
        :return:
        """
        lat_lon_pairs = []
        for lat2 in self.lat:
            for lon2 in self.lon:
                if lat1 == lat2 and lon1 == lon2:
                    continue
                if np.abs(lat1 - lat2) > max_distance_in_degrees:
                    continue
                if np.min([np.abs(lon1 - lon2), 360 - np.abs(lon1 - lon2)]) > max_distance_in_degrees * np.cos(
                        np.deg2rad(lat1)):
                    continue
                else:
                    lat_lon_pairs.append(((lat1, lon1), (lat2, lon2)))
        return lat_lon_pairs

    # def get_lon_distance(self, lon1_index, lon2_index, lat_index):
    #     """
    #     Get the distance between two longitudes
    #     :param lon2_index:
    #     :param lon1_index:
    #     :param lat_index:
    #     :param lon2:
    #     :return:
    #     """
    #     # compute shortest distance based on precalculated base_lon_distances and cos_factors
    #     # Compute the angular difference in degrees
    #     lon_diff_deg = np.abs(self.lon[lon1_index] - self.lon[lon2_index])
    #     print(f"lon diff degree {lon_diff_deg}")
    #
    #     # Ensure the shortest path (handles wraparound, e.g., 170° to -170° should be 20°)
    #     lon_diff_deg = min(lon_diff_deg, 360 - lon_diff_deg)
    #     print(f"lon diff degree {lon_diff_deg}")
    #
    #     # Convert to radians
    #     lon_diff_rad = np.deg2rad(lon_diff_deg)
    #     distance = np.round((lon_diff_rad * self.cos_factors[lat_index] * self.R), 2)
    #
    #     return distance

    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float):
        """
        Calculate the great circle distance between two points using the haversine formula
        :param lon2:
        :param lat2:
        :param lon1:
        :param lat1:
        :return:
        """
        # Convert latitude and longitude from degrees to radians
        lat_radians_1, lon_radians_1, lat_radians_2, lon_radians_2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat_radians_2 - lat_radians_1
        dlon = lon_radians_2 - lon_radians_1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_radians_1) * np.cos(lat_radians_2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = self.R * c
        return ((lat1, lon1), (lat2, lon2), distance)

    def precompute_grid_distances(self):
        """
        Precompute the distances between all points that are within a 500 km radius of each other
        Store the calculated distances in a dictionary {(lat, lon): {(neighbor_lat, neighbor_lon): distance}}
        Latitude ranges from -90 to 90, longitude from -180 to 180
        Use the Haversine formula to calculate the distance between two points on the sphere
        :return:
        """
        logger.info("Precomputing distances")
        max_dist = self.half_width
        distances = {}
        lat_lon_tuples = [(i, j) for i in self.lat for j in self.lon]
        # find pairs for which the approximate distance is less than the half width, to avoid calculating too many distances
        # for lat only need to check up to np.ceil(half_width / 111) degrees north and south, for lon up to np.ceil(half_width / 111) * cos(lat) east and west
        # for lon we additionally need to check for wrap around
        # TODO: Fix progressbar
        logger.info("Selecting eligible latitude longitude pairs")
        max_distance_in_degrees = np.ceil(self.half_width / 111)
        with tqdm_joblib(tqdm(desc="Processing", total=len(lat_lon_tuples))):
            results = Parallel(n_jobs=-2)(delayed(self.select_lat_lon_pairs)
                                          (lat, lon, max_distance_in_degrees) for lat in
                                          self.lat for lon in self.lon)
        lat_lon_pairs = [pair for sublist in results for pair in sublist]
        logger.info(f"Calculating distances for {len(lat_lon_pairs)} pairs of lat lon tuples")

        # calculate these distances in parallel using joblib
        results = Parallel(n_jobs=-2)(
            self.haversine(lat1, lon1, lat2, lon2) for (lat1, lon1), (lat2, lon2) in lat_lon_pairs)
        for (lat1, lon1), (lat2, lon2), distance in results:
            if distance < max_dist:
                if (lat1, lon1) not in distances:
                    distances[(lat1, lon1)] = {}
                distances[(lat1, lon1)][(lat2, lon2)] = distance
        return distances

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
