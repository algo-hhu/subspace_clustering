import math
from os import cpu_count

import numpy as np
from loguru import logger
from tqdm.contrib.concurrent import process_map


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
        self.distances = {}

    def select_lat_lon_pairs(self, triple):
        """
        Select latitude longitude pairs that are within the half-width-distance of each other
        :param triple:
        :return:
        """
        lat1, lon1, max_distance_in_degrees = triple
        lat_lon_dist = []
        # Create bounding box
        lat_min = lat1 - max_distance_in_degrees
        lat_max = lat1 + max_distance_in_degrees
        # Calculate longitude threshold adjusted for latitude
        lon_threshold = max_distance_in_degrees * np.cos(np.deg2rad(lat1))
        lon_min = lon1 - lon_threshold
        lon_max = lon1 + lon_threshold
        # Handle wraparound for longitude
        if lon_min < -180:
            lon_min += 360
        if lon_max > 180:
            lon_max -= 360
        # Create mask for points within the bounding box
        if lon_min < lon_max:
            # Normal case
            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)
        else:
            # Case where we cross the date line
            lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
            lon_mask = (self.lon >= lon_min) | (self.lon <= lon_max)

        valid_lat_indices = np.where(lat_mask)[0]
        valid_lon_indices = np.where(lon_mask)[0]
        candidate_indices = np.array(np.meshgrid(valid_lat_indices, valid_lon_indices)).T.reshape(-1, 2)

        # For the candidates, calculate actual distances
        for idx in candidate_indices:
            lat2, lon2 = self.lat[idx[0]], self.lon[idx[1]]
            # Skip the point itself
            if lat1 == lat2 and lon1 == lon2:
                continue

            # Calculate the great circle distance
            distance = np.round(self.haversine((lat1, lon1, lat2, lon2)))
            if distance < self.half_width:
                lat_lon_dist.extend([((lat1, lon1), (lat2, lon2), distance)])
        return lat_lon_dist

    def haversine(self, coordinate_pair):
        """
        Calculate the great circle distance between two points using the haversine formula
        :param coordinate_pair:
        :return:
        """
        lat1, lon1, lat2, lon2 = coordinate_pair
        # Convert latitude and longitude from degrees to radians
        lat_radians_1, lon_radians_1, lat_radians_2, lon_radians_2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat_radians_2 - lat_radians_1
        dlon = lon_radians_2 - lon_radians_1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_radians_1) * np.cos(lat_radians_2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = self.R * c
        return distance

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
        max_distance_in_degrees = np.ceil(self.half_width / 111)
        lat_lon_tuples = [(i, j, max_distance_in_degrees) for i in self.lat for j in self.lon]
        # find pairs for which the approximate distance is less than the half width, to avoid calculating too many distances
        # for lat only need to check up to np.ceil(half_width / 111) degrees north and south, for lon up to np.ceil(half_width / 111) * cos(lat) east and west
        # for lon we additionally need to check for wrap around
        logger.info("Selecting eligible latitude longitude pairs and calculating distances")
        logger.info(f"Number of cpus available / 2: {cpu_count() / 2}")
        # use shared memory for parallelization
        
        results = process_map(self.select_lat_lon_pairs, lat_lon_tuples, chunksize=1, max_workers=int(cpu_count() / 2))
        for (lat1, lon1), (lat2, lon2), distance in results:
            if distance < max_dist:
                if (lat1, lon1) not in distances:
                    distances[(lat1, lon1)] = {}
                distances[(lat1, lon1)][(lat2, lon2)] = distance
        self.distances = distances

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
