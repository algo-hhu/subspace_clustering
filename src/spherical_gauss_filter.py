import math
import pickle

import haversine
import numpy as np
import xarray
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm


class SphericalGaussFilter:
    def __init__(self, lat: np.ndarray, lon: np.ndarray, cut_off: int):
        """
        Initialize gauss filter with given coordinate system
        :param lat: latitude values
        :param lon: longitude values
        :param cut_off: half width of the Gaussian filter in km
        """
        self.R = 6371  # Radius of the Earth in km
        self.lat = lat
        self.lon = lon
        # create lat/lon meshgrid, create mask where sea level anomaly is nan, filter, transform to list of tuples
        self.lat_lon_grid = None
        self.cut_off = cut_off
        self.grid_cell_size = 360 / len(self.lon)
        self.distances = {}

    def create_valid_data_mask(self, ds, variable_name='sla'):
        # Create a 2D boolean mask indicating grid cells that have
        # at least one valid data point across all time steps
        valid_mask = ~np.isnan(ds[variable_name]).any(dim='time')

        # Convert to a 2D array with original grid structure
        # (True where there's valid data at any time, False where always NaN)
        valid_mask_array = valid_mask.values

        # Create an array of lat-lon coordinates with NaNs where no valid data exists
        lat_2d, lon_2d = np.meshgrid(ds.latitude.values, ds.longitude.values, indexing='ij')

        # Make copies to avoid modifying the original arrays
        lat_mask = lat_2d.copy()
        lon_mask = lon_2d.copy()

        # Set coordinates to NaN where the mask is False
        lat_mask[~valid_mask_array] = np.nan
        lon_mask[~valid_mask_array] = np.nan

        return lat_mask, lon_mask

    # For convenient access to both coordinates in a single structure
    def create_latlon_grid(self, ds, variable_name='sla'):
        lat_mask, lon_mask = self.create_valid_data_mask(ds, variable_name)

        # Create a structured 2D grid with both coordinates
        # This will be a 2D array where each cell contains a (lat,lon) tuple
        # Cells will be (np.nan, np.nan) where no valid data exists
        shape = lat_mask.shape
        latlon_grid = np.zeros(shape, dtype=[('lat', float), ('lon', float)])
        latlon_grid['lat'] = lat_mask
        latlon_grid['lon'] = lon_mask
        nan_mask = np.isnan(latlon_grid['lat'])
        logger.info(f"Number of valid grid cells: {np.sum(~nan_mask)}")
        logger.info(f"Number of invalid grid cells: {np.sum(nan_mask)}")
        return latlon_grid

    def distances_between_lat_lon_pairs(self, triple):
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
        candidate_coords = [(self.lat[i], self.lon[j]) for i, j in candidate_indices if
                            (not np.isnan(self.lat[i]) and not np.isnan(self.lon[j])) and (
                                    self.lat[i] != lat1 or self.lon[j] != lon1)]
        current_point = [(lat1, lon1) for _ in range(len(candidate_coords))]
        distances = haversine.haversine_vector(current_point, candidate_coords)
        lat_lon_dist = [[(lat1, lon1), candidate_coords[i], distances[i]] for i in range(len(candidate_coords)) if
                        distances[i] < self.cut_off]
        # # For the candidates, calculate actual distances
        # for idx in candidate_indices:
        #     lat2, lon2 = self.lat_lon_grid[idx[0], idx[1]]
        #     if np.isnan(lat2) or np.isnan(lon2):
        #         continue
        #     # Skip the point itself
        #     if lat1 == lat2 and lon1 == lon2:
        #         continue
        #
        #     # Calculate the great circle distance
        #     distance = np.round(self.haversine((lat1, lon1, lat2, lon2)))
        #     if distance < self.half_width:
        #         lat_lon_dist.extend([((lat1, lon1), (lat2, lon2), distance)])
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

    def precompute_grid_distances(self, sea_level_anomaly_data: xarray.Dataset):
        """
        Precompute the distances between all points that are within a 500 km radius of each other
        Store the calculated distances in a dictionary {(lat, lon): {(neighbor_lat, neighbor_lon): distance}}
        Latitude ranges from -90 to 90, longitude from -180 to 180
        Use the Haversine formula to calculate the distance between two points on the sphere
        :return:
        """
        logger.info("Precomputing distances")
        max_dist = self.cut_off
        distances = {}
        max_distance_in_degrees = np.ceil(self.cut_off / 111)
        self.lat_lon_grid = self.create_latlon_grid(sea_level_anomaly_data)
        # remove lat/lon that do not have any values in the dataset
        # lat_lon_tuples = [(i, j, max_distance_in_degrees) for (i, j) in self.lat_lon_grid if
        #                   not np.isnan(i) and not np.isnan(j)]
        # find pairs for which the approximate distance is less than the half width, to avoid calculating too many distances
        # for lat only need to check up to np.ceil(half_width / 111) degrees north and south, for lon up to np.ceil(half_width / 111) * cos(lat) east and west
        # for lon we additionally need to check for wrap around
        logger.info("Selecting eligible latitude longitude pairs and calculating distances")
        # TODO: process in chunks to avoid memory issues
        for dimension1 in tqdm(self.lat_lon_grid):
            current_lat_lon_pairs = [(i, j, max_distance_in_degrees) for (i, j) in dimension1 if
                                     not np.isnan(i) and not np.isnan(j)]
            if not current_lat_lon_pairs:
                continue
            current_results = (Parallel(n_jobs=-2)(
                delayed(self.distances_between_lat_lon_pairs)(triple) for triple in current_lat_lon_pairs))
            # results.extend(process_map(self.distances_between_lat_lon_pairs, current_lat_lon_pairs, chunksize=1,
            #                            max_workers=int(cpu_count() / 2)))
            current_distances = {}
            for element in current_results:
                for (lat1, lon1), (lat2, lon2), distance in element:
                    if distance < max_dist:
                        if (lat1, lon1) not in current_distances.keys():
                            current_distances[(lat1, lon1)] = {}
                        current_distances[(lat1, lon1)][(lat2, lon2)] = distance

            # save to disk and delete from memory
            file_path = f"../output/distances_{dimension1[0][0]}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(current_distances, f)
                f.close()
                del current_distances
        self.distances = distances

    def filter(self, data):
        """
        Apply a spherical Gaussian filter to the data.
        Data has the dimensions (time, lat, lon)
        For each point determine all points that are within half_width km of the point and calculate their distances
        Then filter the data at this point using the weights determined by the distances, normalize the weights and apply the filter
        Filter for every time step using the distances, then select next point
        :param data:
        :return:
        """
        logger.info("Applying Gaussian filter")
        filtered_data = data.copy()
        max_dist_in_degrees = np.ceil(self.cut_off / 111)
        self.lat_lon_grid = self.create_latlon_grid(data)
        logger.info("Begin filtering")
        for lat in range(data.latitude.shape[0]):
            for lon in range(data.longitude.shape[0]):
                distances = self.distances_between_lat_lon_pairs(
                    (data.latitude[lat].item(), data.longitude[lon].item(), max_dist_in_degrees))
                if not distances:
                    continue
                print(f"distances {distances}")
                exit()
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
        sigma_latitude = self.cut_off / km_per_degree_latitude / grid_cell_size
        sigma_longitude = self.cut_off / km_per_degree_longitude / grid_cell_size
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
