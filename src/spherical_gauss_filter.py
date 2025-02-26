import math
import time

import haversine
import numpy as np
import xarray
from loguru import logger
from tqdm import tqdm


def filter_at_point(time_step, data_per_point, weights):
    """
    Filter data at a given point
    :param weights:
    :param data_per_point:
    :param time_step:
    :return:
    """
    values = np.array([data_per_point[i][time_step] for i in range(len(data_per_point))])
    values = np.nan_to_num(values)  # Replace NaNs with 0
    # apply filter
    new_point_value = np.dot(values, weights)
    return new_point_value


class SphericalGaussFilter:
    def __init__(self, lat: np.ndarray, lon: np.ndarray, half_width: int):
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
        self.grid_cell_size = 360 / len(self.lon)
        self.distances = {}
        self.half_width = half_width
        self.sigma = half_width / 1.178
        self.cut_off = 3 * self.sigma

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
        candidate_coords = [(float(self.lat[i]), float(self.lon[j])) for i, j in candidate_indices if
                            (not np.isnan(self.lat[i]) and not np.isnan(self.lon[j])) and (
                                    self.lat[i] != lat1 or self.lon[j] != lon1)]
        current_point = [(lat1, lon1) for _ in range(len(candidate_coords))]
        distances = haversine.haversine_vector(current_point, candidate_coords)
        lat_lon_dist = [[(lat1, lon1), candidate_coords[i], distances[i]] for i in range(len(candidate_coords)) if
                        distances[i] < self.cut_off]
        return lat_lon_dist

    def filter(self, data: xarray.Dataset):
        """
        Apply a spherical Gaussian filter to the data.
        Data has the dimensions (time, lat, lon)
        For each point determine all points that are within half_width km of the point and calculate their distances
        Then filter the data at this point using the weights determined by the distances, normalize the weights and apply the filter.
        Filter for every time step using the distances, then select next point
        :param data:
        :return:
        """
        logger.info("Applying Gaussian filter")
        filtered_data = data.copy()
        max_dist_in_degrees = np.ceil(self.cut_off / 111)
        self.lat_lon_grid = self.create_latlon_grid(data)
        logger.info("Begin filtering")
        for lat in self.lat:
            for lon in self.lon:
                time1 = time.time()
                if np.isnan(data["sla"].sel(latitude=lat, longitude=lon).values).any():
                    continue
                distances = self.distances_between_lat_lon_pairs(
                    (lat, lon, max_dist_in_degrees))
                if not distances:
                    continue
                filtered_data = self.filter_all_time_steps_at_point(data, filtered_data, lat, lon, distances)
                del distances
                logger.info(f"Time taken to filter one point: {time.time() - time1}")
                exit()
        return filtered_data

    def filter_all_time_steps_at_point(self, data: xarray.Dataset, filtered_data: xarray.Dataset, lat: float,
                                       lon: float, distances: list):
        """
        Filter all time steps at a given point
        :param data:
        :param filtered_data:
        :param lat:
        :param lon:
        :param distances:
        :return:
        """
        neighbor_weights = {(lat, lon): 1}
        for _, (lat2, lon2), distance in distances:
            current_weight = math.exp(-distance ** 2 / (2 * self.sigma ** 2))
            neighbor_weights[(lat2, lon2)] = current_weight
        # normalize weights
        total_weight = sum([weight for weight in neighbor_weights.values()])
        neighbor_weights = {key: value / total_weight for key, value in neighbor_weights.items()}
        data_per_point = {}
        for key in neighbor_weights.keys():
            data_per_point[key] = data["sla"].sel(latitude=key[0], longitude=key[1]).values
        # turn data_per_point and neighbor_weights into numpy arrays
        weights = np.array(list(neighbor_weights.values()))
        values = np.array([data_per_point[key] for key in neighbor_weights.keys()])
        for time_step in tqdm(range(len(data.time.values))):
            new_point_data = filter_at_point(time_step, values, weights)
            # assign new point data to filtered data
            current_time_step = data.time.values[time_step]
            filtered_data["sla"].loc[dict(time=current_time_step, latitude=lat, longitude=lon)] = new_point_data
        return filtered_data
