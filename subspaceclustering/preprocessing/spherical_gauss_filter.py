import math

import haversine
import numpy as np
import xarray
from joblib import Parallel, delayed
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
        candidate_coords = np.array([(float(self.lat[i]), float(self.lon[j])) for i, j in candidate_indices if
                            (not np.isnan(self.lat[i]) and not np.isnan(self.lon[j])) and (
                                    self.lat[i] != lat1 or self.lon[j] != lon1)])
        current_point = np.array([(lat1, lon1) for _ in range(len(candidate_coords))])
        distances = haversine.haversine_vector(current_point, candidate_coords)
        lat_lon_dist = [[(lat1, lon1), candidate_coords[i], distances[i]] for i in range(len(candidate_coords)) if
                        distances[i] < self.cut_off]
        return lat_lon_dist

    def parallelized_filter(self, data: xarray.Dataset):
        """
        Apply the Gaussian filter to each data point in parallel
        :param data:
        :return:
        """
        max_dist_in_degrees = np.ceil(self.cut_off / 111)  # determine the farthest distance in degrees for cut off
        sla_array = data["sla"].values  # extract the data array, shape: (time, lat, lon)
        # create lat/lon index mapping
        lat_to_index = {lat: i for i, lat in enumerate(self.lat)}
        lon_to_index = {lon: i for i, lon in enumerate(self.lon)}
        idx_to_lat = {i: lat for i, lat in enumerate(self.lat)}
        idx_to_lon = {i: lon for i, lon in enumerate(self.lon)}
        # extract grid points with valid data
        non_nan_mask = ~np.isnan(sla_array).all(axis=0)
        valid_grid_points = list(map(tuple, np.argwhere(non_nan_mask)))
        logger.info(f"Processing {len(valid_grid_points)} valid grid points.")
        # create list of args for parallel processing
        args_list = [(idx_to_lat[grid_point[0]], idx_to_lon[grid_point[1]], lat_to_index, lon_to_index,
                      sla_array, max_dist_in_degrees) for grid_point
                     in valid_grid_points]
        # calculate filtered data for each grid point in parallel
        results = Parallel(n_jobs=-2, verbose=1)(
            delayed(self.call_filtering)(*args) for args in args_list)
        filtered_data = self.process_filtering_results(data, results, sla_array)
        return filtered_data

    def process_filtering_results(self, data, results, sla_array):
        """
        Process the results of the filtering
        :param data:
        :param results:
        :param sla_array:
        :return:
        """
        # filter out None results
        results = [r for r in results if r is not None]
        # write results of filtering to new array
        filtered_data = data.copy()
        filtered_data_array = np.zeros_like(sla_array)
        for id_x, id_y, new_data in results:
            filtered_data_array[:, id_x, id_y] = new_data
        filtered_da = xarray.DataArray(
            filtered_data_array,
            dims=data["sla"].dims,
            coords=data["sla"].coords,
            attrs=data["sla"].attrs
        )
        filtered_data["sla"] = filtered_da
        # put NaN values back
        mask = data.sla.isnull().any(axis=0)
        # apply mask using `.where()`, replacing with NaNs
        filtered_data["sla"] = filtered_data["sla"].where(~mask, np.nan)
        return filtered_data

    def call_filtering(self, lat, lon, lat_to_index, lon_to_index, sla_array, max_dist_in_degrees):
        """
        Call the filtering function for a given lat/lon pair
        :param lat:
        :param lon:
        :param lat_to_index:
        :param lon_to_index:
        :param sla_array:
        :param max_dist_in_degrees:
        :return:
        """
        # calculate distances between all points within the cut-off distance of the current point
        distances = self.distances_between_lat_lon_pairs(
            (lat, lon, max_dist_in_degrees))
        if not distances:
            return None
        id_x, id_y, new_data = self.filter_all_time_steps_at_point(sla_array, lat, lon,
                                                                   distances, lat_to_index, lon_to_index)
        return (id_x, id_y, new_data)

    def filter_all_time_steps_at_point(self, sla_array: np.array, lat: float,
                                       lon: float, distances: list, lat_to_index: dict, lon_to_index: dict):
        """
        Filter all time steps at a given point
        :param lon_to_index:
        :param lat_to_index:
        :param sla_array:
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
        # filter data
        data_values = np.stack(
            [sla_array[:, lat_to_index[key[0]], lon_to_index[key[1]]] for key in neighbor_weights.keys()])
        for i, values in enumerate(data_values):
            if np.isnan(values).all():
                # if the all values are NaN, we do not consider it so set the weight to 0
                neighbor_weights[list(neighbor_weights.keys())[i]] = 0

        # replace nan in data_values with 0
        data_values = np.nan_to_num(data_values)
        weights = np.array(list(neighbor_weights.values()))
        # take weights-vector, multiply with each row of data_values, then sum over first axis, resulting in a 1D array with length of time steps 
        new_data = np.round(np.einsum('i,ij->j', weights, data_values), 4)
        return lat_to_index[lat], lon_to_index[lon], new_data
