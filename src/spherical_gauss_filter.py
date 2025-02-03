import numpy as np
from joblib import Parallel, delayed


class SphericalGaussFilter:
    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        """
        Initialize gauss filter with given coordinate system
        :param lat: latitude values
        :param lon: longitude values
        """
        self.R = 6371  # Radius of the Earth in km
        self.lat = lat
        # TODO: consider that the longitude wraps around the earth and 180 and -180 are the same point
        self.lon = lon

        # Precompute the distance matrix
        self.lat_distances, self.lon_distances = self.precompute_distances()

    def precompute_distances(self):
        """
        Precompute the distances between all points in the coordinate system
        Use the fact that the distances between latitudes are always the same, and the distances between longitude only have to be computed once for each latitude
        :return:
        """
        lat_step = np.deg2rad(np.abs(self.lat[1] - self.lat[0]))
        self.lat_distances = self.R * lat_step

        # Compute the distances between longitudes for each latitude
        lon_step = np.deg2rad(np.abs(self.lon[1] - self.lon[0]))
        self.lon_distances = np.zeros(len(self.lat))
        for i, lat in enumerate(np.deg2rad(self.lat)):
            self.lon_distances[i] = self.R * lon_step * np.cos(lat)

        return self.lat_distances, self.lon_distances

    def filter(self, data, provided_mask, sigma_lat: float, sigma_lon: float):
        """
        Apply a spherical Gaussian filter to the data.
        Data has the dimensions (time, lat, lon)
        :param data:
        :param mask:
        :param sigma_lat:
        :param sigma_lon:
        :return:
        """
        if provided_mask is None:
            mask = np.zeros_like(data, dtype=bool)
        else:
            mask = provided_mask
        filtered_data = data.copy()

        # parallelize using joblib
        # for time_step in range(data.shape[0]):
        results = Parallel(n_jobs=-1)(
            delayed(self.filter_one_time_step)(data, mask, sigma_lat, sigma_lon, time_step) for time_step in
            range(data.shape[0]))
        for result in results:
            filtered_data[result[1]] = result[0]

    def filter_one_time_step(self, data, mask, sigma_lat, sigma_lon, time_step):
        current_filtered_data = np.zeros_like(data[time_step])
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                if mask[time_step, i, j]:
                    continue
                # compute weights for latitude and longitude
                lat_diff = np.abs(np.deg2rad(self.lat - self.lat[i]))
                lon_diff = np.abs(np.deg2rad(self.lon - self.lon[j]))

                # wrap around the earth
                lon_diff = np.minimum(lon_diff, np.pi - lon_diff)

                # compute distances
                lat_dist = self.R * lat_diff
                lon_dist = np.zeros_like(lon_diff)

                # for k, lat in enumerate(np.deg2rad(self.lat)):
                #     lon_dist[k] = self.R * lon_diff[k] * np.cos(lat)
                # more efficient version:
                lon_dist = self.R * lon_diff * np.cos(np.deg2rad(self.lat))[:, np.newaxis]

                # compute weights
                lat_weights = np.exp(- (lat_dist ** 2) / (2 * sigma_lat ** 2))
                lon_weights = np.exp(- (lon_dist ** 2) / (2 * sigma_lon ** 2))
                weights = lat_weights * lon_weights
                weights[mask[time_step]] = 0

                if weights.sum() > 0:
                    weights /= weights.sum()
                    current_filtered_data[i, j] = np.sum(data[time_step] * weights)
                else:
                    current_filtered_data[i, j] = np.nan
                return current_filtered_data, time_step
