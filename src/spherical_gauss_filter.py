import numpy as np


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

    def filter(self, data, mask, sigma_lat: float, sigma_lon: float):
        """
        Apply a spherical Gaussian filter to the data
        :param data:
        :param mask:
        :param sigma_lat:
        :param sigma_lon:
        :return:
        """
