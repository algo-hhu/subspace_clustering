from unittest import TestCase

import haversine
import numpy as np

from src.spherical_gauss_filter import SphericalGaussFilter


class Test(TestCase):
    def test_wraparound_offset(self):
        latitudes = np.arange(start=-90, stop=90, step=0.25)
        longitudes = np.arange(start=-180, stop=180, step=0.25)
        print(longitudes)
        half_width = 500
        spherical_gauss_filter = SphericalGaussFilter(latitudes, longitudes, half_width)
        haversine_distance = np.round(haversine.haversine((0, 170), (0, -170)), 2)
        lon1_index = np.where(longitudes == 170)
        lon2_index = np.where(longitudes == (-170))
        print(lon1_index, lon2_index)
        lat_index = np.where(latitudes == 0)
        distance = spherical_gauss_filter.get_lon_distance(lon1_index, lon2_index, lat_index)
        print(distance)
        assert distance == haversine_distance, f"Expected {haversine_distance}, got {distance}"

    ### print(get_lon_distance(-170, 170, lon_offsets, grid_cell_size))  # Should be 20 degrees
###print(get_lon_distance(-90, 90, lon_offsets, grid_cell_size))    # Should be 180 degrees
### print(get_lon_distance(179, -179, lon_offsets, grid_cell_size))  # Should be 2 degrees (wraparound)
