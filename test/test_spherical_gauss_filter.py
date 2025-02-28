from unittest import TestCase

import haversine
import numpy as np
import xarray as xr

from src.preprocessing.spherical_gauss_filter import SphericalGaussFilter


class TestSphericalGaussFilter(TestCase):
    def test_haversine(self):
        latitudes = np.arange(start=-90, stop=90, step=0.25)
        longitudes = np.arange(start=-180, stop=180, step=0.25)
        lat1 = 0
        lon1 = 170
        lat2 = 0
        lon2 = -170
        sgf = SphericalGaussFilter(latitudes, longitudes, 500)
        distance = sgf.haversine((lat1, lon1, lat2, lon2))
        haversine_distance = haversine.haversine((lat1, lon1), (lat2, lon2))
        assert np.round(haversine_distance) == np.round(distance), f"Expected {haversine_distance} but got {distance}"

    def test_haversine_different_latitude(self):
        latitudes = np.arange(start=-90, stop=90, step=0.25)
        longitudes = np.arange(start=-180, stop=180, step=0.25)
        lat1 = 30
        lon1 = 170
        lat2 = 50
        lon2 = -160
        sgf = SphericalGaussFilter(latitudes, longitudes, 500)
        distance = sgf.haversine((lat1, lon1, lat2, lon2))
        haversine_distance = haversine.haversine((lat1, lon1), (lat2, lon2))
        assert np.round(haversine_distance) == np.round(distance), f"Expected {haversine_distance} but got {distance}"

    def test_haversine_different_latitude_2(self):
        latitudes = np.arange(start=-90, stop=90, step=0.25)
        longitudes = np.arange(start=-180, stop=180, step=0.25)
        lat1 = -90
        lon1 = 110
        lat2 = 40
        lon2 = -150
        sgf = SphericalGaussFilter(latitudes, longitudes, 500)
        distance = sgf.haversine((lat1, lon1, lat2, lon2))
        haversine_distance = haversine.haversine((lat1, lon1), (lat2, lon2))
        assert np.round(haversine_distance) == np.round(distance), f"Expected {haversine_distance} but got {distance}"

    def test_select_lat_lon_pairs(self):
        # TODO: rewrite this test
        lat_lon_max_dist = [(0.0, 0.0, 5.0), (4.0, 2.0, 5.0), (10.0, 50.0, 5.0), (-4.0, -2.0, 5.0)]
        latitudes = np.array([-4.0, 0.0, 4.0, 10.0])
        longitudes = np.array([-2.0, 0.0, 2.0, 50.0])
        sgf = SphericalGaussFilter(latitudes, longitudes, 500)
        sgf.lat_lon_grid = np.array([[(-4.0, -2.0), (-4.0, 0.0), (-4.0, 2.0), (-4.0, 50.0)], [(0.0, -2.0), (0.0, 0.0),
                                                                                              (0.0, 2.0), (0.0, 50.0)],
                                     [(4.0, -2.0), (4.0, 0.0), (4.0, 2.0), (4.0, 50.0)],
                                     [(10.0, -2.0), (10.0, 0.0), (10.0, 2.0), (10.0, 50.0)]])
        lat_lon_pairs = []
        for lat_lon in lat_lon_max_dist:
            lat_lon_pairs.extend(sgf.distances_between_lat_lon_pairs(lat_lon))
        assert len(lat_lon_pairs) == 16, f"Expected 16 but got {len(lat_lon_pairs)}"

    def test_select_lat_lon_pairs_wrap_around(self):
        lat_lon_tuples = [(0, 179, 5), (0, -179, 5)]
        latitudes = np.arange(start=-90, stop=90, step=0.25)
        longitudes = np.arange(start=-180, stop=180, step=0.25)
        sgf = SphericalGaussFilter(latitudes, longitudes, 500)
        ds = xr.open_dataset("../data/sea_level_anomaly_data.nc")
        sgf.lat_lon_grid = sgf.create_latlon_grid(ds, "sla")
        print(sgf.lat_lon_grid)
        lat_lon_pairs = []
        for lat_lon in lat_lon_tuples:
            lat_lon_pairs.extend(sgf.distances_between_lat_lon_pairs(lat_lon))
        haversine_dist = haversine.haversine((0, 179), (0, -179))
        for lat_lon_pair in lat_lon_pairs:
            (lat1, lon1), (lat2, lon2), distance = lat_lon_pair
            if lat1 == 0 and lon1 == 179 and lat2 == 0 and lon2 == -179:
                assert np.round(distance) == np.round(haversine_dist), f"Expected {haversine_dist} but got {distance}"
