from unittest import TestCase

import haversine
import numpy as np

from src.spherical_gauss_filter import SphericalGaussFilter


class TestSphericalGaussFilter(TestCase):
    def test_haversine(self):
        latitudes = np.arange(start=-90, stop=90, step=0.25)
        longitudes = np.arange(start=-180, stop=180, step=0.25)
        lat1 = 0
        lon1 = 170
        lat2 = 0
        lon2 = -170
        sgf = SphericalGaussFilter(latitudes, longitudes, 500)
        (lat1, lon1), (lat2, lon2), distance = sgf.haversine(lat1, lon1, lat2, lon2)
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
        (lat1, lon1), (lat2, lon2), distance = sgf.haversine(lat1, lon1, lat2, lon2)
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
        (lat1, lon1), (lat2, lon2), distance = sgf.haversine(lat1, lon1, lat2, lon2)
        haversine_distance = haversine.haversine((lat1, lon1), (lat2, lon2))
        assert np.round(haversine_distance) == np.round(distance), f"Expected {haversine_distance} but got {distance}"

    def test_select_lat_lon_pairs(self):
        lat_lon_tuples = [(0, 0), (4, 2), (10, 50), (-4, -2)]
        latitudes = np.arange(start=-90, stop=90, step=0.25)
        longitudes = np.arange(start=-180, stop=180, step=0.25)
        sgf = SphericalGaussFilter(latitudes, longitudes, 500)
        lat_lon_pairs = sgf.select_lat_lon_pairs(lat_lon_tuples)
        assert len(lat_lon_pairs) == 4, f"Expected 4 but got {len(lat_lon_pairs)}"
        assert lat_lon_pairs.__contains__((lat_lon_tuples[0], lat_lon_tuples[
            1])), f"Expected {(lat_lon_tuples[0], lat_lon_tuples[1])} but got {lat_lon_pairs}"
        assert lat_lon_pairs.__contains__((lat_lon_tuples[1], lat_lon_tuples[
            0])), f"Expected {(lat_lon_tuples[1], lat_lon_tuples[0])} but got {lat_lon_pairs}"
        assert lat_lon_pairs.__contains__((lat_lon_tuples[0], lat_lon_tuples[
            3])), f"Expected {(lat_lon_tuples[0], lat_lon_tuples[3])} but got {lat_lon_pairs}"
        assert lat_lon_pairs.__contains__((lat_lon_tuples[3], lat_lon_tuples[
            0])), f"Expected {(lat_lon_tuples[3], lat_lon_tuples[0])} but got {lat_lon_pairs}"

    def test_select_lat_lon_pairs_wrap_around(self):
        lat_lon_tuples = [(0, 189), (0, -189)]
        latitudes = np.arange(start=-90, stop=90, step=0.25)
        longitudes = np.arange(start=-180, stop=180, step=0.25)
        sgf = SphericalGaussFilter(latitudes, longitudes, 500)
        lat_lon_pairs = sgf.select_lat_lon_pairs(lat_lon_tuples)
        assert len(lat_lon_pairs) == 2, f"Expected 2 but got {len(lat_lon_pairs)}"
        assert lat_lon_pairs.__contains__((lat_lon_tuples[0], lat_lon_tuples[
            1])), f"Expected {(lat_lon_tuples[0], lat_lon_tuples[1])} but got {lat_lon_pairs}"
        assert lat_lon_pairs.__contains__((lat_lon_tuples[1], lat_lon_tuples[
            0])), f"Expected {(lat_lon_tuples[1], lat_lon_tuples[0])} but got {lat_lon_pairs}"
