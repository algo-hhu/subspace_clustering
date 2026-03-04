import unittest
from unittest import TestCase


import numpy as np
import xarray as xr

from src.preprocessing.spherical_gauss_filter import SphericalGaussFilter


class TestSphericalGaussFilter(TestCase):

    def test_filter_all_time_steps_at_point_weighted_average_and_indices(self):
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 1.0])
        sgf = SphericalGaussFilter(lat, lon, half_width=500)

        # shape: (time=3, lat=2, lon=2)
        sla_array = np.zeros((3, 2, 2), dtype=float)
        sla_array[:, 0, 0] = np.array([1.0, 2.0, 3.0])  # center
        sla_array[:, 0, 1] = np.array([4.0, 5.0, 6.0])  # neighbor

        lat_to_index = {0.0: 0, 1.0: 1}
        lon_to_index = {0.0: 0, 1.0: 1}

        d = 10.0
        distances = [[(0.0, 0.0), (0.0, 1.0), d]]

        idx_x, idx_y, new_data = sgf.filter_all_time_steps_at_point(
            sla_array, 0.0, 0.0, distances, lat_to_index, lon_to_index
        )

        w_neighbor = np.exp(-(d ** 2) / (2 * sgf.sigma ** 2))
        w_center = 1.0
        w_sum = w_center + w_neighbor
        w_center /= w_sum
        w_neighbor /= w_sum

        expected = np.round(
            w_center * np.array([1.0, 2.0, 3.0]) + w_neighbor * np.array([4.0, 5.0, 6.0]), 4
        )

        self.assertEqual(idx_x, 0)
        self.assertEqual(idx_y, 0)
        np.testing.assert_allclose(new_data, expected, rtol=0, atol=1e-12)

    def test_filter_all_time_steps_at_point_all_nan_neighbor_sets_weight_zero_without_renorm(self):
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 1.0])
        sgf = SphericalGaussFilter(lat, lon, half_width=500)

        # shape: (time=2, lat=2, lon=2)
        sla_array = np.zeros((2, 2, 2), dtype=float)
        sla_array[:, 1, 1] = np.array([10.0, 20.0])      # center
        sla_array[:, 1, 0] = np.array([np.nan, np.nan])  # all-NaN neighbor

        lat_to_index = {0.0: 0, 1.0: 1}
        lon_to_index = {0.0: 0, 1.0: 1}

        d = 50.0
        distances = [[(1.0, 1.0), (1.0, 0.0), d]]

        idx_x, idx_y, new_data = sgf.filter_all_time_steps_at_point(
            sla_array, 1.0, 1.0, distances, lat_to_index, lon_to_index
        )

        # Normalize first, then drop all-NaN neighbor weight (no re-normalization).
        w_neighbor = np.exp(-(d ** 2) / (2 * sgf.sigma ** 2))
        w_center = 1.0 / (1.0 + w_neighbor)
        expected = np.round(w_center * np.array([10.0, 20.0]), 4)

        self.assertEqual(idx_x, 1)
        self.assertEqual(idx_y, 1)
        np.testing.assert_allclose(new_data, expected, rtol=0, atol=1e-12)

    def test_filter_all_time_steps_at_point_partial_nan_neighbor_is_zero_per_timestep(self):
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 1.0])
        sgf = SphericalGaussFilter(lat, lon, half_width=500)

        # shape: (time=3, lat=2, lon=2)
        sla_array = np.zeros((3, 2, 2), dtype=float)
        sla_array[:, 0, 0] = np.array([1.0, 1.0, 1.0])        # center
        sla_array[:, 0, 1] = np.array([np.nan, 2.0, np.nan])  # partial NaN neighbor

        lat_to_index = {0.0: 0, 1.0: 1}
        lon_to_index = {0.0: 0, 1.0: 1}

        d = 1.0
        distances = [[(0.0, 0.0), (0.0, 1.0), d]]

        _, _, new_data = sgf.filter_all_time_steps_at_point(
            sla_array, 0.0, 0.0, distances, lat_to_index, lon_to_index
        )

        w_neighbor = np.exp(-(d ** 2) / (2 * sgf.sigma ** 2))
        w_center = 1.0
        w_sum = w_center + w_neighbor
        w_center /= w_sum
        w_neighbor /= w_sum

        expected = np.round(
            np.array(
                [
                    w_center * 1.0 + w_neighbor * 0.0,  # NaN -> 0
                    w_center * 1.0 + w_neighbor * 2.0,
                    w_center * 1.0 + w_neighbor * 0.0,  # NaN -> 0
                ]
            ),
            4,
        )
        np.testing.assert_allclose(new_data, expected, rtol=0, atol=1e-12)

    def test_filter_all_time_steps_at_point_rounds_to_4_decimals(self):
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 1.0])
        sgf = SphericalGaussFilter(lat, lon, half_width=500)

        sla_array = np.zeros((2, 2, 2), dtype=float)
        sla_array[:, 0, 0] = np.array([0.1234567, 0.7654321])  # center
        sla_array[:, 1, 1] = np.array([1.2345678, 9.8765432])  # neighbor

        lat_to_index = {0.0: 0, 1.0: 1}
        lon_to_index = {0.0: 0, 1.0: 1}

        d = 123.0
        distances = [[(0.0, 0.0), (1.0, 1.0), d]]

        _, _, new_data = sgf.filter_all_time_steps_at_point(
            sla_array, 0.0, 0.0, distances, lat_to_index, lon_to_index
        )

        self.assertTrue(np.array_equal(new_data, np.round(new_data, 4)))


class TestSphericalGaussFilterEdgeCases(TestCase):
    def test_filter_all_time_steps_at_point_empty_distances_returns_center_series(self):
        """
        Edge case: no neighbors supplied.
        Assumption for robustness: return original center series (rounded).
        """
        lat = np.array([0.0])
        lon = np.array([0.0])
        sgf = SphericalGaussFilter(lat, lon, half_width=500)

        sla_array = np.array([[[1.11119]], [[2.22229]], [[3.33339]]], dtype=float)
        lat_to_index = {0.0: 0}
        lon_to_index = {0.0: 0}

        idx_x, idx_y, new_data = sgf.filter_all_time_steps_at_point(
            sla_array, 0.0, 0.0, [], lat_to_index, lon_to_index
        )

        self.assertEqual((idx_x, idx_y), (0, 0))
        np.testing.assert_allclose(new_data, np.round(np.array([1.11119, 2.22229, 3.33339]), 4), rtol=0, atol=1e-12)

    def test_filter_all_time_steps_at_point_zero_half_width_raises_or_handles(self):
        """
        Edge case: half_width=0 => sigma=0.
        Accept either:
        - explicit exception, or
        - finite output with no NaN/Inf.
        """
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 1.0])
        sgf = SphericalGaussFilter(lat, lon, half_width=0)

        sla_array = np.zeros((2, 2, 2), dtype=float)
        sla_array[:, 0, 0] = np.array([1.0, 2.0])
        sla_array[:, 0, 1] = np.array([3.0, 4.0])

        lat_to_index = {0.0: 0, 1.0: 1}
        lon_to_index = {0.0: 0, 1.0: 1}
        distances = [[(0.0, 0.0), (0.0, 1.0), 10.0]]

        try:
            _, _, new_data = sgf.filter_all_time_steps_at_point(
                sla_array, 0.0, 0.0, distances, lat_to_index, lon_to_index
            )
            self.assertTrue(np.isfinite(new_data).all())
        except (ZeroDivisionError, FloatingPointError, ValueError):
            pass

    def test_filter_all_time_steps_at_point_very_large_distance_underflow_safe(self):
        """
        Edge case: very large distance => neighbor weight underflows to ~0.
        """
        lat = np.array([0.0, 1.0])
        lon = np.array([0.0, 1.0])
        sgf = SphericalGaussFilter(lat, lon, half_width=500)

        sla_array = np.zeros((3, 2, 2), dtype=float)
        center = np.array([1.0, 2.0, 3.0])
        sla_array[:, 0, 0] = center
        sla_array[:, 0, 1] = np.array([100.0, 100.0, 100.0])

        lat_to_index = {0.0: 0, 1.0: 1}
        lon_to_index = {0.0: 0, 1.0: 1}
        distances = [[(0.0, 0.0), (0.0, 1.0), 1e9]]

        _, _, new_data = sgf.filter_all_time_steps_at_point(
            sla_array, 0.0, 0.0, distances, lat_to_index, lon_to_index
        )

        np.testing.assert_allclose(new_data, np.round(center, 4), rtol=0, atol=1e-8)