import math
import unittest

import numpy as np

from src.distance import (
    euclidean_distance,
    distance_for_wards_method,
    spatio_temporal_distance_function,
    subspace_timeseries_distance_calculation,
)


class TestEuclideanDistance(unittest.TestCase):
    def test_known_distance(self):
        # 3-4-5 right triangle
        self.assertAlmostEqual(euclidean_distance(0, 0, [0, 0], 0, 0, [3, 4]), 5.0)

    def test_identical_series_is_zero(self):
        self.assertAlmostEqual(euclidean_distance(0, 0, [1, 2, 3], 0, 0, [1, 2, 3]), 0.0)


class TestWardsDistance(unittest.TestCase):
    def test_squared_error(self):
        # sum of squared differences: 3^2 + 4^2 = 25
        self.assertAlmostEqual(distance_for_wards_method([0, 0], [3, 4]), 25.0)

    def test_identical_series_is_zero(self):
        self.assertAlmostEqual(distance_for_wards_method([1, 2, 3], [1, 2, 3]), 0.0)


class TestSpatioTemporalDistance(unittest.TestCase):
    def test_identical_point_and_series_is_zero(self):
        # same location (distance term = 1) and perfectly correlated (r = 1) -> 1 - 1*1 = 0
        ts = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(spatio_temporal_distance_function(0, 0, ts, 0, 0, ts), 0.0, places=10)

    def test_anti_correlated_at_same_location_is_two(self):
        # same location (distance term = 1), r = -1 -> 1 - 1*(-1) = 2
        ts1 = [1.0, 2.0, 3.0, 4.0]
        ts2 = [4.0, 3.0, 2.0, 1.0]
        self.assertAlmostEqual(spatio_temporal_distance_function(0, 0, ts1, 0, 0, ts2), 2.0, places=10)

    def test_positively_correlated_distance_grows_with_separation(self):
        # for r > 0, the exponential spatial term decays with distance, so dissimilarity increases
        ts1 = [1.0, 2.0, 3.0, 4.0]
        ts2 = [1.0, 2.0, 3.0, 5.0]  # strongly positively correlated with ts1
        near = spatio_temporal_distance_function(0, 0, ts1, 0, 1, ts2)
        far = spatio_temporal_distance_function(0, 0, ts1, 0, 30, ts2)
        self.assertLess(near, far)

    def test_haversine_component_for_one_degree_at_equator(self):
        # back out the spatial term: with r = 1, result = 1 - exp(-d / (2 a^2)),
        # so d = -2 a^2 * ln(1 - result). One degree of longitude at the equator ~ 111 km.
        ts = [1.0, 2.0, 3.0, 4.0]
        a = math.sqrt(-(1500 / math.log(0.5)))
        result = spatio_temporal_distance_function(0, 0, ts, 0, 1, ts)
        recovered_distance = -2 * a ** 2 * math.log(1 - result)
        self.assertAlmostEqual(recovered_distance, 111.19, delta=1.0)


class TestSubspaceTimeseriesDistance(unittest.TestCase):
    def test_vector_in_subspace_has_zero_distance(self):
        subspace = np.array([[1.0, 0.0, 0.0]])  # span of e1
        mean = np.array([0.0, 0.0, 0.0])
        current = np.array([5.0, 0.0, 0.0])  # lies in the subspace
        self.assertAlmostEqual(subspace_timeseries_distance_calculation([], current, mean, subspace), 0.0)

    def test_orthogonal_vector_distance_equals_squared_norm(self):
        subspace = np.array([[1.0, 0.0, 0.0]])  # span of e1
        mean = np.array([0.0, 0.0, 0.0])
        current = np.array([0.0, 3.0, 4.0])  # orthogonal to the subspace
        # residual is the whole vector -> 3^2 + 4^2 = 25
        self.assertAlmostEqual(subspace_timeseries_distance_calculation([], current, mean, subspace), 25.0)

    def test_mean_is_subtracted_before_projection(self):
        subspace = np.array([[0.0, 1.0, 0.0]])  # span of e2
        mean = np.array([5.0, 0.0, 0.0])
        current = np.array([5.0, 1.0, 1.0])
        # centered = [0, 1, 1], projection onto e2 = [0, 1, 0], residual = [0, 0, 1] -> 1
        self.assertAlmostEqual(subspace_timeseries_distance_calculation([], current, mean, subspace), 1.0)

    def test_distance_is_appended_to_all_distances(self):
        subspace = np.array([[1.0, 0.0, 0.0]])
        mean = np.array([0.0, 0.0, 0.0])
        current = np.array([0.0, 3.0, 4.0])
        all_distances = []
        distance = subspace_timeseries_distance_calculation(all_distances, current, mean, subspace)
        self.assertEqual(len(all_distances), 1)
        self.assertAlmostEqual(all_distances[0], distance)


if __name__ == "__main__":
    unittest.main()
