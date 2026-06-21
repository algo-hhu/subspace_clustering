import unittest

import numpy as np

from src.clustering.subspace_clustering import compare_distances_to_subspaces, determine_subspace_per_cluster


class TestCompareDistancesToSubspaces(unittest.TestCase):

    # def test_basic_functionality(self):
    #     v = np.array([1.0, 2.0]).reshape(-1, 1)  # shape (2,1)

    #     # Subspace A: aligned with vector
    #     dir_vector = v / np.linalg.norm(v)  # unit vector
    #     subspace_a = dir_vector.reshape(-1, 1)  # shape (2, 1)

    #     # Subspace B: orthogonal
    #     orth_vector = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    #     subspace_b = orth_vector.reshape(-1, 1)  # shape (2, 1)

    #     subspaces = {
    #         0: (subspace_a, np.array([0.0, 0.0])),
    #         1: (subspace_b, np.array([0.0, 0.0]))
    #     }
    #     avg_dists = {0: 0.0, 1: 0.0}
    #     distances, closest_cluster, best_distance = compare_distances_to_subspaces(avg_dists, v, subspaces)

    #     self.assertEqual(len(distances), 2)
    #     self.assertTrue(distances[0] < distances[1])
    #     self.assertEqual(closest_cluster, 0)

    def test_multiple_subspaces(self):
        subspace1 = np.array([
            [1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
            [0, 1, 0, 0],
            [-1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]
        ]).T  # Make them columns by transposing
        subspace1, _ = np.linalg.qr(subspace1)  # Orthonormalize the vectors
        subspace1 = subspace1.T
        subspace2 = np.array([
            [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)],
            [0, 1, 0, 0],
            [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]
        ]).T
        subspace2, _ = np.linalg.qr(subspace2)
        subspace2 = subspace2.T
        subspace3 = np.array([
            [1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0]
        ]).T
        subspace3, _ = np.linalg.qr(subspace3)
        subspace3 = subspace3.T
        subspace4 = np.array([
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
        ]).T
        subspace4, _ = np.linalg.qr(subspace4)
        subspace4 = subspace4.T
        # Vector to compare to the subspaces
        v = np.array([4.0, 1.0, 3.0, 5.0])
        subspaces = {0: (subspace1, np.array([0.0, 0.0, 0.0, 0.0])), 1: (subspace2, np.array([0.0, 0.0, 0.0, 0.0])),
                     2: (subspace3, np.array([0.0, 0.0, 0.0, 0.0])), 3: (subspace4, np.array([0.0, 0.0, 0.0, 0.0]))}
        avg_dists = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        distances, closest_cluster, best_distance = compare_distances_to_subspaces(avg_dists, v, subspaces)
        self.assertEqual(len(distances), 4)
        self.assertEqual(closest_cluster, 3)


class TestDetermineSubspacePerCluster(unittest.TestCase):

    def test_returns_none_when_too_few_points(self):
        data = np.random.rand(4, 2, 2)
        # a single grid point cannot span a 5-dimensional subspace
        self.assertIsNone(determine_subspace_per_cluster([(0, 0)], data, 5))

    def test_collinear_cluster_is_captured_by_one_component(self):
        # all grid-point time series are scalar multiples of a common base vector, so after centering
        # they lie on a single line -> the first principal component explains ~100% of the variance
        base = np.array([1.0, 2.0, 3.0, 4.0])
        data = np.zeros((4, 2, 2))
        data[:, 0, 0] = 1.0 * base
        data[:, 0, 1] = 2.0 * base
        data[:, 1, 0] = 3.0 * base
        data[:, 1, 1] = 5.0 * base
        grid_points = [(0, 0), (0, 1), (1, 0), (1, 1)]

        components, mean, explained_variance = determine_subspace_per_cluster(grid_points, data, 1)

        self.assertEqual(components.shape, (1, 4))
        self.assertEqual(mean.shape, (4,))
        self.assertAlmostEqual(explained_variance, 1.0, places=5)

    def test_raises_runtime_error_when_pca_cannot_fit(self):
        # all-NaN time series are dropped, leaving an empty data matrix that PCA cannot fit;
        # this must surface as a RuntimeError (not a bare exit) per the library-error contract
        data = np.full((4, 2, 2), np.nan)
        grid_points = [(0, 0), (0, 1), (1, 0)]
        with self.assertRaises(RuntimeError):
            determine_subspace_per_cluster(grid_points, data, 1)
