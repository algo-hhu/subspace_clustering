import unittest

import numpy as np

from src.clustering.subspace_clustering import compare_distances_to_subspaces


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
