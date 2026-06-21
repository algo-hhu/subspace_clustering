import unittest

import numpy as np

from src.clustering.subspace_clustering import (
    calculate_subspaces_for_clusters,
    compare_distances_to_subspaces,
    convert_idx_idy_to_lat_lon,
    create_cluster_map,
    determine_closest_subspace,
    determine_subspace_per_cluster,
    modify_clustering_with_subspaces,
)


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


class TestCreateClusterMap(unittest.TestCase):

    def test_grid_points_are_labelled_with_their_cluster_id(self):
        cluster_data = np.zeros((3, 3))  # only the shape is used
        assignment = {0: [(0, 0), (0, 1)], 1: [(2, 2)]}

        cluster_map = create_cluster_map(cluster_data, assignment)

        self.assertEqual(cluster_map[0, 0], 0)
        self.assertEqual(cluster_map[0, 1], 0)
        self.assertEqual(cluster_map[2, 2], 1)
        # unassigned grid points stay NaN
        self.assertTrue(np.isnan(cluster_map[1, 1]))


class TestConvertIdxIdyToLatLon(unittest.TestCase):

    def test_indices_map_to_coordinates(self):
        assignment = {0: [(0, 0), (1, 2)]}
        result = convert_idx_idy_to_lat_lon(assignment, min_lat=-10, min_lon=20, resolution=2)
        # lat = min_lat + x*res, lon = min_lon + y*res
        self.assertEqual(result, {0: [(-10, 20), (-8, 24)]})


class TestCalculateSubspacesForClusters(unittest.TestCase):

    def test_clusters_too_small_for_a_subspace_are_skipped(self):
        base = np.array([1.0, 2.0, 3.0, 4.0])
        data = np.zeros((4, 2, 3))
        data[:, 0, 0] = 1.0 * base
        data[:, 0, 1] = 2.0 * base
        data[:, 0, 2] = 3.0 * base
        data[:, 1, 0] = 5.0 * base
        data[:, 1, 1] = base  # the single-point cluster below
        cluster_id_dict = {0: [(0, 0), (0, 1), (0, 2), (1, 0)], 1: [(1, 1)]}

        subspaces, explained_variance = calculate_subspaces_for_clusters(cluster_id_dict, 1, data)

        # cluster 0 has enough points for a 1-d subspace; cluster 1 (one point) is skipped
        self.assertIn(0, subspaces)
        self.assertNotIn(1, subspaces)
        components, mean = subspaces[0]
        self.assertEqual(components.shape, (1, 4))


class TestDetermineClosestSubspace(unittest.TestCase):

    def _orthogonal_subspaces(self):
        # subspace 0 = span(e1), subspace 1 = span(e2)
        return {
            0: (np.array([[1.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0])),
            1: (np.array([[0.0, 1.0, 0.0]]), np.array([0.0, 0.0, 0.0])),
        }

    def _data(self):
        data = np.zeros((3, 1, 2))
        data[:, 0, 0] = [5.0, 0.0, 0.0]  # lies in subspace 0
        data[:, 0, 1] = [0.0, 5.0, 0.0]  # lies in subspace 1
        return data

    def test_points_are_assigned_to_their_own_subspace(self):
        previous = {0: [(0, 0)], 1: [(0, 1)]}
        assignment, change, summed = determine_closest_subspace(self._data(), self._orthogonal_subspaces(), 1, previous)
        self.assertEqual(assignment[0], [(0, 0)])
        self.assertEqual(assignment[1], [(0, 1)])
        self.assertAlmostEqual(summed, 0.0)

    def test_change_flag_is_false_when_assignment_is_stable(self):
        previous = {0: [(0, 0)], 1: [(0, 1)]}
        _, change, _ = determine_closest_subspace(self._data(), self._orthogonal_subspaces(), 1, previous)
        self.assertFalse(change)

    def test_change_flag_is_true_when_assignment_moves(self):
        previous = {0: [(0, 1)], 1: [(0, 0)]}  # swapped vs. the true assignment
        _, change, _ = determine_closest_subspace(self._data(), self._orthogonal_subspaces(), 1, previous)
        self.assertTrue(change)


class TestModifyClusteringWithSubspaces(unittest.TestCase):

    def test_misassigned_point_migrates_to_neighbouring_better_subspace(self):
        # 1x3 grid; the middle point belongs in subspace 1 but starts in cluster 0,
        # and its right neighbour is in cluster 1, so it should migrate there (Alg. 3).
        subspaces = {
            0: (np.array([[1.0, 0.0, 0.0]]), np.array([0.0, 0.0, 0.0])),  # span(e1)
            1: (np.array([[0.0, 1.0, 0.0]]), np.array([0.0, 0.0, 0.0])),  # span(e2)
        }
        sla_data = np.zeros((3, 1, 3))
        sla_data[:, 0, 0] = [5.0, 0.0, 0.0]  # in subspace 0
        sla_data[:, 0, 1] = [0.0, 5.0, 0.0]  # belongs in subspace 1, misassigned to cluster 0
        sla_data[:, 0, 2] = [0.0, 5.0, 0.0]  # in subspace 1
        cluster_data = np.zeros((1, 3))  # only its shape is used
        assignment = {0: [(0, 0), (0, 1)], 1: [(0, 2)]}

        new_assignment, change, summed = modify_clustering_with_subspaces(
            assignment, sla_data, subspaces, cluster_data)

        self.assertTrue(change)
        self.assertIn((0, 1), new_assignment[1])
        self.assertNotIn((0, 1), new_assignment[0])
        self.assertEqual(new_assignment[0], [(0, 0)])
