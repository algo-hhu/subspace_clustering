import tempfile
import unittest

import numpy as np
import xarray as xr

from src.helper import adjust_resolution, extract_clusters_from_xarray_dataset, save_clustering


class TestAdjustResolution(unittest.TestCase):
    def _dataset_with_latitudes(self, latitudes):
        return xr.Dataset(coords={"latitude": latitudes})

    def test_finer_resolution_raises_value_error(self):
        # data is on a 2 degree grid; requesting a 1 degree grid would be upsampling
        ds = self._dataset_with_latitudes([0.0, 2.0, 4.0])
        with self.assertRaises(ValueError):
            adjust_resolution(1, "/tmp/unused", ds)

    def test_sub_degree_resolution_raises_value_error(self):
        # resolutions below 1 degree are not supported
        ds = self._dataset_with_latitudes([0.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            adjust_resolution(0.8, "/tmp/unused", ds)

    def test_matching_resolution_returns_dataset_unchanged(self):
        ds = self._dataset_with_latitudes([0.0, 2.0, 4.0])
        result = adjust_resolution(2, "/tmp/unused", ds)
        self.assertIs(result, ds)


class TestSaveAndExtractClusteringRoundTrip(unittest.TestCase):
    def _sea_level_dataset(self):
        # 3 time steps on a 2x2 grid at 2-degree resolution, all values finite
        sla = np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2) + 1.0
        return xr.Dataset(
            {"sla": (["time", "latitude", "longitude"], sla)},
            coords={"time": [0, 1, 2], "latitude": [0.0, 2.0], "longitude": [10.0, 12.0]},
        )

    def test_partition_is_recovered_after_save_and_reload(self):
        sea_level = self._sea_level_dataset()
        # partition every grid point (by lat/lon value) into two clusters
        clustering_dict = {
            0: [(0.0, 10.0), (0.0, 12.0)],
            1: [(2.0, 10.0), (2.0, 12.0)],
        }

        with tempfile.TemporaryDirectory() as out_dir:
            save_clustering(clustering_dict, out_dir, sea_level, "clustering_test")
            reloaded = xr.open_dataset(f"{out_dir}/clustering_test.nc")

            _, cluster_to_grid_point_id = extract_clusters_from_xarray_dataset(
                reloaded, min_lat=0.0, min_lon=10.0, resolution=2.0,
                sla_data=sea_level["sla"].values)

        # two clusters, and the grid-point partition matches (compared as sets, since save_clustering
        # renumbers cluster ids 0..k-1); this guards the "__xarray_dataarray_variable__" round trip
        self.assertEqual(len(cluster_to_grid_point_id), 2)
        recovered = {frozenset(points) for points in cluster_to_grid_point_id.values()}
        expected = {frozenset({(0, 0), (0, 1)}), frozenset({(1, 0), (1, 1)})}
        self.assertEqual(recovered, expected)


if __name__ == "__main__":
    unittest.main()
