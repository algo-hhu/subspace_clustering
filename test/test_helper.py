import unittest

import xarray as xr

from src.helper import adjust_resolution


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


if __name__ == "__main__":
    unittest.main()
