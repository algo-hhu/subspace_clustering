import unittest

import numpy as np
import xarray as xr

from src.weighting import apply_weights_to_sea_level_anomaly_data


class TestApplyWeights(unittest.TestCase):
    def _make_dataset(self):
        # sla of all ones over time x latitude x longitude, with latitudes 0 and 60 degrees
        sla = np.ones((2, 2, 1))
        return xr.Dataset(
            {"sla": (["time", "latitude", "longitude"], sla)},
            coords={"time": [0, 1], "latitude": [0.0, 60.0], "longitude": [0.0]},
        )

    def test_weights_are_cosine_of_latitude(self):
        ds = self._make_dataset()
        weighted = apply_weights_to_sea_level_anomaly_data(ds)
        # cos(0 deg) = 1, cos(60 deg) = 0.5
        np.testing.assert_allclose(weighted["sla"].sel(latitude=0.0).values, 1.0)
        np.testing.assert_allclose(weighted["sla"].sel(latitude=60.0).values, 0.5, atol=1e-12)

    def test_original_dataset_is_not_mutated(self):
        ds = self._make_dataset()
        apply_weights_to_sea_level_anomaly_data(ds)
        # the function deep-copies, so the input must stay all ones
        np.testing.assert_allclose(ds["sla"].values, 1.0)


if __name__ == "__main__":
    unittest.main()
