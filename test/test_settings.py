import os
import unittest

import numpy as np

from src.clustering import subspace_clustering
from src.evaluation import evaluate
from src.main import seed_random_number_generators
from src.settings.settings import GlobalSettings


class TestGlobalSettings(unittest.TestCase):
    def test_random_seed_is_an_int(self):
        # the field exists with an integer default; the exact value is a configurable choice
        self.assertIsInstance(GlobalSettings().random_seed, int)

    def test_random_seed_is_overridable(self):
        self.assertEqual(GlobalSettings(random_seed=7).random_seed, 7)

    def test_paths_are_absolute(self):
        settings = GlobalSettings()
        self.assertTrue(os.path.isabs(settings.output_path))
        self.assertTrue(os.path.isabs(settings.data_path))

    def test_filtered_data_path_default(self):
        settings = GlobalSettings()
        self.assertTrue(
            settings.filtered_data_path.endswith(
                "spherical_gaussian_filtering/sea_level_anomaly_data_filtered_500.nc"
            )
        )

    def test_filtered_data_path_tracks_overrides(self):
        # the computed property must reflect overridden output_path and half_width,
        # unlike the old class-body f-string that froze them at definition time
        settings = GlobalSettings(half_width=300, output_path="/tmp/out")
        self.assertEqual(
            settings.filtered_data_path,
            "/tmp/out/spherical_gaussian_filtering/sea_level_anomaly_data_filtered_300.nc",
        )


class TestSeeding(unittest.TestCase):
    def test_seed_propagates_to_pca_modules(self):
        seed_random_number_generators(123)
        self.assertEqual(subspace_clustering.PCA_RANDOM_STATE, 123)
        self.assertEqual(evaluate.PCA_RANDOM_STATE, 123)

    def test_seeding_makes_numpy_rng_deterministic(self):
        seed_random_number_generators(7)
        first = np.random.rand(5)
        seed_random_number_generators(7)
        second = np.random.rand(5)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
