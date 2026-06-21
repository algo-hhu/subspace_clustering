from enum import Enum
from pathlib import Path
from typing import Annotated, Callable

from pydantic import BaseModel, SkipValidation

from src import distance

# Project root, resolved from this file's location (src/settings/settings.py -> repo root).
# Anchoring paths here keeps them independent of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class InitialClusteringMethod(str, Enum):
    agglomerative_clustering = "agglomerative_clustering"
    agglomerative_connected_clustering = "agglomerative_connected_clustering"
    k_means_clustering = "k_means_clustering_with_connectivity"
    wards_method_connected = "wards_method_connected"


class InitialDistanceFunction(str, Enum):
    euclidean = distance.euclidean_distance
    spatio_temporal_distance_function = distance.spatio_temporal_distance_function


class GlobalSettings(BaseModel):
    """
    global parameters
    """
    output_path: str = str(PROJECT_ROOT / "output/principle_angles")
    data_path: str = str(PROJECT_ROOT.parent / "data")
    sea_level_anomaly_data_download_path: str = str(PROJECT_ROOT.parent / "data" / "SEALEVEL_GLO_PHY_L4_MY_008_047")
    variable: str = "sla"
    resolution: int = 2
    filtering_sla: bool = True
    half_width: int = 500
    random_seed: int = 13

    @property
    def filtered_data_path(self) -> str:
        # Computed from the live values so overriding output_path or half_width stays consistent.
        return (f"{self.output_path}/spherical_gaussian_filtering/"
                f"sea_level_anomaly_data_filtered_{self.half_width}.nc")


class InitialClusteringSettings(BaseModel):
    """
    specific parameters for initial clustering
    """
    method: InitialClusteringMethod = InitialClusteringMethod.agglomerative_clustering
    distance_function: Annotated[Callable, SkipValidation] = InitialDistanceFunction.spatio_temporal_distance_function
    number_of_clusters: list[int] = [25]


class SubspaceClusteringSettings(BaseModel):
    """
    specific parameters for subspace clustering
    """
    # specific parameters for subspace clustering
    apply_weights: bool = True
    do_subspace_clustering: bool = True
    number_of_clusters: int = 25
    number_of_components: list[int] = [5, 10, 15, 30]
    integrated_connectivity: bool = True


class EvaluationSettings(BaseModel):
    """
    specific parameters for evaluation
    """
    do_evaluation: bool = False
    number_of_clusters: int = 25
