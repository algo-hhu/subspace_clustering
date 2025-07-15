from enum import Enum
from typing import Annotated, Callable

from pydantic import SkipValidation
from pydantic_settings import BaseSettings

from src import distance


class InitialClusteringMethod(str, Enum):
    full_hierarchical_clustering = "full_hierarchical_clustering"
    hierarchical_neighbor_clustering = "hierarchical_neighbor_clustering"
    k_means_clustering = "k_means_clustering"


class InitialDistanceFunction(str, Enum):
    euclidean = distance.euclidean_distance
    thompson = distance.thompson_distance_function


class GlobalSettings(BaseSettings):
    """
    global parameters
    """
    output_path: str = "../output"
    data_path: str = "../data"
    sea_level_anomaly_data_download_path: str = "../data/SEALEVEL_GLO_PHY_L4_MY_008_047"
    half_width: int = 500
    filtered_data_path: str = f"../output/spherical_gaussian_filtering/sea_level_anomaly_data_filtered_{half_width}.nc"
    filtering_sla: bool = True
    resolution: int = 2


class InitialClusteringSettings(BaseSettings):
    """
    specific parameters for initial clustering
    """
    method: InitialClusteringMethod = InitialClusteringMethod.full_hierarchical_clustering
    distance_function: Annotated[Callable, SkipValidation] = InitialDistanceFunction.thompson
    number_of_clusters: list[int] = [25, 20, 15, 10, 8]


class SubspaceClusteringSettings(BaseSettings):
    """
    specific parameters for subspace clustering
    """
    # specific parameters for subspace clustering
    apply_weights: bool = True
    do_subspace_clustering: bool = True
    number_of_clusters: int = 8
    number_of_components: list[int] = [5, 10, 15, 30]
    integrated_connectivity: bool = True


class EvaluationSettings(BaseSettings):
    """
    specific parameters for evaluation
    """
    do_evaluation: bool = True
    number_of_clusters: int = 8
