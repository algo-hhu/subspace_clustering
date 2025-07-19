from enum import Enum
from typing import Annotated, Callable

from pydantic import SkipValidation
from pydantic_settings import BaseSettings

from src import distance


class InitialClusteringMethod(str, Enum):
    agglomerative_clustering = "agglomerative_clustering"
    agglomerative_connected_clustering = "agglomerative_connected_clustering"
    k_means_clustering = "k_means_clustering_with_connectivity"
    wards_method_connected = "wards_method_connected"


class InitialDistanceFunction(str, Enum):
    euclidean = distance.euclidean_distance
    spatio_temporal_distance_function = distance.spatio_temporal_distance_function


class GlobalSettings(BaseSettings):
    """
    global parameters
    """
    output_path: str = "../output/final_results"
    data_path: str = "../data"
    sea_level_anomaly_data_download_path: str = "../data/SEALEVEL_GLO_PHY_L4_MY_008_047"
    half_width: int = 500
    filtered_data_path: str = f"../output/spherical_gaussian_filtering/sea_level_anomaly_data_filtered_{half_width}.nc"
    filtering_sla: bool = True
    # resolution: float = 0.25
    resolution: int = 2


class InitialClusteringSettings(BaseSettings):
    """
    specific parameters for initial clustering
    """
    method: InitialClusteringMethod = InitialClusteringMethod.agglomerative_clustering
    distance_function: Annotated[Callable, SkipValidation] = InitialDistanceFunction.spatio_temporal_distance_function
    number_of_clusters: list[int] = [25, 20, 15, 12, 10, 8]


class SubspaceClusteringSettings(BaseSettings):
    """
    specific parameters for subspace clustering
    """
    # specific parameters for subspace clustering
    apply_weights: bool = True
    do_subspace_clustering: bool = True
    number_of_clusters: int = 15
    # number_of_components: list[int] = [30]
    number_of_components: list[int] = [30]
    integrated_connectivity: bool = True


class EvaluationSettings(BaseSettings):
    """
    specific parameters for evaluation
    """
    do_evaluation: bool = True
    number_of_clusters: int = 15
