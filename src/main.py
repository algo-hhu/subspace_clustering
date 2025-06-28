import os

import xarray as xr
from loguru import logger

from src.clustering.complete_hierarchical_clustering import CompleteHierarchicalClustering
from src.clustering.neighborhood_clustering import NeighborhoodClustering
from src.clustering.subspace_clustering import calculate_subspace_clustering
from src.evaluation.evaluate import evaluate_clustering
from src.preprocessing.preprocessing_data import start_preprocessing
from src.settings import GlobalSettings, InitialClusteringSettings, SubspaceClusteringSettings, EvaluationSettings, \
    InitialClusteringMethod


def main():
    global_settings = GlobalSettings()
    initial_clustering_settings = InitialClusteringSettings()
    subspace_clustering_settings = SubspaceClusteringSettings()
    evaluation_settings = EvaluationSettings()
    variable_to_plot = "sla"

    out_dir, sea_level_anomaly_data, unfiltered_sea_level_anomaly_data = start_preprocessing(global_settings,
                                                                                             variable_to_plot)

    # initial clustering
    out_dir = calculate_initial_clustering(initial_clustering_settings, out_dir, sea_level_anomaly_data)

    # subspace clustering
    calculate_subspace_clustering(global_settings, out_dir, subspace_clustering_settings,
                                  unfiltered_sea_level_anomaly_data)

    # evaluate clustering results
    evaluate_clustering(evaluation_settings, out_dir, unfiltered_sea_level_anomaly_data)


def calculate_initial_clustering(initial_clustering_settings: InitialClusteringSettings, out_dir: str,
                                 sea_level_anomaly_data: xr.Dataset):
    """
    Calculate initial clustering
    :param initial_clustering_settings:
    :param out_dir:
    :param sea_level_anomaly_data:
    :return:
    """
    out_dir = f"{out_dir}/{initial_clustering_settings.method.value}_{initial_clustering_settings.distance_function.__name__}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"output directory: {out_dir}")
    # check if clustering has already been calculated
    number_of_clusters_left = []
    for k in initial_clustering_settings.number_of_clusters:
        if os.path.exists(f"{out_dir}/clustering_{k}.nc"):
            continue
        else:
            number_of_clusters_left.append(k)
    if not number_of_clusters_left:
        logger.info(
            f"All clusters for initial clustering with method {initial_clustering_settings.method} and distance "
            f"function {initial_clustering_settings.distance_function} have already been calculated")
        return out_dir

    if initial_clustering_settings.method == InitialClusteringMethod.full_hierarchical_clustering:
        initial_clustering = CompleteHierarchicalClustering(sea_level_anomaly_data,
                                                            initial_clustering_settings.number_of_clusters,
                                                            initial_clustering_settings.distance_function, out_dir)
    elif initial_clustering_settings.method == InitialClusteringMethod.hierarchical_neighbor_clustering:
        initial_clustering = NeighborhoodClustering(sea_level_anomaly_data,
                                                    initial_clustering_settings.number_of_clusters,
                                                    initial_clustering_settings.distance_function, out_dir)
    else:
        raise NotImplementedError
    initial_clustering.start_initial_clustering()
    return out_dir


if __name__ == "__main__":
    main()
