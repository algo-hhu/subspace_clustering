import os

import numpy as np
import xarray
import xarray as xr
from loguru import logger

from src import plotting, weighting, distance
from src.clustering.complete_hierarchical_clustering import CompleteHierarchicalClustering
from src.clustering.k_means_clustering import KMeansClustering
from src.clustering.neighborhood_clustering import NeighborhoodClustering
from src.clustering.subspace_clustering import calculate_subspace_clustering
from src.clustering.wards_method_scikit_learn_connectivity import WardsMethodConnected
from src.evaluation.evaluate import evaluate_clustering
from src.preprocessing.preprocessing_data import start_preprocessing
from src.settings import settings
from src.settings.settings import InitialClusteringMethod


def main():
    # the default settings are: filter with 500 km halfwidth, thompson distance function, full hierarchical clustering
    global_settings = settings.GlobalSettings()
    initial_clustering_settings = settings.InitialClusteringSettings()
    subspace_clustering_settings = settings.SubspaceClusteringSettings()
    evaluation_settings = settings.EvaluationSettings()

    collect_output_file_path = prepare_output_file(global_settings, initial_clustering_settings,
                                                   subspace_clustering_settings)
    logger.info("Starting preprocessing")
    variable_to_plot = global_settings.variable
    out_dir, sea_level_anomaly_data, unfiltered_sea_level_anomaly_data, unprocessed_sea_level_anomaly_data = (
        start_preprocessing(
            global_settings,
            variable_to_plot))

    # plot_gradient(out_dir, unfiltered_sea_level_anomaly_data)

    # initial clustering
    out_dir = calculate_initial_clustering(initial_clustering_settings, out_dir, sea_level_anomaly_data)

    if subspace_clustering_settings.apply_weights:
        logger.info("Applying weights before subspace clustering")
        unfiltered_sea_level_anomaly_data = weighting.apply_weights_to_sea_level_anomaly_data(
            unfiltered_sea_level_anomaly_data)

    initial_clustering = xarray.open_dataset(
        f"{out_dir}/clustering_{subspace_clustering_settings.number_of_clusters}.nc")
    # subspace clustering
    calculate_subspace_clustering(global_settings, out_dir, subspace_clustering_settings,
                                  unfiltered_sea_level_anomaly_data, initial_clustering, collect_output_file_path)

    # evaluate clustering results
    evaluate_clustering(evaluation_settings, out_dir, unfiltered_sea_level_anomaly_data,
                        subspace_clustering_settings)


def prepare_output_file(global_settings, initial_clustering_settings, subspace_clustering_settings):
    # output_file to collect distances from clusters to subspaces in a latex table
    collect_output_file_path = (f"{global_settings.output_path}/results_"
                                f"{subspace_clustering_settings.number_of_clusters}_clusters.csv")
    if not os.path.exists(global_settings.output_path):
        os.makedirs(global_settings.output_path)
    with open(collect_output_file_path, "a") as f:
        f.write(
            f"Distances from clusters to subspaces for {subspace_clustering_settings.number_of_clusters} clusters:\n")
        if global_settings.filtering_sla:
            f.write("Filter 500 km halfwidth \n")
        else:
            f.write("No filtering \n")
        f.write(f"Method: {initial_clustering_settings.method.value}\n")
        f.write(f"Distance function: {initial_clustering_settings.distance_function.__name__}\n")
        if subspace_clustering_settings.do_subspace_clustering:
            f.write(f"Subspace clustering with {subspace_clustering_settings.number_of_components} components\n")
            f.write("Weights applied: "
                    f"{subspace_clustering_settings.apply_weights}\n\n")
    return collect_output_file_path


def calculate_initial_clustering(initial_clustering_settings, out_dir: str,
                                 sea_level_anomaly_data: xr.Dataset):
    """
    Calculate initial clustering
    :param initial_clustering_settings:
    :param out_dir:
    :param sea_level_anomaly_data:
    :return:
    """
    out_dir = (
        f"{out_dir}/{initial_clustering_settings.method.value}_"
        f"{initial_clustering_settings.distance_function.__name__}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"output directory: {out_dir}")
    # plot used data
    plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, "sla", "used_data")
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
            f"function {initial_clustering_settings.distance_function.__name__} have already been calculated")
        return out_dir

    if initial_clustering_settings.method == InitialClusteringMethod.agglomerative_clustering:
        initial_clustering = CompleteHierarchicalClustering(sea_level_anomaly_data,
                                                            initial_clustering_settings.number_of_clusters,
                                                            initial_clustering_settings.distance_function, out_dir)
    elif initial_clustering_settings.method == InitialClusteringMethod.agglomerative_connected_clustering:
        initial_clustering = NeighborhoodClustering(sea_level_anomaly_data,
                                                    initial_clustering_settings.number_of_clusters,
                                                    initial_clustering_settings.distance_function, out_dir)
    elif initial_clustering_settings.method == InitialClusteringMethod.k_means_clustering:
        initial_clustering = KMeansClustering(sea_level_anomaly_data, initial_clustering_settings.number_of_clusters,
                                              initial_clustering_settings.distance_function, out_dir)
    elif initial_clustering_settings.method == InitialClusteringMethod.wards_method_connected:
        initial_clustering = WardsMethodConnected(sea_level_anomaly_data,
                                                  initial_clustering_settings.number_of_clusters,
                                                  distance.distance_for_wards_method, out_dir)
    else:
        raise NotImplementedError
    initial_clustering.start_initial_clustering()
    return out_dir


def compute_nan_mask(dataset: xarray.Dataset, var: str) -> xarray.DataArray:
    """
    Compute a nan mask for the dataset
    :param dataset:
    :param var:
    :return:
    """
    nan_mask = np.isnan(dataset[var].values).any(axis=0)
    return xarray.DataArray(
        nan_mask,
        coords={"latitude": dataset.latitude, "longitude": dataset.longitude},
        dims=["latitude", "longitude"]
    )


def change_clustering_resolution(sea_level_data: xarray.Dataset, clustering_data: xarray.Dataset, out_dir: str):
    """
    Change the clustering resolution to match the sea level data
    :param out_dir:
    :param sea_level_data:
    :param clustering_data:
    :return:
    """
    if not os.path.exists(f"{out_dir}/clustering_15_025_resolution.nc"):
        logger.info("Regridding clustering data to match sea level data resolution")
        # interpolate the sea level data to the clustering resolution
        interpolated_sea_level_data = sea_level_data.interp(latitude=clustering_data['latitude'],
                                                            longitude=clustering_data['longitude'])
        # change every point in the clustering to nan, where the sea level data is nan in any time step
        nan_mask_da = compute_nan_mask(interpolated_sea_level_data, "sla")
        clustering_data = clustering_data.where(~nan_mask_da, np.nan)
        # regrid the clustering data to match the sea level data
        ref_grid = sea_level_data.isel(time=0)
        fine_resolution_clusters = clustering_data.interp_like(ref_grid, method="nearest")
        # save the fine resolution clusters
        fine_resolution_clusters.to_netcdf(f"{out_dir}/clustering_15_025_resolution.nc")
        # put nan values where there are nan values in the sea level data
        # masking happens twice as the interpolation might leak into the nan areas
        nan_mask_da = compute_nan_mask(sea_level_data, "sla")
        fine_resolution_clusters = fine_resolution_clusters.where(~nan_mask_da, np.nan)

    else:
        fine_resolution_clusters = xarray.open_dataset(f"{out_dir}/clustering_15_025_resolution.nc")

    # plot clustering before and after regridding
    plotting.plot_xarray_dataset_on_map(clustering_data, out_dir, "clustering_before_regridding")
    plotting.plot_xarray_dataset_on_map(fine_resolution_clusters, out_dir, "clustering_after_regridding")
    return fine_resolution_clusters


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as e:
        logger.error(f"Run failed: {e}")
        raise SystemExit(1)
