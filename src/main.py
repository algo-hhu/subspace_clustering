import os

import xarray as xr
from loguru import logger

from src import plotting, distance, helper
from src.clustering import complete_hierarchical_clustering, subspace_clustering
from src.clustering.complete_hierarchical_clustering import CompleteHierarchicalClustering
from src.clustering.neighborhood_clustering import NeighborhoodClustering
from src.evaluation import evaluate
from src.helper import adjust_resolution
from src.preprocessing import preprocessing_data
from src.settings import GlobalSettings, InitialClusteringSettings, SubspaceClusteringSettings, InitialClusteringMethod, \
    EvaluationSettings


def main():
    global_settings = GlobalSettings()
    initial_clustering_settings = InitialClusteringSettings()
    subspace_clustering_settings = SubspaceClusteringSettings()
    evaluation_settings = EvaluationSettings()
    # set parameters for filtering
    # filtering_sla = False
    # half_width = 50
    # # set parameters for initial clustering
    # resolution = 2  # resolution of the grid
    # number_of_clusters = 15  # number of clusters to reduce to
    # k = [100, 50, 25, 20, 15, 10, 8]  # number of clusters for initial clustering
    # full_hierarchical_clustering = False  # Clustering all grid points hierarchically with a given distance function
    # do_neighborhood_clustering = True  # Clustering the grid points hierarchically that are neighbors to each other
    # # parameters for subspace clustering
    # do_subspace_clustering = False  # Given a start clustering, perform subspace clustering
    # do_subspace_clustering_with_integrated_connectivity = False  # In each iteration of the subspace clustering, only
    # evaluate_clustering = False
    # # the border of a cluster is allowed to change its cluster
    # number_of_components = [3, 5, 7]  # set the dimension of the subspaces
    # out_dir = (
    #     f"../output")
    # if filtering_sla:
    #     out_dir = f"{out_dir}/filter_{half_width}/{resolution}_degree_grid"
    # else:
    #     out_dir = f"{out_dir}/no_filtering/{resolution}_degree_grid"
    # # thompson clustering: full_hierarchical_clustering_thompson_distance_function
    # # neighborhood_clustering_thompson_distance_function
    # # neighborhood_clustering_euclidean_distance
    # initial_clustering_path = (
    #     f"{out_dir}/full_hierarchical_clustering_thompson_distance_function/{number_of_clusters}_clusters.nc")
    # filtered_data_path = f"../output/spherical_gaussian_filtering/sea_level_anomaly_data_filtered_{half_width}.nc"
    # if do_subspace_clustering or do_subspace_clustering_with_integrated_connectivity:
    #     out_dir = initial_clustering_path.rsplit('/', 1)[0]
    #     out_dir = f"{out_dir}/subspace_clustering"
    #
    # # path to clustering that should be evaluated
    # eval_clustering_path = f"{out_dir}/neighborhood_clustering_euclidean_distance/subspace_clustering/establish_connectivity_every_iteration/components_3/clustering_14"
    # # create output directory
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    variable_to_plot = "sla"

    # satellite altimetry data is from 1993 to 2023, latitude: -89.875 to 89.875, longitude: 0.125 to 359.875
    # There are 720 latitude points and 1440 longitude points => 1036800 grid points (the resolution is 0.25 degrees)
    # Merge the data from all files into one xarray dataset
    if not os.path.exists(f"{global_settings.data_path}/sea_level_anomaly_data.nc"):
        logger.info(f"Reading sea level anomaly data from files in ../data/SEALEVEL_GLO_PHY_L4_MY_008_047")
        unfiltered_sea_level_anomaly_data = preprocessing_data.read_satellite_data(
            global_settings.sea_level_anomaly_data_download_path)
        # change longitude from 0-360 to -180-180
        unfiltered_sea_level_anomaly_data = unfiltered_sea_level_anomaly_data.assign_coords(
            longitude=(unfiltered_sea_level_anomaly_data.longitude + 180) % 360 - 180)
        unfiltered_sea_level_anomaly_data = unfiltered_sea_level_anomaly_data.sortby("longitude")
        # save netCDF file
        helper.save_xarray_dataset(f"{global_settings.data_path}/sea_level_anomaly_data.nc",
                                   unfiltered_sea_level_anomaly_data)
    else:
        unfiltered_sea_level_anomaly_data = xr.open_dataset(f"{global_settings.data_path}/sea_level_anomaly_data.nc")
    out_dir = global_settings.output_path
    # filtering
    if global_settings.filtering_sla:  # apply Gaussian
        out_dir = f"{out_dir}/filter_{global_settings.half_width}"
        # filter & temporal low-pass filter
        if not os.path.exists(global_settings.filtered_data_path):
            # filter spatially with a symmetric Gaussian filter of half-width 500 km
            sea_level_anomaly_data = preprocessing_data.filtering(unfiltered_sea_level_anomaly_data,
                                                                  global_settings.filtered_data_path,
                                                                  global_settings.half_width)
            # plot
            plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, global_settings.output_path, variable_to_plot,
                                                name=f"gaussian_filtered_{global_settings.half_width}")
        else:
            sea_level_anomaly_data = xr.open_dataset(global_settings.filtered_data_path)
        # adjust resolution
        out_dir = f"{out_dir}/{global_settings.resolution}_degree_grid"
        resolution_path = f"../output/resolutions/sea_level_anomaly_data_filtered_{global_settings.half_width}_{global_settings.resolution}_degree.nc"
        sea_level_anomaly_data = adjust_resolution(global_settings.resolution, resolution_path, sea_level_anomaly_data)
    else:
        out_dir = f"{out_dir}/no_filtering"
        out_dir = f"{out_dir}/{global_settings.resolution}_degree_grid"
        resolution_path = f"../output/resolutions/sea_level_anomaly_data_no_filter_{global_settings.resolution}_degree.nc"
        # check for correct resolution
        sea_level_anomaly_data = adjust_resolution(global_settings.resolution, resolution_path,
                                                   unfiltered_sea_level_anomaly_data)

    # initial clustering
    out_dir = f"{out_dir}/{initial_clustering_settings.method.value}_{initial_clustering_settings.distance_function.__name__}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print(f"output directory: {out_dir}")
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

    # initial clustering either with hierarchical clustering or neighborhood clustering
    if initial_clustering_settings.method == InitialClusteringMethod.hierarchical_neighbor_clustering:
        logger.info("Starting neighborhood clustering")
        current_distance_function = initial_clustering_settings.distance_function
        name = current_distance_function.__name__
        out_dir = f"{out_dir}/{initial_clustering_settings.method.value}_{name}/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print(f"output directory: {out_dir}")
        initial_clustering = NeighborhoodClustering(sea_level_anomaly_data,
                                                    initial_clustering_settings.number_of_clusters,
                                                    initial_clustering_settings.distance_function, out_dir)
        # check if this has already been done
        for current_k in initial_clustering_settings.number_of_clusters:  # check if the clustering for this k already exists
            clustering_path = f"{out_dir}/clustering_{current_k}.nc"
            # TODO: modify this such that it only calls the function for the missing k
            if not os.path.exists(clustering_path):
                # calculate the neighborhood clustering
                initial_clustering.start_initial_clustering()
                break
            else:
                logger.info(f"Neighborhood clustering for k={current_k} already exists. Skipping.")

    elif initial_clustering_settings.method == InitialClusteringMethod.full_hierarchical_clustering:
        distance_function = distance.thompson_distance_function
        distance_function_name = distance_function.__name__
        out_dir = f"{out_dir}/{initial_clustering_settings.method.value}_{distance_function_name}/"
        print(f"output directory: {out_dir}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # plot used data
        plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, variable_to_plot, name="used_data")

        # check if this has already been done
        for current_k in initial_clustering_settings.number_of_clusters:
            clustering_path = f"{out_dir}/clustering_{current_k}.nc"
            if not os.path.exists(clustering_path):
                # calculate the hierarchical clustering
                complete_hierarchical_clustering.start_clustering(initial_clustering_settings.number_of_clusters,
                                                                  sea_level_anomaly_data, out_dir,
                                                                  distance_function)
                break
            else:
                logger.info(f"Hierarchical clustering for k={current_k} already exists. Skipping.")

    # subspace clustering
    if subspace_clustering_settings.do_subspace_clustering:
        print(f"output directory: {global_settings.output_path}")
        initial_clustering = xr.open_dataset(
            f"{out_dir}/clustering_{subspace_clustering_settings.number_of_clusters}.nc")
        subspace_clustering.start_subspace_clustering(sea_level_anomaly_data, initial_clustering,
                                                      f"{out_dir}",
                                                      subspace_clustering_settings.number_of_components)

    if subspace_clustering_settings.integrated_connectivity:

        current_out_dir = f"{out_dir}/integrated_connectivity"
        print(f"output directory: {current_out_dir}")
        if not os.path.exists(current_out_dir):
            os.makedirs(current_out_dir)
        initial_clustering = xr.open_dataset(
            f"{out_dir}/clustering_{subspace_clustering_settings.number_of_clusters}.nc")
        subspace_clustering.start_subspace_clustering_with_integrated_connectivity(sea_level_anomaly_data,
                                                                                   initial_clustering,
                                                                                   current_out_dir,
                                                                                   subspace_clustering_settings.number_of_components,
                                                                                   global_settings.resolution)

    if evaluation_settings.do_evaluation:
        options = ("establish_connectivity_every_iteration", "establish_connectivity_once",
                   "filter_every_round_connectivity_once" "integrated_connectivity")
        for connectivity_option in options:
            current_out_dir = f"{out_dir}/{connectivity_option}/evaluation"
            eval_clustering_path = f"{out_dir}/{connectivity_option}/clustering.nc"
            if not os.path.exists(current_out_dir):
                os.makedirs(current_out_dir)
            print(f"output directory: {current_out_dir}")
            clustering = xr.open_dataset(eval_clustering_path)
            evaluate.start_evaluation(clustering, current_out_dir, sea_level_anomaly_data)


if __name__ == "__main__":
    main()
