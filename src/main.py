import asyncio
import os

import xarray as xr

from src import plotting, distance, helper
from src.clustering import complete_hierarchical_clustering, subspace_clustering, \
    neighborhood_clustering
from src.preprocessing import preprocessing_data


# from prisma.engine import http


# # Configure the timeout globally for all Prisma HTTP requests
# http.DEFAULT_TIMEOUT = 120.0  # 60 seconds
# db = Prisma()


async def main():
    # netcdf encoding
    encoding = {
        'sla': {
            'zlib': True,  # Enable compression
            'complevel': 4,  # Compression level (1-9, trade-off between speed and compression ratio)
            'shuffle': True,  # Improve compression efficiency
            'dtype': 'float32',  # Convert from float64 to float32 to save space (optional)
            'chunksizes': (73, 144, 288),  # Use the same efficient chunking as in the smaller dataset
            '_FillValue': -2147483648,  # Match fill value from the smaller dataset
            'scale_factor': 0.0001  # Match scale factor for consistency
        }
    }
    # set parameters for filtering
    filtering_sla = True
    half_width = 500
    # set parameters for initial clustering
    resolution = 2  # resolution of the grid
    number_of_clusters = 15  # number of clusters to reduce to
    full_hierarchical_clustering = False  # Clustering all grid points hierarchically with a given distance function
    do_neighborhood_clustering = False  # Clustering the grid points hierarchically that are neighbors to each other
    # parameters for subspace clustering
    do_subspace_clustering = False  # Given a start clustering, perform subspace clustering
    do_subspace_clustering_with_integrated_connectivity = True  # In each iteration of the subspace clustering, only
    # the border of a cluster is allowed to change its cluster
    number_of_components = [3]  # set the dimension of the subspaces
    out_dir = (
        f"../output")
    if filtering_sla:
        out_dir = f"{out_dir}/filter_{half_width}/{resolution}_degree_grid"
    else:
        out_dir = f"{out_dir}/no_filtering/{resolution}_degree_grid"
    # thompson clustering: full_hierarchical_clustering_thompson_distance_function
    # neighborhood_clustering_thompson_distance_function
    # neighborhood_clustering_euclidean_distance
    initial_clustering_path = (
        f"{out_dir}/neighborhood_clustering_euclidean_distance/{number_of_clusters}_clusters.nc")
    filtered_data_path = f"../output/spherical_gaussian_filtering/sea_level_anomaly_data_filtered_{half_width}.nc"
    if subspace_clustering:
        out_dir = initial_clustering_path.rsplit('/', 1)[0]
        out_dir = f"{out_dir}/subspace_clustering"
    # create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    variable_to_plot = "sla"

    # satellite altimetry data is from 1993 to 2023, latitude: -89.875 to 89.875, longitude: 0.125 to 359.875
    # There are 720 latitude points and 1440 longitude points => 1036800 grid points (the resolution is 0.25 degrees)
    # Merge the data from all files into one xarray dataset
    if not os.path.exists("../data/sea_level_anomaly_data.nc"):
        sea_level_anomaly_data = preprocessing_data.read_satellite_data("../data/SEALEVEL_GLO_PHY_L4_MY_008_047")
        # change longitude from 0-360 to -180-180
        sea_level_anomaly_data = sea_level_anomaly_data.assign_coords(
            longitude=(sea_level_anomaly_data.longitude + 180) % 360 - 180)
        sea_level_anomaly_data = sea_level_anomaly_data.sortby("longitude")
        # save netCDF file
        helper.save_xarray_dataset("../data/sea_level_anomaly_data.nc", sea_level_anomaly_data)
    else:
        sea_level_anomaly_data = xr.open_dataset("../data/sea_level_anomaly_data.nc")

    # filtering
    if filtering_sla and not do_subspace_clustering:  # apply Gaussian filter & temporal low-pass filter
        if not os.path.exists(filtered_data_path):
            # filter spatially with a symmetric Gaussian filter of half-width 500 km
            sea_level_anomaly_data = preprocessing_data.filtering(sea_level_anomaly_data, filtered_data_path,
                                                                  half_width)
            # plot
            plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, variable_to_plot,
                                                name=f"gaussian_filtered_{half_width}")
        else:
            sea_level_anomaly_data = xr.open_dataset(filtered_data_path)

    # initial clustering either with hierarchical clustering or neighborhood clustering
    if do_neighborhood_clustering:
        distance_function = distance.thompson_distance_function
        name = distance_function.__name__
        current_out_dir = f"{out_dir}/neighborhood_clustering_{name}/"
        if not os.path.exists(current_out_dir):
            os.makedirs(current_out_dir)
        # interpolate the grid to match the desired resolution, this is mainly a performance consideration
        sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=range(-90, 91, resolution),
                                                               longitude=range(-180, 180, resolution))
        # calculate the neighborhood clustering
        neighborhood_clustering.start_clustering(sea_level_anomaly_data, [100, 80, 90, 70, 60, 50, 25, 20, 15, 10],
                                                 distance_function, current_out_dir)

    if full_hierarchical_clustering:
        distance_function = distance.thompson_distance_function
        distance_function_name = distance_function.__name__
        current_out_dir = f"{out_dir}/full_hierarchical_clustering_{distance_function_name}/"
        if not os.path.exists(current_out_dir):
            os.makedirs(current_out_dir)
        # plot used data
        plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, current_out_dir, variable_to_plot, name="used_data")

        # interpolate to 5 degree grid
        sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=range(-90, 91, resolution),
                                                               longitude=range(-180, 180, resolution))
        plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, current_out_dir, variable_to_plot,
                                            name=f"{resolution}_degree_grid_filtered")
        k = [100, 50, 25, 20, 15, 10, 8]
        complete_hierarchical_clustering.start_clustering(k, sea_level_anomaly_data, current_out_dir,
                                                          distance_function)

    # subspace clustering
    if do_subspace_clustering:
        initial_clustering = xr.open_dataset(initial_clustering_path)
        subspace_clustering.start_subspace_clustering(sea_level_anomaly_data, initial_clustering,
                                                      f"{out_dir}",
                                                      number_of_components)

    if do_subspace_clustering_with_integrated_connectivity:
        initial_clustering = xr.open_dataset(initial_clustering_path)
        subspace_clustering.start_subspace_clustering_with_integrated_connectivity(sea_level_anomaly_data,
                                                                                   initial_clustering,
                                                                                   f"{out_dir}",
                                                                                   number_of_components)


if __name__ == "__main__":
    asyncio.run(main())
