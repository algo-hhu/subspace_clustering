import asyncio
import os

import xarray as xr
from loguru import logger

from src import plotting, distance
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
    filtering_sla = False
    half_width = 500
    # set parameters for initial clustering
    out_dir = "../output/test_reestablishing_connectivity3/"
    resolution = 2  # resolution of the grid
    full_hierarchical_clustering = False  # Clustering all grid points hierarchically with a given distance function
    do_neighborhood_clustering = False  # Clustering the grid points hierarchically that are neighbors to each other
    # parameters for subspace clustering
    do_subspace_clustering = True  # Given a start clustering, perform subspace clustering
    number_of_components = [3]  # set the dimension of the subspaces
    initial_clustering_path = "../output/no_filter/neighborhood_clustering_euclidean_distance_no_filter/clusters_15.nc"
    # create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    variable_to_plot = "sla"

    # read data
    # satellite altimetry data is from 1993 to 2023, latitude: -89.875 to 89.875, longitude: 0.125 to 359.875
    # There are 720 latitude points and 1440 longitude points => 1036800 grid points
    if not os.path.exists("../data/sea_level_anomaly_data.nc"):
        sea_level_anomaly_data = preprocessing_data.read_satellite_data("../data/SEALEVEL_GLO_PHY_L4_MY_008_047")
        # change longitude from 0-360 to -180-180
        sea_level_anomaly_data = sea_level_anomaly_data.assign_coords(
            longitude=(sea_level_anomaly_data.longitude + 180) % 360 - 180)
        sea_level_anomaly_data = sea_level_anomaly_data.sortby("longitude")
        # save netCDF file
        sea_level_anomaly_data.to_netcdf("../data/sea_level_anomaly_data.nc", encoding=encoding, format="NETCDF4")
    else:
        sea_level_anomaly_data = xr.open_dataset("../data/sea_level_anomaly_data.nc")

    # filtering
    if filtering_sla:  # apply Gaussian filter & temporal low-pass filter
        if not os.path.exists(f"../data/sea_level_anomaly_data_filtered_{half_width}.nc"):
            # filter spatially with a symmetric Gaussian filter of half-width 500 km
            sea_level_anomaly_data = preprocessing_data.filtering(sea_level_anomaly_data, out_dir, half_width)
            # plot
            plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, variable_to_plot,
                                                name=f"gaussian_filtered_{half_width}")
        else:
            sea_level_anomaly_data = xr.open_dataset(f"../data/sea_level_anomaly_data_filtered_{half_width}.nc")

    # initial clustering either with hierarchical clustering or neighborhood clustering
    if do_neighborhood_clustering:
        current_out_dir = f"{out_dir}/neighborhood_clustering/"
        if not os.path.exists(current_out_dir):
            os.makedirs(current_out_dir)
        distance_function = distance.euclidean_distance

        sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=range(-90, 91, resolution),
                                                               longitude=range(-180, 180, resolution))
        neighborhood_clustering.start_clustering(sea_level_anomaly_data, [100, 80, 90, 70, 60, 50, 25, 20, 15, 10],
                                                 distance_function, current_out_dir)
    if full_hierarchical_clustering:
        current_out_dir = f"{out_dir}/full_hierarchical_clustering/"
        if not os.path.exists(current_out_dir):
            os.makedirs(current_out_dir)
        # plot used data
        plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, current_out_dir, variable_to_plot, name="used_data")
        logger.info("Interpolating to 5 degree grid")
        # interpolate to 5 degree grid
        sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=range(-90, 91, resolution),
                                                               longitude=range(-180, 180, resolution))
        plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, current_out_dir, variable_to_plot,
                                            name="5_degree_grid_filtered")
        k = [100, 50, 25, 20, 15, 10, 8]
        complete_hierarchical_clustering.start_clustering(k, sea_level_anomaly_data, current_out_dir,
                                                          distance_function=distance.distance_function)

    # subspace clustering
    if do_subspace_clustering:
        for component in number_of_components:
            current_out_dir = f"{out_dir}/components_{component}/"
            if not os.path.exists(current_out_dir):
                os.makedirs(current_out_dir)
        initial_clustering = xr.open_dataset(initial_clustering_path)
        subspace_clustering.start_subspace_clustering(sea_level_anomaly_data, initial_clustering, out_dir,
                                                      number_of_components)


if __name__ == "__main__":
    asyncio.run(main())
