import asyncio
import multiprocessing
import os

import xarray as xr
from loguru import logger
from prisma import Prisma
from prisma.engine import http

from src import plotting, distance
from src.clustering import hierarchical_clustering, complete_hierarchical_clustering, subspace_clustering, \
    neighborhood_clustering
from src.preprocessing import populate_database, preprocessing_data

# Configure the timeout globally for all Prisma HTTP requests
http.DEFAULT_TIMEOUT = 120.0  # 60 seconds
db = Prisma()


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
    filtering_sla = False
    use_neighborhood_clustering = False
    full_hierarchical_clustering = False
    do_subspace_clustering = True
    do_neighborhood_clustering_without_db = False
    out_dir = "../output/filter_250_halfwidth/"
    number_of_components = 20
    initial_clustering_path = "../output/test_new_clustering/2deg/"
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

    if filtering_sla:  # apply Gaussian filter & temporal low-pass filter
        if not os.path.exists("../data/sea_level_anomaly_data_filtered.nc"):
            # filter spatially with a symmetric Gaussian filter of half-width 500 km
            sea_level_anomaly_data = preprocessing_data.filtering(sea_level_anomaly_data, out_dir, half_width=250)
            # plot
            plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, variable_to_plot,
                                                name="gaussian_filtered")
        else:
            sea_level_anomaly_data = xr.open_dataset("../data/sea_level_anomaly_data_filtered.nc")

    if do_neighborhood_clustering_without_db:
        distance_function = distance.euclidean_distance

        sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=range(-90, 91, 2),
                                                               longitude=range(-180, 180, 2))
        neighborhood_clustering.start_clustering(sea_level_anomaly_data, [100, 80, 90, 70, 60, 50, 25, 20, 15, 10],
                                                 distance_function, out_dir)

    if use_neighborhood_clustering:  # hierarchical clustering using only the neighborhood of each point
        # plot used data
        plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, variable_to_plot, name="used_data")
        # create database tables
        logger.info(f"Establish connection to database and create tables")
        await db.connect()
        logger.info("Database tables created")
        logger.info(f"Initially populating database with grid points, differences, clusters and merge history")
        sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=range(-90, 91, 5),
                                                               longitude=range(-180, 180, 5))
        # generate grid_point objects for each grid point - only needs to be done once
        await (populate_database.generate_grid_points_and_initial_clusters(sea_level_anomaly_data, db))
        # calculate initial differences between grid points
        await populate_database.calculate_initial_differences(db)
        logger.info(f"Start hierarchical clustering")
        # TODO: recalculate the distances between new cluster and neighbors in parallel
        # TODO: use caching to decrease database queries
        await hierarchical_clustering.start_clustering(db, [100, 80, 90, 70, 60, 50, 25, 20, 15, 10],
                                                       sea_level_anomaly_data)

    if full_hierarchical_clustering:
        # plot used data
        plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, variable_to_plot, name="used_data")
        logger.info("Interpolating to 5 degree grid")
        # interpolate to 5 degree grid
        sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=range(-90, 91, 5),
                                                               longitude=range(-180, 180, 5))
        plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, variable_to_plot,
                                            name="5_degree_grid_filtered")
        k = [100, 50, 25, 20, 15, 10, 8]
        complete_hierarchical_clustering.start_clustering(k, sea_level_anomaly_data)

    if do_subspace_clustering:
        initial_clustering = xr.open_dataset(initial_clustering_path)
        subspace_clustering.start_subspace_clustering(sea_level_anomaly_data, initial_clustering, out_dir,
                                                      number_of_components)


# filter spatially with a symmetric Gaussian filter of half-width 500 km
# interpolate to 5 degree grid
# Apply a convolution low-pass filter passing 90% of the amplitude at 24 months to each time series.  (To emphasize inter annual and longer variability)
# implement distance function between two grid points x_i and x_j - D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
# d is Euclidean distance, r is temporal correlation coefficient, a is constant such that the value of the exponential is 0.5, when d=3000 km
# calculate distances between each pair of grid points
# hierarchical clustering

if __name__ == "__main__":
    asyncio.run(main())
    multiprocessing.set_start_method("forkserver")
