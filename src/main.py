import asyncio
import os
import time

import xarray as xr
from loguru import logger
from prisma import Prisma

from src import populate_database, hierarchical_clustering
from src.preprocessing import read_satellite_data

db = Prisma()


async def main():
    # create database tables
    logger.info(f"Establish connection to database and create tables")
    await db.connect()
    logger.info("Database tables created")
    out_dir = "../output/Preprocessing/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    time_1 = time.time()
    # data is from 1993 to 2023, latitude: -89.875 to 89.875, longitude: 0.125 to 359.875
    # There are 720 latitude points and 1440 longitude points => 1036800 grid points
    if not os.path.exists("../data/sea_level_anomaly_data.nc"):
        sea_level_anomaly_data = read_satellite_data("../data/SEALEVEL_GLO_PHY_L4_MY_008_047")
        # change longitude from 0-360 to -180-180
        sea_level_anomaly_data = sea_level_anomaly_data.assign_coords(
            longitude=(sea_level_anomaly_data.longitude + 180) % 360 - 180)
        sea_level_anomaly_data = sea_level_anomaly_data.sortby("longitude")
        # save netCDF file
        sea_level_anomaly_data.to_netcdf("../data/sea_level_anomaly_data.nc")
        time_2 = time.time()
        logger.info(f"Time taken to read and save data: {time_2 - time_1}")
    else:
        sea_level_anomaly_data = xr.open_dataset("../data/sea_level_anomaly_data.nc")
        time_2 = time.time()
        logger.info(f"Time taken to read data: {time_2 - time_1}")
    logger.info(f"Preprocessing done")

    # filter spatially with a symmetric Gaussian filter of half-width 500 km
    # leave filtering for now and decide later if it is necessary
    # filtered_data = filtering(sea_level_anomaly_data, time_2, out_dir)
    # Apply a convolution low-pass filter passing 90% of the amplitude at 24 months to each time series.

    logger.info(f"Initially populating database with grid points, differences, clusters and merge history")

    # distance function
    # D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
    # d is Euclidean distance, r is temporal correlation coefficient, a is constant such that the value of the
    # exponential is 0.5, when d=3000 km -> can be found in

    # generate grid_point objects for each grid point - only needs to be done once
    calculating_initial_grid_points = True
    if calculating_initial_grid_points:
        await (populate_database.generate_grid_points_and_initial_clusters(sea_level_anomaly_data, db))

    # calculate initial differences between grid points
    calculate_initial_differences = True
    if calculate_initial_differences:
        await populate_database.calculate_initial_differences(db)
    logger.info(f"Start hierarchical clustering")
    # start clustering
    await hierarchical_clustering.start_clustering(db, [100, 10, 12, 8], sea_level_anomaly_data)


# TODO: filter spatially with a symmetric Gaussian filter of half-width 500 km
# TODO: interpolate to 5 degree grid
# TODO: Apply a convolution low-pass filter passing 90% of the amplitude at 24 months to each time series.  (To emphasize inter annual and longer variability)
# TODO: implement distance function between two grid points x_i and x_j - D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
# TODO: d is Euclidean distance, r is temporal correlation coefficient, a is constant such that the value of the exponential is 0.5, when d=3000 km
# TODO: calculate distances between each pair of grid points
# TODO: hierarchical clustering

if __name__ == "__main__":
    asyncio.run(main())
