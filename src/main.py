import math
import os
import time

import xarray as xr
from loguru import logger
from sqlalchemy import create_engine, URL

from src import preprocessing
from src.grid_point import Base
from src.preprocessing import read_satellite_data

# Import all models to ensure they are registered with SQLAlchemy
from src.cluster import Cluster
from src.grid_point import GridPoint
from src.sea_level_differences import Difference
from src.merge_history import MergeHistory
from src.cluster_grid_point_relationship import cluster_grid_points

if __name__ == '__main__':
    # database engine
    url = URL.create("postgresql", username="postgres", password="postgres", host="localhost", port=5432,
                     database="postgres")
    engine = create_engine(url)
    Base.metadata.create_all(engine)
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
    # exponential is 0.5, when d=3000 km
    # calculate distances between each pair of grid points
    a = math.sqrt(- (1500 / (math.log(0.5))))
    # generate grid_point objects for each grid point
    preprocessing.generate_grid_points(sea_level_anomaly_data)
    logger.info(f"Start hierarchical clustering")

# TODO: filter spatially with a symmetric Gaussian filter of half-width 500 km
# TODO: interpolate to 5 degree grid
# TODO: Apply a convolution low-pass filter passing 90% of the amplitude at 24 months to each time series.  (To emphasize inter annual and longer variability)
# TODO: implement distance function between two grid points x_i and x_j - D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
# TODO: d is Euclidean distance, r is temporal correlation coefficient, a is constant such that the value of the exponential is 0.5, when d=3000 km
# TODO: calculate distances between each pair of grid points
# TODO: hierarchical clustering
