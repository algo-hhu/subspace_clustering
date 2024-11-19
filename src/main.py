import math
import os
import time
from itertools import combinations

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from src.preprocessing import read_satellite_data, filtering, distance_function

if __name__ == '__main__':
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

    filtered_data = filtering(sea_level_anomaly_data, time_2, out_dir)
    # Apply a convolution low-pass filter passing 90% of the amplitude at 24 months to each time series.

    time_3 = time.time()
    # flatten the data to have a shape of (365, 1,036,800)
    # select which data to use (either 0.25 or 5 degree grid, additionally filtered or not)
    data = filtered_data['filtered_sla']
    flattened_data = data.stack(spatial=('latitude', 'longitude'))
    flattened_data = flattened_data.dropna(dim='spatial', how='any')
    data_array = flattened_data.values
    time_4 = time.time()
    logger.info(f"Time taken to prepare data for distance calculation: {time_4 - time_3}")

    # get spatial coordinates
    spatial_coords = np.array(flattened_data.spatial.values.tolist())

    # Normalize the data
    mean = np.mean(data_array, axis=0)
    std_dev = np.std(data_array, axis=0)
    normalized_data = (data_array - mean) / std_dev

    time_5 = time.time()
    logger.info(f"Time taken to normalize data: {time_5 - time_4}")

    num_points = spatial_coords.shape[0]
    pairs = list(combinations(range(num_points), 2))  # pairs of indices (i,j) where i != j
    # distance function
    # D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
    # d is Euclidean distance, r is temporal correlation coefficient, a is constant such that the value of the
    # exponential is 0.5, when d=3000 km
    # calculate distances between each pair of grid points
    a = math.sqrt(- (1500 / (math.log(0.5))))

    # compute pairwise distances
    distances = np.array([distance_function(pair, normalized_data, spatial_coords, a) for pair in pairs])

    time_5 = time.time()
    logger.info(f"Time taken to calculate distances: {time_5 - time_4}")

    spatial_indices = flattened_data['spatial'].values  # Coordinates of the grid points
    pairwise_distances = pd.DataFrame(
        {'point1': [spatial_indices[pair[0]] for pair in pairs], 'point2': [spatial_indices[pair[1]] for pair in pairs],
         'distance': distances})

    pairwise_distances.to_pickle("../data/pairwise_distances.pkl")

    # hierarchical clustering
    linkage_matrix = linkage(distances, method='average')

    dendrogram(linkage_matrix)
    plt.show()

# TODO: filter spatially with a symmetric Gaussian filter of half-width 500 km
# TODO: interpolate to 5 degree grid
# TODO: Apply a convolution low-pass filter passing 90% of the amplitude at 24 months to each time series.  (To emphasize inter annual and longer variability)
# TODO: implement distance function between two grid points x_i and x_j - D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
# TODO: d is Euclidean distance, r is temporal correlation coefficient, a is constant such that the value of the exponential is 0.5, when d=3000 km
# TODO: calculate distances between each pair of grid points
# TODO: hierarchical clustering
