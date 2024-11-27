import math
import os
import time
import uuid

import geopandas
import numpy as np
import tqdm
import xarray as xr
from loguru import logger
from scipy.ndimage import gaussian_filter

from src.plotting import plot_sla_for_point_in_time


def calculate_sigma_per_latitude(half_width: int, latitude: float, grid_cell_size: float):
    """
    Given the latitude of a point, calculate the sigma for the Gaussian filter
    :param half_width:
    :param latitude:
    :return:
    """
    km_per_degree_latitude = 111
    km_per_degree_longitude = 111 * math.cos(math.radians(latitude))
    sigma_latitude = half_width / km_per_degree_latitude / grid_cell_size
    sigma_longitude = half_width / km_per_degree_longitude / grid_cell_size
    return (sigma_latitude + sigma_longitude)


def apply_gaussian_filter(sea_level_anomaly_data_set: xr.Dataset, half_width: int):
    """
    Apply a Gaussian filter of half width 500 to the sea level anomaly data
    :param half_width:
    :param sea_level_anomaly_data:
    :return:
    """
    # Todo: check again if this transformation is needed.
    # # change CRS to geocentric CRS EPSG:3857
    # transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    # # Extract latitude and longitude values
    # latitudes = sea_level_anomaly_data["latitude"].values
    # longitudes = sea_level_anomaly_data["longitude"].values
    # latitudes = latitudes.flatten()
    # longitudes = longitudes.flatten()
    # transformed_coords = np.array([transformer.transform(lat, lon) for lat, lon in zip(latitudes, longitudes)])
    # transformed_coords_x = transformed_coords[:, 0].reshape(sea_level_anomaly_data["latitude"].shape)
    # transformed_coords_y = transformed_coords[:, 1].reshape(sea_level_anomaly_data["longitude"].shape)
    #
    # # Assign the transformed coordinates back to the dataset
    # sea_level_anomaly_data["transformed_coords_x"] = xr.DataArray(transformed_coords_x,
    #                                                               dims=sea_level_anomaly_data["latitude"].dims)
    # sea_level_anomaly_data["transformed_coords_y"] = xr.DataArray(transformed_coords_y,
    #                                                               dims=sea_level_anomaly_data["longitude"].dims)

    grid_cell_size = 0.25
    sea_level_data = sea_level_anomaly_data_set['sla']

    filtered_sea_level = np.zeros_like(sea_level_data)
    # Loop over all latitudes
    for time in tqdm.tqdm(range(sea_level_data.shape[0])):
        current_slice = sea_level_data[time, :, :].values
        for i in range(current_slice.shape[0]):
            current_latitude = sea_level_data.coords['latitude'][i].item()
            current_sigma = calculate_sigma_per_latitude(half_width, current_latitude, grid_cell_size)
            # Loop over all longitudes
            for j in range(current_slice.shape[1]):
                current_longitude = sea_level_data.coords['longitude'][j].item()
                filtered_sea_level[time, i, j] = gaussian_filter(current_slice[i, j], current_sigma)
    sea_level_anomaly_data_set['filtered_sla'] = xr.DataArray(filtered_sea_level, dims=sea_level_data.dims)
    return sea_level_anomaly_data_set


def read_satellite_data(data_directory: str):
    """

    :param data_directory:
    :return:
    """
    data_list = [f for f in os.listdir(data_directory) if f.endswith(".nc")]
    data_list.sort()
    first_element = True
    for data_file in tqdm.tqdm(data_list):
        file_path = os.path.join(data_directory, data_file)
        if os.path.exists(file_path):
            with xr.open_dataset(file_path) as dataset:
                if first_element:
                    first_element = False
                    satellite_altimeter_data = dataset
                else:
                    satellite_altimeter_data = xr.concat([satellite_altimeter_data, dataset], dim="time")
        else:
            logger.warning(f"File {file_path} does not exist")
    return satellite_altimeter_data


def filtering(sea_level_anomaly_data: xr.Dataset, time_2: float, out_dir: str):
    """
    Filter the sea level anomaly data
    :param out_dir:
    :param time_2:
    :param sea_level_anomaly_data:
    :return:
    """
    # check if longitude is correct (-180 to 180)
    if sea_level_anomaly_data.longitude.max() > 180 or sea_level_anomaly_data.longitude.min() < -180:
        logger.warning(
            "Longitude is not correct, it should range from -180 to 180, try deleting the sea_level_anomaly_data.nc file and rerun the program")
    time_3 = time.time()
    logger.info(f"Time taken to check longitude: {time_3 - time_2}")
    if not os.path.exists("../data/sea_level_anomaly_data_filtered.nc"):
        # filter spatially with a symmetric Gaussian filter of half-width 500 km (here the C$S is transformed to meters using a geocentric CRS EPSG:4978)
        sea_level_anomaly_data = apply_gaussian_filter(sea_level_anomaly_data, 500)

        time_4 = time.time()
        logger.info(f"Time taken to apply Gaussian filter: {time_4 - time_3}")
        variable_to_plot = "filtered_sla"
        plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, variable_to_plot, name="filtered")
        # interpolate to 5 degree grid
        sea_level_anomaly_data_5_degree_grid = sea_level_anomaly_data.interp(latitude=range(-90, 91, 5),
                                                                             longitude=range(-180, 180, 5))
        time_5 = time.time()
        logger.info(f"Time taken to interpolate to 5 degree grid: {time_5 - time_4}")
        plot_sla_for_point_in_time(sea_level_anomaly_data_5_degree_grid, out_dir, variable_to_plot,
                                   name="5_degree_grid_filtered")
        time_6 = time.time()
        logger.info(f"Time taken to plot sea level anomaly for one point in time: {time_6 - time_5}")
        sea_level_anomaly_data.to_netcdf("../data/sea_level_anomaly_data_filtered.nc")
    else:
        sea_level_anomaly_data_5_degree_grid = xr.open_dataset("../data/sea_level_anomaly_data_filtered.nc")
        time_4 = time.time()
        logger.info(f"Time taken to read filtered data: {time_4 - time_3}")

    return sea_level_anomaly_data_5_degree_grid


def distance_function(pair, data_array, spatial_coords, a):
    """
    Calculate the distance function between two points D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
    :param a:
    :param spatial_coords:
    :param data_array:
    :param pair:
    :return:
    """
    i, j = pair

    # Pearson correlation coefficient
    r = np.corrcoef(data_array[:, i], data_array[:, j])[0, 1]
    # Euclidean distance
    d = np.linalg.norm(spatial_coords[i] - spatial_coords[j])

    return 1 - np.exp(-d / (2 * a ** 2)) * r
