import os
import time

import tqdm
import xarray as xr
from loguru import logger

from src import helper, plotting
from src.helper import save_xarray_dataset, adjust_resolution
from src.preprocessing import spherical_gauss_filter


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


def filtering(sea_level_anomaly_data: xr.Dataset, filtered_data_path: str, half_width: int):
    """
    Filter the sea level anomaly data
    :param filtered_data_path:
    :param half_width:
    :param sea_level_anomaly_data:
    :return:
    """
    logger.warning("Filtering data - this may take some time, if this is not wanted, set filtering_sla to False")
    # check if longitude is correct (-180 to 180)
    if sea_level_anomaly_data.longitude.max() > 180 or sea_level_anomaly_data.longitude.min() < -180:
        logger.warning(
            "Longitude is not correct, it should range from -180 to 180, try deleting the sea_level_anomaly_data.nc "
            "file and rerun the program")

    # filter spatially with a symmetric Gaussian filter of half-width 500 km (here the CRS is transformed to meters
    # using a geocentric CRS EPSG:4978)
    # and temporally with a low-pass filter of 15 months
    sea_level_anomaly_data = apply_filters(sea_level_anomaly_data, half_width)
    save_xarray_dataset(filtered_data_path, sea_level_anomaly_data)
    return sea_level_anomaly_data


def apply_filters(sea_level_anomaly_data: xr.Dataset, half_width: int):
    """
    Apply a Gaussian filter of half width 500 to the sea level anomaly data and a temporal low-pass filter of 15 months
    :param sea_level_anomaly_data:
    :param half_width:
    :return:
    """
    nan_mask = sea_level_anomaly_data["sla"].isnull()
    current_time = time.time()
    spatial_filter = spherical_gauss_filter.SphericalGaussFilter(sea_level_anomaly_data.latitude.values,
                                                                 sea_level_anomaly_data.longitude.values,
                                                                 half_width)
    sea_level_anomaly_data_set = spatial_filter.parallelized_filter(sea_level_anomaly_data)
    logger.info(f"Time taken for gaussian filtering {time.time() - current_time}")
    current_time = time.time()
    # Apply temporal low-pass filter
    sea_level_anomaly_data["sla"] = (
        sea_level_anomaly_data["sla"]
        .rolling(time=15, center=True, min_periods=1)
        .mean(skipna=True)
    )
    # Restore original NaN locations
    sea_level_anomaly_data["sla"] = sea_level_anomaly_data["sla"].where(~nan_mask)
    logger.info(f"Time taken for temporal filtering {time.time() - current_time}")
    return sea_level_anomaly_data_set


def start_preprocessing(global_settings, variable_to_plot: str):
    """
    Read sea level anomaly data and preprocess it according to the settings
    :param subspace_clustering_settings:
    :param global_settings:
    :param variable_to_plot:
    :return:
    """
    # satellite altimetry data is from 1993 to 2023, latitude: -89.875 to 89.875, longitude: 0.125 to 359.875
    # There are 720 latitude points and 1440 longitude points => 1036800 grid points (the resolution is 0.25 degrees)
    # Merge the data from all files into one xarray dataset
    if not os.path.exists(f"{global_settings.data_path}/sea_level_anomaly_data.nc"):
        logger.info(
            f"Reading sea level anomaly data from files in {global_settings.sea_level_anomaly_data_download_path}")
        unfiltered_sea_level_anomaly_data = read_satellite_data(
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
    unprocessed_sea_level_anomaly_data = unfiltered_sea_level_anomaly_data.copy()
    out_dir = global_settings.output_path
    # filtering
    if global_settings.filtering_sla:  # apply Gaussian
        out_dir = f"{out_dir}/filter_{global_settings.half_width}"
        # filter & temporal low-pass filter
        if not os.path.exists(global_settings.filtered_data_path):
            # filter spatially with a symmetric Gaussian filter of half-width 500 km
            sea_level_anomaly_data = filtering(unfiltered_sea_level_anomaly_data,
                                               global_settings.filtered_data_path,
                                               global_settings.half_width)
            # plot
            plotting.plot_sla_for_point_in_time(sea_level_anomaly_data, global_settings.output_path, variable_to_plot,
                                                name=f"gaussian_filtered_{global_settings.half_width}")
        else:
            sea_level_anomaly_data = xr.open_dataset(global_settings.filtered_data_path)
        # adjust resolution
        out_dir = f"{out_dir}/{global_settings.resolution}_degree_grid"
        resolution_path = f"../output/resolutions/sea_level_anomaly_data_filtered_{global_settings.half_width}"
        sea_level_anomaly_data = adjust_resolution(global_settings.resolution, resolution_path, sea_level_anomaly_data)
    else:
        out_dir = f"{out_dir}/no_filtering"
        out_dir = f"{out_dir}/{global_settings.resolution}_degree_grid"
        resolution_path = (f"../output/resolutions/sea_level_anomaly_data_no_filter")
        # check for correct resolution
        sea_level_anomaly_data = adjust_resolution(global_settings.resolution, resolution_path,
                                                   unfiltered_sea_level_anomaly_data)
    resolution_path = f"../output/resolutions/sea_level_anomaly_data_no_filter"
    unfiltered_sea_level_anomaly_data = adjust_resolution(global_settings.resolution, resolution_path,
                                                          unfiltered_sea_level_anomaly_data)
    return out_dir, sea_level_anomaly_data, unfiltered_sea_level_anomaly_data, unprocessed_sea_level_anomaly_data
