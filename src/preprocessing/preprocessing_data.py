import os
import time

import tqdm
import xarray as xr
from loguru import logger

from src.plotting import plot_sla_for_point_in_time
from src.preprocessing import spherical_gauss_filter


def apply_gaussian_filter(sea_level_anomaly_data_set: xr.Dataset, half_width: int):
    """
    Apply a Gaussian filter of half width 500 to the sea level anomaly data
    :param half_width:
    :param sea_level_anomaly_data_set:
    :return:
    """
    spatial_filter = spherical_gauss_filter.SphericalGaussFilter(sea_level_anomaly_data_set.latitude.values,
                                                                 sea_level_anomaly_data_set.longitude.values,
                                                                 half_width)
    sea_level_anomaly_data_set = spatial_filter.filter(sea_level_anomaly_data_set)
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


def filtering(sea_level_anomaly_data: xr.Dataset, out_dir: str, half_width: int):
    """
    Filter the sea level anomaly data
    :param half_width:
    :param out_dir:
    :param sea_level_anomaly_data:
    :return:
    """
    logger.warning("Filtering data - this may take some time, if this is not wanted, set filtering_sla to False")
    logger.info("filtering data")
    # check if longitude is correct (-180 to 180)
    if sea_level_anomaly_data.longitude.max() > 180 or sea_level_anomaly_data.longitude.min() < -180:
        logger.warning(
            "Longitude is not correct, it should range from -180 to 180, try deleting the sea_level_anomaly_data.nc file and rerun the program")

    # filter spatially with a symmetric Gaussian filter of half-width 500 km (here the C$S is transformed to meters using a geocentric CRS EPSG:4978)
    current_time = time.time()
    sea_level_anomaly_data = apply_gaussian_filter(sea_level_anomaly_data, half_width)
    logger.info(f"Time taken for gaussian filtering {time.time() - current_time}")
    current_time = time.time()
    sea_level_anomaly_data["sla"] = (
        sea_level_anomaly_data["sla"]
        .rolling(time=15, center=True, min_periods=1)
        .mean(skipna=True)
    )
    logger.info(f"Time taken for temporal filtering {time.time() - current_time}")
    # save netcdf
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
    sea_level_anomaly_data.to_netcdf("../data/sea_level_anomaly_data_filtered.nc", encoding=encoding,
                                     format="NETCDF4")
    variable_to_plot = "sla"
    plot_sla_for_point_in_time(sea_level_anomaly_data, out_dir, variable_to_plot, name="filtered_sla")

    return sea_level_anomaly_data
