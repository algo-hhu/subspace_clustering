import os
from os import mkdir

import xarray as xr
from loguru import logger


def index_to_lat_lon(x, y, lat_min, lon_min, resolution) -> (float, float):
    """
    Convert an index to a latitude and longitude
    :param x:
    :param y:
    :param lat_min:
    :param lon_min:
    :param resolution:
    :return:
    """
    lat = lat_min + x * resolution
    lon = lon_min + y * resolution
    return lat, lon


def lat_lon_to_index(lat, lon, lat_min, lon_min, resolution) -> (int, int):
    """
    Convert a latitude and longitude to an index
    :param lat:
    :param lon:
    :param lat_min:
    :param lon_min:
    :param resolution:
    :return:
    """
    x = int((lat - lat_min) / resolution)
    y = int((lon - lon_min) / resolution)
    return x, y


def save_xarray_dataset(out_file_path, sea_level_anomaly_data):
    """
    Save the filtered sea level anomaly data to a netCDF file and plot it
    :param out_file_path:
    :param sea_level_anomaly_data:
    :return:
    """
    # Get the actual shape of the variable
    shape = sea_level_anomaly_data['sla'].shape

    # Ensure chunk sizes do not exceed dimension sizes
    safe_chunks = tuple(min(c, s) for c, s in zip((73, 144, 288), shape))

    # save to netcdf
    encoding = {
        'sla': {
            'zlib': True,  # Enable compression
            'complevel': 4,  # Compression level (1-9, trade-off between speed and compression ratio)
            'shuffle': True,  # Improve compression efficiency
            'dtype': 'float32',  # Convert from float64 to float32 to save space (optional)
            'chunksizes': safe_chunks,  # Use the same efficient chunking as in the smaller dataset
            '_FillValue': -2147483648,  # Match fill value from the smaller dataset
            'scale_factor': 0.0001  # Match scale factor for consistency
        }
    }
    sea_level_anomaly_data.to_netcdf(out_file_path, encoding=encoding,
                                     format="NETCDF4")


async def adjust_resolution(resolution: int, resolution_path: str, sea_level_anomaly_data: xr.Dataset) -> xr.Dataset:
    """
    Adjust the resolution of the sea level anomaly data to the desired resolution.
    :param resolution:
    :param resolution_path:
    :param sea_level_anomaly_data:
    :return:
    """
    if resolution != sea_level_anomaly_data.latitude[1] - sea_level_anomaly_data.latitude[0]:
        if resolution < sea_level_anomaly_data.latitude[1] - sea_level_anomaly_data.latitude[0]:
            logger.warning("The desired resolution is smaller than the current resolution. This is not supported.")
            exit()
        if resolution < 1:
            logger.warning("The desired resolution is smaller than 1 degree. This is not supported.")
            exit()
        # interpolate the data to the desired resolution
        if not os.path.exists(resolution_path):
            logger.info(f"Interpolating sea level anomaly data to {resolution} degree resolution")
            if not os.path.exists("../output/resolutions"):
                mkdir("../output/resolutions")
            sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=range(-90, 91, resolution),
                                                                   longitude=range(-180, 180, resolution))
            # save the interpolated data
            save_xarray_dataset(resolution_path, sea_level_anomaly_data)
        else:
            sea_level_anomaly_data = xr.open_dataset(resolution_path)
    return sea_level_anomaly_data
