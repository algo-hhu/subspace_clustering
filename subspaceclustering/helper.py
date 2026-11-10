import os

import numpy
import numpy as np
import xarray
import xarray as xr
from loguru import logger

# Variable name used for the cluster labels in all saved/loaded clustering NetCDF files. It is the
# default name xarray assigns to an unnamed DataArray, kept explicit here as the single source of truth.
CLUSTERING_VARIABLE_NAME = "__xarray_dataarray_variable__"


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


def adjust_resolution(resolution: int, resolution_path: str, sea_level_anomaly_data: xr.Dataset) -> xr.Dataset:
    """
    Adjust the resolution of the sea level anomaly data to the desired resolution.
    :param resolution:
    :param resolution_path:
    :param sea_level_anomaly_data:
    :return:
    """
    current_resolution = sea_level_anomaly_data.latitude[1] - sea_level_anomaly_data.latitude[0]
    if resolution != current_resolution:
        if resolution < current_resolution:
            raise ValueError(
                f"Requested resolution {resolution}° is finer than the data's {float(current_resolution)}°; "
                f"upsampling is not supported.")
        if resolution < 1:
            raise ValueError(f"Resolution must be >= 1 degree, got {resolution}.")
        resolution = int(resolution)
        resolution_path = resolution_path + f"_{resolution}_degree.nc"
        # interpolate the data to the desired resolution
        if not os.path.exists(resolution_path):
            logger.info(f"Interpolating sea level anomaly data to {resolution} degree resolution")
            os.makedirs(os.path.dirname(resolution_path), exist_ok=True)
            sea_level_anomaly_data = sea_level_anomaly_data.interp(latitude=range(-90, 91, int(resolution)),
                                                                   longitude=range(-180, 180, int(resolution)))
            # save the interpolated data
            save_xarray_dataset(resolution_path, sea_level_anomaly_data)
        else:
            sea_level_anomaly_data = xr.open_dataset(resolution_path)
    return sea_level_anomaly_data


def extract_clusters_from_xarray_dataset(clustering: xarray.Dataset, min_lat: float, min_lon: float, resolution: float,
                                         sla_data: np.ndarray) -> tuple[
    dict[int, list[tuple[float, float]]], dict[int, list[tuple[float, float]]]]:
    """
    Extract the original clusters from the clustering data
    :param sla_data:
    :param clustering:
    :param min_lat:
    :param min_lon:
    :param resolution:
    :return: cluster_id_to_lat_lon_pairs: dict[int, list[tuple[float, float]]], cluster_id_to_grid_point_id: dict[
    int, list[tuple[int, int]]]
    """
    # nan mask
    non_nan_mask = ~np.isnan(sla_data).any(axis=0)
    # apply nan mask to clustering data (copy first so we don't mutate the input dataset in place)
    cluster_data = clustering[CLUSTERING_VARIABLE_NAME].values.copy()
    # assign nans where there are all nans in the cluster data
    cluster_data[~non_nan_mask] = np.nan
    unique_numbers, counts = np.unique(cluster_data, return_counts=True)

    unique_numbers = unique_numbers[~np.isnan(unique_numbers)]
    # 2D array of size lat x lon that contains only the lat or lon values at each point
    extended_lats = np.tile(clustering["latitude"].values[:, np.newaxis],
                            (1, clustering["longitude"].values.shape[0]))
    extended_lons = np.tile(clustering["longitude"].values[np.newaxis, :],
                            (clustering["latitude"].values.shape[0], 1))
    cluster_id_to_lat_lon_pairs = {}
    cluster_id_to_grid_point_id = {}
    for cluster_id in unique_numbers:
        if np.isnan(cluster_id):
            continue
        # find lat/lon pairs for each cluster
        current_cluster_mask = cluster_data == cluster_id
        filtered_lats = extended_lats[current_cluster_mask]
        filtered_lons = extended_lons[current_cluster_mask]
        lat_lon_pairs = list(zip(filtered_lats, filtered_lons))
        cluster_id_to_lat_lon_pairs[cluster_id] = lat_lon_pairs
        cluster_id_to_grid_point_id[cluster_id] = []
        for lat_lon_pair in lat_lon_pairs:
            id_x, id_y = lat_lon_to_index(lat_lon_pair[0], lat_lon_pair[1], min_lat, min_lon, resolution)
            cluster_id_to_grid_point_id[cluster_id].append((id_x, id_y))
    return cluster_id_to_lat_lon_pairs, cluster_id_to_grid_point_id


def save_clustering(clustering_dict: dict[int, list[tuple[float, float]]], out_dir: str,
                    sea_level_anomaly_data: xarray.Dataset, filename: str) -> None:
    """
    Save the clustering results to a netCDF file.
    :param filename:
    :param clustering_dict:
    :param out_dir:
    :param sea_level_anomaly_data:
    :return:
    """
    cluster_data = numpy.zeros((sea_level_anomaly_data.latitude.size, sea_level_anomaly_data.longitude.size))
    cluster_number = 0
    for cluster in clustering_dict.keys():
        for grid_point in clustering_dict[cluster]:
            # get index of lat long in sea_level_anomaly_data
            lat_index = numpy.where(sea_level_anomaly_data.latitude.values == grid_point[0])[0][0]
            long_index = numpy.where(sea_level_anomaly_data.longitude.values == grid_point[1])[0][0]
            cluster_data[lat_index, long_index] = cluster_number
        cluster_number += 1
    cluster_data = xarray.DataArray(cluster_data, dims=["latitude", "longitude"], name=CLUSTERING_VARIABLE_NAME)
    cluster_data = cluster_data.assign_coords(latitude=sea_level_anomaly_data.latitude,
                                              longitude=sea_level_anomaly_data.longitude)
    cluster_data.to_netcdf(f"{out_dir}/{filename}.nc")
