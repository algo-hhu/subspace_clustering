import numpy as np
import xarray


def start_clustering(sea_level_anomaly_data: xarray.Dataset, k: [int]):
    """
    start hierarchical neighborhood clustering
    :param sea_level_anomaly_data:
    :param k:
    :return:
    """
    k = sorted(k)
    data = sea_level_anomaly_data["sla"].values
    nan_mask = sea_level_anomaly_data["sla"].isnull().values
    nan_mask = nan_mask[0, :, :]
    clusters = np.full((sea_level_anomaly_data.latitude.size, sea_level_anomaly_data.longitude.size), fill_value=-1,
                       dtype=int)
    lat_lon_to_idx = {(lat, lon): (i, j) for i, lat in enumerate(sea_level_anomaly_data.latitude.values) for j, lon in
                      enumerate(sea_level_anomaly_data.longitude.values)}
    idx_to_lat_lon = {(i, j): (lat, lon) for (lat, lon), (i, j) in lat_lon_to_idx.items()}
    distances = {}
