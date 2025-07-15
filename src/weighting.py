import numpy as np
import xarray


def apply_weights_to_sea_level_anomaly_data(unfiltered_sea_level_anomaly_data: xarray.DataArray) -> xarray.DataArray:
    """
    Apply the cosine of the latitude as weights to the sea level anomaly data.
    :param unfiltered_sea_level_anomaly_data:
    :return:
    """
    weighted_sea_level_anomaly_data = unfiltered_sea_level_anomaly_data.copy(deep=True)
    cosine_weights = np.cos(np.deg2rad(weighted_sea_level_anomaly_data["latitude"]))
    # Apply the weights to the sea level anomaly data
    weighted_sea_level_anomaly_data["sla"] = weighted_sea_level_anomaly_data["sla"] * cosine_weights
    return weighted_sea_level_anomaly_data
