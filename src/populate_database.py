import uuid
import xarray as xr

from src.grid_point import GridPoint


def generate_grid_points(sea_level_anomaly_data: xr.Dataset):
    """
    Generate grid points for each grid point
    :param sea_level_anomaly_data:
    :return:
    """
    # iterate over all grid points and generate a GridPoint object for each grid point and save it to the database
    # this should be done in parallel using joblib
    # maybe only do this for a certain radius around the point (e.g. 3000 km
    for i in range(sea_level_anomaly_data["latitude"].shape[0]):
        for j in range(sea_level_anomaly_data["longitude"].shape[0]):
            GridPoint(id=uuid.uuid4(), latitude=sea_level_anomaly_data["latitude"][i].item(),
                      longitude=sea_level_anomaly_data["longitude"][j].item(),
                      sea_level_anomaly_values=sea_level_anomaly_data["sla"][:, i, j].values)
            # save to database
    pass
