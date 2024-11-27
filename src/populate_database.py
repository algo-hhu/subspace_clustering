import xarray as xr
from prisma import Prisma


async def generate_grid_points(sea_level_anomaly_data: xr.Dataset, db: Prisma):
    """
    Generate grid points for each grid point
    :param db:
    :param sea_level_anomaly_data:
    :return:
    """
    # iterate over all grid points and generate a GridPoint object for each grid point and save it to the database
    # this should be done in parallel using joblib
    # maybe only do this for a certain radius around the point (e.g. 3000 km
    counter = 0
    for i in range(sea_level_anomaly_data["latitude"].shape[0]):
        if counter == 10:
            break
        counter2 = 0
        for j in range(sea_level_anomaly_data["longitude"].shape[0]):
            if counter2 == 10:
                break
            # TODO: figure out how to handle nan values in the time series, remove beforehand?
            print((sea_level_anomaly_data["sla"][:, i, j].values.tolist()))
            # create a grid point object in the database
            grid_point = await db.gridpoint.create(
                data={
                    "latitude": float(sea_level_anomaly_data["latitude"].values[i]),
                    "longitude": float(sea_level_anomaly_data["longitude"].values[j]),
                    "timeseries": sea_level_anomaly_data["sla"][:, i, j].values
                }
            )
    pass
