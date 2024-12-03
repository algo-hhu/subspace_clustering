import math
from itertools import combinations

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from prisma import Prisma
from tqdm import tqdm


async def generate_grid_points(sea_level_anomaly_data: xr.Dataset, db: Prisma):
    """
    Generate grid points for each grid point
    :param db:
    :param sea_level_anomaly_data:
    :return:
    """
    sea_level = sea_level_anomaly_data['sla']
    # filter grid points
    valid_points = ~sea_level.isnull().all(dim="time")  # Replace "time" with the dimension to check

    # Filter the dataset to keep only valid grid points
    filtered_sea_level = sea_level_anomaly_data.where(valid_points, drop=True)
    # iterate over all grid points and generate a GridPoint object for each grid point and save it to the database
    # this should be done in parallel using joblib
    # maybe only do this for a certain radius around the point (e.g. 3000 km
    counter = 0
    for i in tqdm(range(filtered_sea_level["latitude"].shape[0])):
        for j in (range(filtered_sea_level["longitude"].shape[0])):
            # TODO: figure out how to handle nan values in the time series, remove beforehand?
            if filtered_sea_level["sla"][:, i, j].isnull().values.any():
                continue
            # create a cluster object that contains a grid point in the database
            try:
                cluster = await db.cluster.create(
                    data={
                        "gridpoints": {
                            "create": [
                                {"latitude": float(filtered_sea_level["latitude"].values[i]),
                                 "longitude": float(filtered_sea_level["longitude"].values[j]),
                                 "timeseries": filtered_sea_level["sla"][:, i, j].values.tolist()}
                            ]
                        }
                    }
                )
            except Exception as e:
                pass

    return


async def fetch_all_clusters(db: Prisma):
    """
    Fetch all cluster ids from the database
    :param db:
    :return:
    """
    # fetch all cluster ids from the database
    entities = await db.cluster.find_many()
    # return [entity['id'] for entity in entities]
    return entities


async def calculate_and_save_difference(db, cluster1, cluster2):
    """
    Calculate and save difference between two grid points that are associated with the clusters
    :param db:
    :param cluster1:
    :param cluster2:
    :return:
    """
    # get grid points that belong to clusters with id1 and id2
    grid_point_1 = cluster1['gridpoints']
    grid_point_2 = cluster2['gridpoints']
    timeseries1 = grid_point_1['timeseries']
    timeseries2 = grid_point_2['timeseries']
    lat1 = grid_point_1['latitude']
    long1 = grid_point_1['longitude']
    lat2 = grid_point_2['latitude']
    long2 = grid_point_2['longitude']
    difference = calculate_difference(lat1, long1, lat2, long2, timeseries1, timeseries2)
    # save diff to database
    await db.difference.create(
        data={
            "value": difference,
            "cluster1": {
                "connect": cluster1['id']
            },
            "cluster2": {
                "connect": cluster2['id']
            }
        }
    )
    return


def calculate_difference(lat1, long1, lat2, long2, timeseries1, timeseries2):
    """
    D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
    d is Euclidean distance, r is temporal correlation coefficient, a is constant such that the value of the
    exponential is 0.5, when d=3000 km
    calculate distances between each pair of grid points
    :param lat1:
    :param long1:
    :param lat2:
    :param long2:
    :param timeseries1:
    :param timeseries2:
    :return:
    """
    # Pearsons correlation coefficient
    r = np.corrcoef(timeseries1, timeseries2)[0, 1]

    # distance in km between two points > using the haversine distance instead of Euclidean, otherwise the error could
    # be substantial
    earth_radius = 6371  # in km
    # convert lat/long from degree to radians and calculate diff
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(long2 - long1)
    # calculate haversine
    # Haversine formula
    b = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi_1) * math.cos(phi_2) *
         math.sin(delta_lambda / 2) ** 2)
    haversine_distance = 2 * earth_radius * math.asin(math.sqrt(b))
    a = math.sqrt(- (1500 / (math.log(0.5))))
    difference = 1 - math.exp(- (haversine_distance / (2 * a ** 2))) * r
    return difference


async def calculate_initial_differences(db: Prisma):
    """
    Calculate initial differences between grid points
    :param db:
    :return:
    """
    # get all grid points from database and calculate differences between them
    # this should be done in parallel using joblib
    clusters = await fetch_all_clusters(db)
    cluster_pairs = list(combinations(clusters, 2))
    await Parallel(n_jobs=-2)(
        delayed(calculate_and_save_difference)(db, cluster1, cluster2) for cluster1, cluster2 in cluster_pairs)
