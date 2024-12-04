import asyncio
import math
import time
from itertools import combinations, product

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger
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


async def fetch_clusters_in_batches(db: Prisma, batch_size: int, last_id: str):
    """
    Fetch clusters and associated grid points in batches.
    :param db: Prisma database instance
    :param batch_size: Number of clusters to fetch in each batch
    :param last_id: ID of the last fetched cluster (for pagination)
    :return: List of clusters
    """
    return await db.cluster.find_many(
        take=batch_size,
        where={"id": {"gt": last_id}} if last_id else {},
        order={"id": "asc"},
        include={"gridpoints": True}
    )


async def calculate_and_save_difference(db, cluster1, cluster2, executor, a):
    """
    Calculate and save difference between two grid points that are associated with the clusters
    :param executor:
    :param db:
    :param cluster1:
    :param cluster2:
    :return:
    """
    # get grid points that belong to clusters with id1 and id2
    grid_point_1, grid_point_2 = await asyncio.gather(
        db.gridpoint.find_first(where={"clusters": {"some": {"id": cluster1.id}}}),
        db.gridpoint.find_first(where={"clusters": {"some": {"id": cluster2.id}}}),
    )
    timeseries1 = grid_point_1.timeseries
    timeseries2 = grid_point_2.timeseries
    lat1 = grid_point_1.latitude
    long1 = grid_point_1.longitude
    lat2 = grid_point_2.latitude
    long2 = grid_point_2.longitude
    # Offload the CPU-bound calculation to the process pool
    loop = asyncio.get_event_loop()
    difference = await loop.run_in_executor(
        executor,
        calculate_difference,
        lat1, long1, lat2, long2, timeseries1, timeseries2, a
    )

    # Save the result to the database
    await write_difference_to_db(db, cluster1, cluster2, difference)
    return


def calculate_difference(cluster1, cluster2, a: float):
    """
    D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
    d is Euclidean distance, r is temporal correlation coefficient, a is constant such that the value of the
    exponential is 0.5, when d=3000 km
    calculate distances between each pair of grid points
    :param a:
    :param lat1:
    :param long1:
    :param lat2:
    :param long2:
    :param timeseries1:
    :param timeseries2:
    :return:
    """
    grid_point1 = cluster1.gridpoints[0]
    grid_point2 = cluster2.gridpoints[0]
    timeseries1 = grid_point1.timeseries
    timeseries2 = grid_point2.timeseries
    lat1 = grid_point1.latitude
    long1 = grid_point1.longitude
    lat2 = grid_point2.latitude
    long2 = grid_point2.longitude
    # Pearsons correlation coefficient
    r = np.corrcoef(timeseries1, timeseries2)[0, 1]

    # distance in km between two points > using the haversine distance instead of Euclidean, otherwise the error could
    # be substantial
    earth_radius = 6371  # km
    lat1, lat2, long1, long2 = map(np.radians, [lat1, lat2, long1, long2])
    delta_phi = lat2 - lat1
    delta_lambda = long2 - long1
    haversine_distance = 2 * earth_radius * np.arcsin(
        np.sqrt(np.sin(delta_phi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lambda / 2) ** 2)
    )
    # calculate difference
    difference = 1 - np.exp(-haversine_distance / (2 * a ** 2)) * r
    return cluster1, cluster2, difference


async def write_difference_to_db(db: Prisma, differences):
    """
    Save the calculated difference between two clusters to the database asynchronously.
    :param db: Prisma database instance
    :param cluster1: First cluster
    :param cluster2: Second cluster
    :param difference: Difference between the two clusters
    """
    data = [
        {"difference": diff, "cluster1Id": c1.id, "cluster2Id": c2.id}
        for c1, c2, diff in differences
    ]
    await db.difference.create_many(data, skip_duplicates=True)


async def calculate_initial_differences(db: Prisma):
    """
    Calculate initial differences between grid points

    :param db:
    :return:
    """
    # get all grid points from database and calculate differences between them
    # this should be done in parallel using joblib
    # fetch the clusters in 100-element-chunks from the database and calculate differences between them
    logger.info("Calculating initial differences between grid points")
    last_id = None
    batch_size = 500
    current_cluster_batch = await fetch_clusters_in_batches(db, batch_size, last_id)
    counter = 0
    time_1 = time.time()
    a = math.sqrt(- (1500 / (math.log(0.5))))
    while current_cluster_batch:
        logger.info(f"Calculated differences for batch {counter}")
        print(f"Took time {time.time() - time_1}")
        # calculate differences for all pairs of clusters in current_cluster_batch
        cluster_pairs = list(combinations(current_cluster_batch, 2))
        results = Parallel(n_jobs=-2)(
            delayed(calculate_difference)(cluster1, cluster2, a) for cluster1, cluster2 in cluster_pairs)
        # Save the result to the database
        await write_difference_to_db(db, results)
        last_id = current_cluster_batch[-1].id
        comparison_batch = await fetch_clusters_in_batches(db, batch_size, last_id)
        while comparison_batch:
            cluster_pairs = list(product(current_cluster_batch, comparison_batch))
            results = Parallel(n_jobs=-2)(
                delayed(calculate_difference)(cluster1, cluster2, a) for cluster1, cluster2 in cluster_pairs)
            # Save the result to the database
            await write_difference_to_db(db, results)
            last_id = comparison_batch[-1].id
            comparison_batch = await fetch_clusters_in_batches(db, batch_size, last_id)
