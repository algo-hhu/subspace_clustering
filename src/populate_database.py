import asyncio
import math
import time

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger
from prisma import Prisma
from tqdm import tqdm


async def generate_grid_points_and_initial_clusters(sea_level_anomaly_data: xr.Dataset, db: Prisma):
    """
    Processes sea level anomaly data, generates grid points, stores them in the database, and assigns cluster information.
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
    counter = 0
    # create a nan array of the same shape as the sea level data
    ids = {}
    neighbors = {}
    for i in tqdm(range(filtered_sea_level["latitude"].shape[0])):
        counter += 1
        for j in range(filtered_sea_level["longitude"].shape[0]):
            if filtered_sea_level["sla"][:, i, j].isnull().values.any():
                continue
            latitude = float(filtered_sea_level["latitude"].values[i])
            longitude = float(filtered_sea_level["longitude"].values[j])
            timeseries = filtered_sea_level["sla"][:, i, j].values.tolist()
            current_neighbor_ids = []
            try:
                current_cluster = await db.cluster.create(
                    data={
                        "grid_points": {
                            "create": [
                                {"latitude": float(filtered_sea_level["latitude"].values[i]),
                                 "longitude": float(filtered_sea_level["longitude"].values[j]),
                                 "timeseries": filtered_sea_level["sla"][:, i, j].values.tolist()}
                            ]
                        }, "neighbor_ids": []
                    }
                )
            except:
                continue
            ids[(i, j)] = current_cluster.id
            if i > 0:
                if filtered_sea_level["latitude"].values[i - 1] - latitude <= 0.3:
                    try:
                        current_neighbor_ids.append(ids[(i - 1, j)])
                        neighbors[ids[(i - 1, j)]].append(current_cluster.id)
                    except KeyError:
                        pass
            if j > 0:
                if filtered_sea_level["longitude"].values[j - 1] - longitude <= 0.3:
                    try:
                        current_neighbor_ids.append(ids[(i, j - 1)])
                        neighbors[ids[(i, j - 1)]].append(current_cluster.id)
                    except KeyError:
                        pass
            neighbors[current_cluster.id] = current_neighbor_ids
    logger.info(f"Generated {await db.gridpoint.count()} grid points and {await db.cluster.count()} clusters")
    logger.info(f"Establishing neighbor relationships between clusters")
    for cluster_id in neighbors.keys():  # Add neighbors to each cluster
        if neighbors[cluster_id]:
            await db.cluster.update(where={"id": cluster_id}, data={"neighbor_ids": neighbors[cluster_id]})
    return


async def fetch_cluster_pairs_in_batches(db: Prisma, batch_size: int, last_id: str):
    """
    Fetch clusters and associated grid points in batches.
    :param db: Prisma database instance
    :param batch_size: Number of clusters to fetch in each batch
    :param last_id: ID of the last fetched cluster (for pagination)
    :return: List of clusters
    """
    logger.info(f"Fetching clusters in batches of {batch_size}")
    cluster_pairs = []
    clusters = await db.cluster.find_many(
        take=batch_size,
        where={"id": {"gt": last_id}} if last_id else {},
        order={"id": "asc"},
        include={"grid_points": True}
    )
    if not clusters:
        logger.info(f"No cluster found")
        return None
    logger.info(f"Fetched {len(clusters)} clusters")
    for cluster in clusters:
        current_neighbors = cluster.neighbor_ids
        for second_cluster_id in current_neighbors:
            second_cluster = await db.cluster.find_first(where={"id": second_cluster_id}, include={"grid_points": True})
            cluster_pairs.append((cluster, second_cluster))
    return cluster_pairs


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
        db.gridpoint.find_first(where={"id": cluster1.grid_points[0]}),
        db.gridpoint.find_first(where={"id": cluster2.grid_points[0]}),
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


def calculate_difference(cluster1: (Prisma.cluster, Prisma.gridpoint),
                         cluster2: (Prisma.cluster, Prisma.gridpoint), a: float):
    """
    D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
    d is Euclidean distance, r is temporal correlation coefficient, a is constant such that the value of the
    exponential is 0.5, when d=3000 km
    calculate distances between each pair of grid points
    :param cluster2:
    :param cluster1:
    :param a:
    :return:
    """
    grid_point1 = cluster1.grid_points[0]
    grid_point2 = cluster2.grid_points[0]
    # distance in km between two points > using the haversine distance instead of Euclidean, otherwise the error could
    # be substantial
    lat1 = grid_point1.latitude
    long1 = grid_point1.longitude
    lat2 = grid_point2.latitude
    long2 = grid_point2.longitude
    earth_radius = 6371  # km
    lat1, lat2, long1, long2 = map(np.radians, [lat1, lat2, long1, long2])
    delta_phi = lat2 - lat1
    delta_lambda = long2 - long1
    haversine_distance = 2 * earth_radius * np.arcsin(
        np.sqrt(np.sin(delta_phi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lambda / 2) ** 2)
    )

    timeseries1 = grid_point1.timeseries
    timeseries2 = grid_point2.timeseries
    # Pearsons correlation coefficient
    r = np.corrcoef(timeseries1, timeseries2)[0, 1]

    # calculate difference
    difference = 1 - np.exp(-haversine_distance / (2 * a ** 2)) * r
    return cluster1, cluster2, difference


async def write_difference_to_db(db: Prisma, differences):
    """
    Save the calculated difference between two clusters to the database asynchronously.
    :param differences:
    :param db: Prisma database instance
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
    # Only calculate differences for grid points that are neighbors
    # get all grid points from database and calculate differences between them
    # fetch the clusters in 500-element-chunks from the database and calculate differences between them
    logger.info("Calculating initial differences between grid points")
    last_id = None
    batch_size = 500
    cluster_pairs = await fetch_cluster_pairs_in_batches(db, batch_size, last_id)
    logger.info(f"Calculating differences for {len(cluster_pairs)} cluster pairs")
    counter = 1
    time_1 = time.time()
    a = math.sqrt(- (1500 / (math.log(0.5))))
    while cluster_pairs:
        # calculate differences for all pairs of clusters in current_cluster_batch
        results = Parallel(n_jobs=-2)(
            delayed(calculate_difference)(cluster1, cluster2, a) for
            (cluster1, cluster2) in cluster_pairs)
        # Remove None values from the results
        results = [result for result in results if result is not None]
        if not results:
            logger.info(f"No differences calculated for batch {counter}")
        # Save the result to the database
        await write_difference_to_db(db, results)
        logger.info(f"Wrote {await db.difference.count()} differences to the database")
        logger.info(f"Calculated differences for batch {counter}")
        print(f"Took time {time.time() - time_1}")
        last_id = cluster_pairs[-1][0].id
        cluster_pairs = await fetch_cluster_pairs_in_batches(db, batch_size, last_id)
        counter += 1
