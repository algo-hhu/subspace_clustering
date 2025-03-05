import math

import numpy as np
import xarray as xr
from loguru import logger
from prisma import Prisma

from src.plotting import save_and_plot_clusters

NO_DIFF_COUNTER_BOTH_NEIGHBORS = 0
NO_DIFF_COUNTER = 0
DIFF_COUNTER = 0


async def ensure_bidirectional_neighbors(db: Prisma, cluster_id: int, neighbor_id: int):
    """
    Ensure that two clusters are properly set up as neighbors of each other
    """
    # Get both clusters
    cluster = await db.cluster.find_first(where={"id": cluster_id})
    neighbor = await db.cluster.find_first(where={"id": neighbor_id})

    if not cluster or not neighbor:
        return False

    # Update neighbor relationships in both directions
    cluster_neighbors = list(set(cluster.neighbor_ids))  # Remove any existing duplicates
    neighbor_neighbors = list(set(neighbor.neighbor_ids))

    # Add bidirectional relationship if it doesn't exist
    if neighbor_id not in cluster_neighbors:
        cluster_neighbors.append(neighbor_id)
    if cluster_id not in neighbor_neighbors:
        neighbor_neighbors.append(cluster_id)

    # Update both clusters
    await db.cluster.update(
        where={"id": cluster_id},
        data={"neighbor_ids": cluster_neighbors}
    )
    await db.cluster.update(
        where={"id": neighbor_id},
        data={"neighbor_ids": neighbor_neighbors}
    )
    return True


async def recalculate_difference(db: Prisma, new_cluster, cluster1, cluster2, grid_points1,
                                 grid_points2):
    """
    Recalculate differences between new cluster and its neighbors
    :param db:
    :param grid_points1:
    :param grid_points2:
    :param new_cluster:
    :param cluster1:
    :param cluster2:
    :param new_grid_points:
    :return:
    """
    global NO_DIFF_COUNTER
    global NO_DIFF_COUNTER_BOTH_NEIGHBORS
    all_neighbors = new_cluster.neighbor_ids
    neighbors_1 = cluster1.neighbor_ids
    neighbors_2 = cluster2.neighbor_ids
    new_difference = None
    for neighbor_id in all_neighbors:
        # find first cluster which has the neighbor_id
        current_neighbor = await db.cluster.find_first(where={"id": neighbor_id}, include={"grid_points": True})
        # check if the currently considered neighbor is a neighbor of both clusters that are being merged
        if neighbor_id in neighbors_1 and neighbor_id in neighbors_2:
            # fetch difference between cluster1 and neighbor (the order of the clusters in the difference table is not certain, so we need to check both)
            difference1 = await db.difference.find_first(
                where={"cluster1Id": cluster1.id, "cluster2Id": neighbor_id}
            )
            if not difference1:
                difference1 = await db.difference.find_first(
                    where={"cluster1Id": neighbor_id, "cluster2Id": cluster1.id}
                )
            # fetch difference between cluster2 and neighbor (the order of the clusters in the difference table is not certain, so we need to check both)
            difference2 = await db.difference.find_first(
                where={"cluster1Id": cluster2.id, "cluster2Id": neighbor_id}
            )
            if not difference2:
                difference2 = await db.difference.find_first(
                    where={"cluster1Id": neighbor_id, "cluster2Id": cluster2.id}
                )
            if not difference1 or not difference2:
                NO_DIFF_COUNTER_BOTH_NEIGHBORS += 1
                logger.error("Difference not found")
                logger.info(f"Clusters remaining {await db.cluster.count()}")
                exit()
                continue
            new_difference = (difference1.difference * len(grid_points1) + difference2.difference * len(
                grid_points2)) / (len(grid_points1) + len(grid_points2))
            await db.difference.delete(where={"id": difference1.id})
            await db.difference.delete(where={"id": difference2.id})
            # update neighbor_ids of the neighbor_cluster to be the new merged cluster and remove the old clusters
            neighbor_neighbors = current_neighbor.neighbor_ids
            neighbor_neighbors.remove(cluster1.id)
            neighbor_neighbors.remove(cluster2.id)
            if new_cluster.id not in neighbor_neighbors:
                neighbor_neighbors.append(new_cluster.id)
            if neighbor_id not in new_cluster.neighbor_ids:
                updated_neighbors = new_cluster.neighbor_ids + [neighbor_id]
                await db.cluster.update(
                    where={"id": new_cluster.id},
                    data={"neighbor_ids": updated_neighbors}
                )
                new_cluster.neighbor_ids = updated_neighbors  # Update local object

            await db.cluster.update(where={"id": current_neighbor.id}, data={"neighbor_ids": neighbor_neighbors})
        elif neighbor_id in neighbors_1 and not neighbor_id in neighbors_2:
            new_difference = await recalculate_difference_if_one_neighbor(cluster1, current_neighbor, db, grid_points1,
                                                                          grid_points2, neighbor_id)
            neighbor_neighbors = current_neighbor.neighbor_ids
            neighbor_neighbors.remove(cluster1.id)
            if new_cluster.id not in neighbor_neighbors:
                neighbor_neighbors.append(new_cluster.id)
            if neighbor_id not in new_cluster.neighbor_ids:
                updated_neighbors = new_cluster.neighbor_ids + [neighbor_id]
                await db.cluster.update(
                    where={"id": new_cluster.id},
                    data={"neighbor_ids": updated_neighbors}
                )
                new_cluster.neighbor_ids = updated_neighbors  # Update local object
            await db.cluster.update(where={"id": current_neighbor.id}, data={"neighbor_ids": neighbor_neighbors})

        elif neighbor_id in neighbors_2 and not neighbor_id in neighbors_1:
            new_difference = await recalculate_difference_if_one_neighbor(cluster2, current_neighbor, db, grid_points2,
                                                                          grid_points1, neighbor_id)
            neighbor_neighbors = current_neighbor.neighbor_ids
            neighbor_neighbors.remove(cluster2.id)
            if neighbor_id not in new_cluster.neighbor_ids:
                updated_neighbors = new_cluster.neighbor_ids + [neighbor_id]
                await db.cluster.update(
                    where={"id": new_cluster.id},
                    data={"neighbor_ids": updated_neighbors}
                )
                new_cluster.neighbor_ids = updated_neighbors  # Update local object
            if new_cluster.id not in neighbor_neighbors:
                neighbor_neighbors.append(new_cluster.id)

            await db.cluster.update(where={"id": current_neighbor.id}, data={"neighbor_ids": neighbor_neighbors})
        else:
            logger.error("Neighbor not found")

        if not new_difference:
            NO_DIFF_COUNTER += 1
            logger.error("Difference not found")
            logger.info(f"Clusters remaining {await db.cluster.count()}")
            exit()
            continue
        await db.difference.create(
            data={
                "cluster1Id": new_cluster.id,
                "cluster2Id": current_neighbor.id,
                "difference": new_difference
            }
        )
        global DIFF_COUNTER
        DIFF_COUNTER += 1
    return


async def recalculate_difference_if_one_neighbor(cluster1, current_neighbor, db, grid_points1, grid_points2,
                                                 neighbor_id):
    """
    Recalculate difference between new cluster and its neighbors if only one part of cluster was a neighbor before
    :param cluster1:
    :param current_neighbor:
    :param db:
    :param grid_points1:
    :param grid_points2:
    :param neighbor_id:
    :return:
    """

    difference1 = await db.difference.find_first(
        where={"cluster1Id": cluster1.id, "cluster2Id": neighbor_id}
    )
    if not difference1:
        difference1 = await db.difference.find_first(
            where={"cluster1Id": neighbor_id, "cluster2Id": cluster1.id}
        )
    if not difference1:
        # Calculate new difference instead of returning None
        difference1_value = 0
        grid_point_pairs = [(gp1, gp2) for gp1 in grid_points1 for gp2 in current_neighbor.grid_points]
        for gp1, gp2 in grid_point_pairs:
            difference1_value += distance_function(gp1.latitude, gp1.longitude, gp1.timeseries,
                                                   gp2.latitude, gp2.longitude, gp2.timeseries)
        difference1_value /= len(grid_point_pairs)
        # write diff to db
        difference1 = await db.difference.create(
            data={
                "cluster1Id": cluster1.id,
                "cluster2Id": current_neighbor.id,
                "difference": difference1_value
            })

    # calculate difference2 as the average difference between all grid points in cluster2 and all gridpoints in the neighbor-cluster
    difference2 = 0
    grid_point_pairs = [(grid_point1, grid_point2) for grid_point1 in grid_points2 for grid_point2 in
                        current_neighbor.grid_points]
    sum_difference = 0
    for grid_point1, grid_point2 in grid_point_pairs:
        sum_difference += distance_function(grid_point1.latitude, grid_point1.longitude, grid_point1.timeseries,
                                            grid_point2.latitude, grid_point2.longitude, grid_point2.timeseries)
    difference2 = sum_difference / len(grid_point_pairs)
    new_difference = (difference1.difference * len(grid_points1) + difference2 * len(grid_points2)) / (
            len(grid_points1) + len(grid_points2)
    )
    # remove old difference from db, also delete relation
    await db.difference.delete(where={"id": difference1.id})
    return new_difference


def distance_function(lat1: float, long1: float, timeseries1: [float], lat2: float, long2: float, timeseries2: [float]):
    """
    Calculate the distance function between two points D(x_i, x_j) = 1 - exp(- d(x_i, x_j)/2a^2) r(x_i, x_j)
    :param timeseries2:
    :param long2:
    :param lat2:
    :param timeseries1:
    :param long1:
    :param lat1:
    :return:
    """
    a = math.sqrt(- (1500 / (math.log(0.5))))
    earth_radius = 6371  # km
    lat1, lat2, long1, long2 = map(np.radians, [lat1, lat2, long1, long2])
    delta_phi = lat2 - lat1
    delta_lambda = long2 - long1
    haversine_distance = 2 * earth_radius * np.arcsin(
        np.sqrt(np.sin(delta_phi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lambda / 2) ** 2)
    )

    # Pearsons correlation coefficient
    r = np.corrcoef(timeseries1, timeseries2)[0, 1]

    # calculate difference
    difference = 1 - np.exp(-haversine_distance / (2 * a ** 2)) * r
    return difference


async def merge_clusters(db: Prisma, cluster1: Prisma.cluster, cluster2: Prisma.cluster, difference: Prisma.difference):
    """
    Merge two clusters with proper bidirectional neighbor handling
    """
    cluster1 = await db.cluster.find_first(where={"id": cluster1.id}, include={"grid_points": True})
    cluster2 = await db.cluster.find_first(where={"id": cluster2.id}, include={"grid_points": True})

    # Combine grid points
    grid_points1 = cluster1.grid_points
    grid_points2 = cluster2.grid_points
    new_grid_points = [grid_point for grid_point in grid_points1] + [grid_point for grid_point in grid_points2]

    # Get unique neighbors from both clusters
    new_neighbors = list(set(cluster1.neighbor_ids + cluster2.neighbor_ids))
    new_neighbors = [n for n in new_neighbors if n not in (cluster1.id, cluster2.id)]

    new_cluster = await db.cluster.create({
        "grid_points": {
            "connect": [{"id": grid_point.id} for grid_point in new_grid_points]
        },
        "neighbor_ids": []  # Start with empty neighbors
    })

    # establish all neighbor relationships
    for neighbor_id in new_neighbors:
        await ensure_bidirectional_neighbors(db, new_cluster.id, neighbor_id)

    # update the new_cluster object with final neighbor list
    new_cluster = await db.cluster.find_first(where={"id": new_cluster.id})

    # Continue with difference recalculation and cleanup
    await recalculate_difference(db, new_cluster, cluster1, cluster2, grid_points1, grid_points2)

    # Clean up old relationships and clusters
    await cleanup_old_clusters(db, cluster1.id, cluster2.id, difference.id)
    return


async def cleanup_old_clusters(db: Prisma, cluster1_id: int, cluster2_id: int, difference_id: int):
    """
    Clean up old clusters and their relationships
    """
    # Delete the old difference
    await db.difference.delete(where={"id": difference_id})

    # Delete all differences involving the old clusters
    await db.difference.delete_many(where={
        "OR": [
            {"cluster1Id": cluster1_id},
            {"cluster2Id": cluster1_id},
            {"cluster1Id": cluster2_id},
            {"cluster2Id": cluster2_id}
        ]
    })

    # Remove references to old clusters from their neighbors
    clusters_to_update = await db.cluster.find_many(where={"neighbor_ids": {"hasSome": [cluster1_id, cluster2_id]}})

    for cluster in clusters_to_update:
        updated_neighbors = [n for n in cluster.neighbor_ids if n not in (cluster1_id, cluster2_id)]
        await db.cluster.update(
            where={"id": cluster.id},
            data={"neighbor_ids": updated_neighbors}
        )

    # Delete the old clusters
    await db.cluster.delete(where={"id": cluster1_id})
    await db.cluster.delete(where={"id": cluster2_id})


async def start_clustering(db: Prisma, k: [int], sea_level_anomaly_data: xr.Dataset):
    """
    Start hierarchical clustering
    :param sea_level_anomaly_data:
    :param db:
    :param k:
    :return:
    """
    sorted_k = sorted(k)
    number_of_clusters = await db.cluster.count()
    logger.info(f"Clustering {number_of_clusters} clusters")
    while number_of_clusters > sorted_k[0]:
        min_difference = await db.difference.find_first(
            order={"difference": "asc"},
            include={"cluster1": True, "cluster2": True}
        )

        if not min_difference:
            # Verify if there should be more differences
            clusters = await db.cluster.find_many()
            for c1 in clusters:
                for neighbor_id in c1.neighbor_ids:
                    diff = await db.difference.find_first(
                        where={
                            "OR": [
                                {"cluster1Id": c1.id, "cluster2Id": neighbor_id},
                                {"cluster1Id": neighbor_id, "cluster2Id": c1.id}
                            ]
                        }
                    )
                    if not diff:
                        logger.error(f"Missing difference between clusters {c1.id} and {neighbor_id}")
                        # Recalculate missing difference here

            logger.info("No more valid differences to process")
            break

        await merge_clusters(db, min_difference.cluster1, min_difference.cluster2, min_difference)
        number_of_clusters -= 1
        if number_of_clusters % 1000 == 0:
            logger.info(f"Number of clusters left: {number_of_clusters}")
        number_of_differences = await db.difference.count()
        if number_of_differences == 0:
            logger.info(f"No more neighbors to merge, there are {number_of_clusters} clusters left")
            await save_and_plot_clusters(db, number_of_clusters, sea_level_anomaly_data.copy())
            logger.info(f"Did not create new diffs: {NO_DIFF_COUNTER} times")
            logger.info(f"Created new diffs: {DIFF_COUNTER} times")
            logger.info(f"No diffs between both neighbors: {NO_DIFF_COUNTER_BOTH_NEIGHBORS} times")
            exit()
        if number_of_clusters in sorted_k:
            await save_and_plot_clusters(db, number_of_clusters, sea_level_anomaly_data.copy())

    return
