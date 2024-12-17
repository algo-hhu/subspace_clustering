import math

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib import pyplot as plt
from prisma import Prisma


async def recalculate_difference(db: Prisma, new_cluster, cluster1, cluster2, new_grid_points, grid_points1,
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
    all_neighbors = new_cluster.neighbor_ids
    neighbors_1 = cluster1.neighbor_ids
    neighbors_2 = cluster2.neighbor_ids
    for neighbor_id in all_neighbors:
        current_neighbor = await db.cluster.find_first(where={"id": neighbor_id}, include={"grid_points": True})
        if neighbor_id in neighbors_1 and neighbor_id in neighbors_2:
            difference1 = await db.difference.find_first(
                where={"cluster1Id": cluster1.id, "cluster2Id": neighbor_id}
            )
            if not difference1:
                difference1 = await db.difference.find_first(
                    where={"cluster1Id": neighbor_id, "cluster2Id": cluster1.id}
                )
            difference2 = await db.difference.find_first(
                where={"cluster1Id": cluster2.id, "cluster2Id": neighbor_id}
            )
            if not difference2:
                difference2 = await db.difference.find_first(
                    where={"cluster1Id": neighbor_id, "cluster2Id": cluster2.id}
                )
            if not difference1 or not difference2:
                continue
            new_difference = (difference1.difference * len(grid_points1) + difference2.difference * len(
                grid_points2)) / (len(grid_points1) + len(grid_points2))
            await db.difference.delete(where={"id": difference1.id})
            await db.difference.delete(where={"id": difference2.id})
        elif neighbor_id in neighbors_1:
            new_difference = await recalculate_difference_if_one_neighbor(cluster1, current_neighbor, db, grid_points1,
                                                                          grid_points2, neighbor_id)
        elif neighbor_id in neighbors_2:
            new_difference = await recalculate_difference_if_one_neighbor(cluster2, current_neighbor, db, grid_points2,
                                                                          grid_points1, neighbor_id)
        if not new_difference:
            return
        await db.difference.create(
            data={
                "cluster1Id": new_cluster.id,
                "cluster2Id": current_neighbor.id,
                "difference": new_difference
            }
        )

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
        return
    # calculate difference2 as the average difference between all grid points in cluster1 and all gridpoints inthe neighbor-cluster
    difference2 = 0
    grid_point_pairs = [(grid_point1, grid_point2) for grid_point1 in grid_points1 for grid_point2 in
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
    Merge two clusters
    :param db:
    :param difference:
    :param cluster1:
    :param cluster2:
    :return:
    """
    # add new cluster with the grid points from cluster1 and cluster2
    cluster1 = await db.cluster.find_first(where={"id": cluster1.id}, include={"grid_points": True})
    cluster2 = await db.cluster.find_first(where={"id": cluster2.id}, include={"grid_points": True})
    grid_points1 = cluster1.grid_points
    grid_points2 = cluster2.grid_points
    new_grid_points = [grid_point for grid_point in grid_points1] + [grid_point for grid_point in grid_points2]
    new_neighbors = cluster1.neighbor_ids + cluster2.neighbor_ids
    if cluster2.id in new_neighbors:
        new_neighbors.remove(cluster2.id)
    if cluster1.id in new_neighbors:
        new_neighbors.remove(cluster1.id)
    new_cluster = await db.cluster.create(
        {"grid_points": {
            "connect": [{"id": grid_point.id} for grid_point in new_grid_points]
        }, "neighbor_ids": new_neighbors}
    )
    await recalculate_difference(db, new_cluster, cluster1, cluster2, new_grid_points, grid_points1, grid_points2)

    # delete the difference between the two old clusters
    await db.difference.delete(where={"id": difference.id})
    # delete all differences that reference cluster1 or cluster2
    await db.difference.delete_many(where={"OR": [{"cluster1Id": cluster1.id}, {"cluster2Id": cluster1.id}]})
    await db.difference.delete_many(where={"OR": [{"cluster1Id": cluster2.id}, {"cluster2Id": cluster2.id}]})

    # remove cluster1 and cluster2 and references to them
    await db.cluster.delete(where={"id": cluster1.id})
    await db.cluster.delete(where={"id": cluster2.id})
    return


async def save_and_plot_clusters(db: Prisma, number_of_clusters: int, sea_level_anomaly_data: xr.Dataset):
    """
    Save and plot clusters
    :param sea_level_anomaly_data:
    :param db:
    :param number_of_clusters:
    :return:
    """
    cluster_data = np.zeros((sea_level_anomaly_data.latitude.size, sea_level_anomaly_data.longitude.size))
    clusters = await db.cluster.find_many(include={"grid_points": True})
    # create a netcdf file with the cluster information
    cluster_number = 0
    for cluster in clusters:
        for grid_point in cluster.grid_points:
            # get index of lat long in sea_level_anomaly_data
            lat_index = np.where(sea_level_anomaly_data.latitude.values == grid_point.latitude)[0][0]
            long_index = np.where(sea_level_anomaly_data.longitude.values == grid_point.longitude)[0][0]
            cluster_data[lat_index, long_index] = cluster_number
        cluster_number += 1
    cluster_data = xr.DataArray(cluster_data, dims=["latitude", "longitude"])
    cluster_data = cluster_data.assign_coords(latitude=sea_level_anomaly_data.latitude,
                                              longitude=sea_level_anomaly_data.longitude)
    cluster_data.to_netcdf(f"../output/clusters_{number_of_clusters}.nc")
    logger.info(f"plot clusters")
    # plot the clusters
    data = cluster_data
    fig = plt.figure(figsize=(50, 25))
    ax = plt.axes(projection=ccrs.PlateCarree())
    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet', add_colorbar=True)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.savefig(f"../output/clusters_{number_of_clusters}.svg", dpi=500)
    plt.close(fig)
    logger.info(f"Clusters saved and plotted")
    return


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
    logger.info(f"Number of differences in db: {await db.difference.count()}")
    while number_of_clusters > sorted_k[0]:
        min_difference = await db.difference.find_first(order={"difference": "asc"},
                                                        include={"cluster1": True, "cluster2": True})
        await merge_clusters(db, min_difference.cluster1, min_difference.cluster2, min_difference)
        number_of_clusters -= 1
        if number_of_clusters % 1000 == 0:
            logger.info(f"Number of clusters left: {number_of_clusters}")
        number_of_differences = await db.difference.count()
        if number_of_differences == 0:
            logger.info(f"No more neighbors to merge, there are {number_of_clusters} clusters left")
            await save_and_plot_clusters(db, number_of_clusters, sea_level_anomaly_data.copy())
            exit()
        if number_of_clusters in sorted_k:
            await save_and_plot_clusters(db, number_of_clusters, sea_level_anomaly_data.copy())
    return
