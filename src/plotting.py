import colorsys
import os
import random

import geopandas
import numpy as np
import shapely
import xarray as xr
from cartopy import crs as ccrs
from loguru import logger
from matplotlib import pyplot as plt
from prisma import Prisma
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import transform, unary_union


def plot_sla_for_point_in_time(sea_level_anomaly_data: xr.Dataset, out_dir: str, feature, name: str):
    """
    Plot sea level anomaly for one point in time
    :param feature:
    :param name:
    :param out_dir:
    :param sea_level_anomaly_data:
    :return:
    """
    # plot data for one point in time
    data = sea_level_anomaly_data[feature].isel(time=0)
    fig = plt.figure(figsize=(50, 25))
    ax = plt.axes(projection=ccrs.PlateCarree())
    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='jet', add_colorbar=True)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.savefig(os.path.join(out_dir, f"{name}.pdf"), dpi=500)
    plt.close(fig)


def shift_all_longitudes(geom):
    """
    Shift geometries from -180/180 to 0/360 format, handling invalid geometries
    """
    if not geom.is_valid:
        geom = geom.buffer(0)  # Try to fix invalid geometries

    def shift_coords(x, y):
        return (x + 360 if x < 0 else x, y)

    if isinstance(geom, (Polygon, MultiPolygon)):
        bounds = geom.bounds
        # If geometry crosses the meridian
        if bounds[0] < 0 and bounds[2] > 0:
            try:
                # Create a meridian line with a small buffer to ensure clean cuts
                meridian = shapely.geometry.LineString([(0, -90), (0, 90)]).buffer(0.0001)
                # Split the geometry
                west_half = geom.difference(meridian)
                east_half = geom.intersection(meridian)

                # Shift the western part
                shifted_west = transform(shift_coords, west_half)

                # Ensure both parts are valid
                if not shifted_west.is_valid:
                    shifted_west = shifted_west.buffer(0)
                if not east_half.is_valid:
                    east_half = east_half.buffer(0)

                # Combine parts using a more robust approach
                try:
                    result = shifted_west.union(east_half)
                    if result.is_valid:
                        return result
                except:
                    pass  # If union fails, fall back to simple transformation
            except:
                pass  # If splitting fails, fall back to simple transformation

    # Fall back to simple transformation for all other cases
    try:
        result = transform(shift_coords, geom)
        if result.is_valid:
            return result
        else:
            return result.buffer(0)
    except:
        # If all else fails, return the original geometry
        return geom


def plot_regions(land_gdf: geopandas.GeoDataFrame, output_path: str,
                 clusters_gdf: geopandas.GeoDataFrame, name: str):
    """
    Plot the regions on a map
    :param name:
    :param land_gdf:
    :param output_path:
    :param clusters_gdf:
    :return:
    """

    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    clusters_gdf.plot(ax=ax, color=clusters_gdf["color"], zorder=4, linewidth=4)
    clusters_gdf.boundary.plot(ax=ax, color=clusters_gdf["color"], zorder=5, linewidth=0.5)
    plt.xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.yticks([-90, -45, 0, 45, 90])
    plt.savefig(os.path.join(output_path, f"{name}.svg"))
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    plt.close()
    # also plot ranging from 0 to 360 degrees longitude for better comparability of the images
    land_gdf_360 = land_gdf.copy()
    land_gdf_360["geometry"] = land_gdf_360["geometry"].apply(shift_all_longitudes)
    clusters_gdf_360 = clusters_gdf.copy()
    clusters_gdf_360["geometry"] = clusters_gdf_360["geometry"].apply(shift_all_longitudes)
    ax = land_gdf_360.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    clusters_gdf_360.plot(ax=ax, color=clusters_gdf_360["color"], zorder=4, linewidth=4)
    clusters_gdf_360.boundary.plot(ax=ax, color=clusters_gdf_360["color"], zorder=5, linewidth=0.5)
    plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    plt.yticks([-90, -45, 0, 45, 90])
    plt.savefig(os.path.join(output_path, f"{name}_360.svg"))
    plt.savefig(os.path.join(output_path, f"{name}_360.png"))
    plt.close()
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
    cluster_gdf, land_gdf = await create_gdf_from_xarray_dataset(clusters, number_of_clusters)

    plot_regions(land_gdf, "../output/", cluster_gdf, f"clusters_{number_of_clusters}")

    logger.info(f"Clusters saved and plotted")
    return


async def create_gdf_from_xarray_dataset(clusters, number_of_clusters):
    """
    Create a geopandas dataframe from the clusters
    :param clusters:
    :param number_of_clusters:
    :return:
    """
    land_gdf = geopandas.read_file("../data/ne_10m_land/ne_10m_land.shp")
    colors = random_color_generator(number_of_clusters + 1)
    grid_point_area = 2.5
    # turn clusters into a geopandas dataframe
    counter = 0
    cluster_ids = []
    polygons = []
    for cluster in clusters:
        # create a polygon from all grid points in the current cluster
        cluster_squares = []
        cluster_ids.append(counter)
        counter += 1

        for grid_point in cluster.grid_points:
            square = shapely.Polygon([
                (grid_point.longitude + grid_point_area, grid_point.latitude + grid_point_area),
                (grid_point.longitude + grid_point_area, grid_point.latitude - grid_point_area),
                (grid_point.longitude - grid_point_area, grid_point.latitude - grid_point_area),
                (grid_point.longitude - grid_point_area, grid_point.latitude + grid_point_area)
            ])
            cluster_squares.append(square)

        # Merge all squares in the cluster using unary_union
        merged_polygon = unary_union(cluster_squares)
        polygons.append(merged_polygon)

        # Create GeoDataFrame with merged polygons
    cluster_gdf = geopandas.GeoDataFrame(
        {'cluster_id': cluster_ids, 'color': colors, 'geometry': polygons}
        # ,crs="EPSG:4326"  # WGS 84 coordinate system
    )
    return cluster_gdf, land_gdf


def random_color_generator(num_colors: int):
    """
    Generates a list of random colors
    :param num_colors:
    :return:
    """
    colors = []

    for i in range(num_colors - 1):
        h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
        r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
        colors.append('#%02x%02x%02x' % (r, g, b))
        # colors.append(random.choice(list(mcolors.CSS4_COLORS.keys())))
    return colors


def plot_nan_values(data, time_step):
    """
    Plot the NaN distribution at a given time step
    :param data:
    :param time_step:
    :return:
    """
    nan_mask = data['sla'].isel(time=time_step).isnull()
    # Extract lat/lon for correct map projection
    lat = data['latitude']
    lon = data['longitude']
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(lon, lat, nan_mask, cmap='gray', shading='auto')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'NaN Distribution at Time Step {time_step}')
    plt.colorbar(label='NaN Mask (1 = NaN, 0 = Valid Data)')
    plt.savefig(f'../output/nan_distribution_{time_step}.png')
