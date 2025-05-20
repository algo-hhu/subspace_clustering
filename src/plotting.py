import colorsys
import os
import random

import geopandas
import matplotlib.patches as mpatches
import shapely
import xarray as xr
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.point import Point
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
    # clusters_gdf.boundary.plot(ax=ax, color=clusters_gdf["color"], zorder=5, linewidth=0.5)
    plt.xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.yticks([-90, -45, 0, 45, 90])
    handles = [mpatches.Patch(color=color, label=f"Cluster {cluster_id}")
               for cluster_id, color in
               zip(clusters_gdf["cluster_id"].unique(), clusters_gdf["color"].unique())]
    ax.legend(handles=handles, title="Clusters")
    plt.savefig(os.path.join(output_path, f"{name}.svg"))
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    plt.close()
    # # also plot ranging from 0 to 360 degrees longitude for better comparability of the images
    # land_gdf_360 = land_gdf.copy()
    # land_gdf_360["geometry"] = land_gdf_360["geometry"].apply(shift_all_longitudes)
    # clusters_gdf_360 = clusters_gdf.copy()
    # clusters_gdf_360["geometry"] = clusters_gdf_360["geometry"].apply(shift_all_longitudes)
    # ax = land_gdf_360.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    # ax.set_facecolor("aliceblue")
    # clusters_gdf_360.plot(ax=ax, color=clusters_gdf_360["color"], zorder=4, linewidth=4)
    # handles = [mpatches.Patch(color=color, label=f"Cluster {cluster_id}")
    #            for cluster_id, color in
    #            zip(clusters_gdf_360["cluster_id"].unique(), clusters_gdf_360["color"].unique())]
    # # clusters_gdf_360.boundary.plot(ax=ax, color=clusters_gdf_360["color"], zorder=5, linewidth=0.5)
    # plt.xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    # plt.yticks([-90, -45, 0, 45, 90])
    # ax.legend(handles=handles, title="Clusters")
    # plt.savefig(os.path.join(output_path, f"{name}_360.svg"))
    # plt.savefig(os.path.join(output_path, f"{name}_360.png"))
    # plt.close()
    return


def plot_one_timeseries(sea_level_anomaly_data, out_dir, id_x, id_y, name: str):
    """
    Plot one time series
    :param sea_level_anomaly_data:
    :param out_dir:
    :param id_x:
    :param id_y:
    :return:
    """
    data = sea_level_anomaly_data["sla"].isel(latitude=id_y, longitude=id_x)
    plt.figure(figsize=(12, 6))
    data.plot()
    plt.xlabel('Time')
    plt.ylabel('Sea Level Anomaly')
    plt.title(f'Sea Level Anomaly at ({data.latitude.values}, {data.longitude.values})')
    plt.savefig(os.path.join(out_dir, f"{name}timeseries_{id_x}_{id_y}.png"))
    plt.close()


# async def save_and_plot_clusters(db: Prisma, number_of_clusters: int, sea_level_anomaly_data: xr.Dataset):
#     """
#     Save and plot clusters
#     :param sea_level_anomaly_data:
#     :param db:
#     :param number_of_clusters:
#     :return:
#     """
#     cluster_data = np.zeros((sea_level_anomaly_data.latitude.size, sea_level_anomaly_data.longitude.size))
#     clusters = await db.cluster.find_many(include={"grid_points": True})
#     # create a netcdf file with the cluster information
#     cluster_number = 0
#     for cluster in clusters:
#         for grid_point in cluster.grid_points:
#             # get index of lat long in sea_level_anomaly_data
#             lat_index = np.where(sea_level_anomaly_data.latitude.values == grid_point.latitude)[0][0]
#             long_index = np.where(sea_level_anomaly_data.longitude.values == grid_point.longitude)[0][0]
#             cluster_data[lat_index, long_index] = cluster_number
#         cluster_number += 1
#     cluster_data = xr.DataArray(cluster_data, dims=["latitude", "longitude"])
#     cluster_data = cluster_data.assign_coords(latitude=sea_level_anomaly_data.latitude,
#                                               longitude=sea_level_anomaly_data.longitude)
#     cluster_data.to_netcdf(f"../output/clusters_{number_of_clusters}.nc")
#     logger.info(f"plot clusters")
#     cluster_gdf, land_gdf = await create_gdf_from_xarray_dataset(clusters, number_of_clusters)
#
#     plot_regions(land_gdf, "../output/", cluster_gdf, f"clusters_{number_of_clusters}")
#
#     logger.info(f"Clusters saved and plotted")
#     return


async def create_gdf_from_xarray_dataset(clusters, number_of_clusters):
    """
    Create a geopandas dataframe from the clusters
    :param clusters:
    :param number_of_clusters:
    :return:
    """
    land_gdf = geopandas.read_file("../data/ne_10m_land/ne_10m_land.shp")
    colors = random_color_generator(number_of_clusters + 1)
    grid_point_area = 1
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


def turn_dict_into_gdf(cluster_dict: {float: [(float, float)]}, grid_point_area: float, cluster_colors: [str]):
    """
    Turn a dictionary into a geopandas dataframe
    :param cluster_colors:
    :param grid_point_area:
    :param cluster_dict:
    :return:
    """
    land_gdf = geopandas.read_file("../data/ne_10m_land/ne_10m_land.shp")
    if not len(cluster_colors) >= len(cluster_dict.keys()):
        colors = random_color_generator(len(cluster_dict.keys()) + 1)
    else:
        colors = cluster_colors[:len(cluster_dict.keys())]
    # turn clusters into a geopandas dataframe
    # counter = 0
    cluster_ids = []
    polygons = []
    for cluster in cluster_dict.keys():
        # create a polygon from all grid points in the current cluster
        cluster_squares = []
        cluster_ids.append(cluster)
        # counter += 1

        for grid_point in cluster_dict[cluster]:
            square = shapely.Polygon([
                (grid_point[1] + grid_point_area, grid_point[0] + grid_point_area),
                (grid_point[1] + grid_point_area, grid_point[0] - grid_point_area),
                (grid_point[1] - grid_point_area, grid_point[0] - grid_point_area),
                (grid_point[1] - grid_point_area, grid_point[0] + grid_point_area)
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


def plot_clustering(cluster_dict, out_dir, resolution, name):
    """
    Plot the clustering
    :param cluster_dict:
    :param out_dir:
    :param resolution:
    :param name:
    :return:
    """
    cluster_colors = ["gold", "yellowgreen", "dodgerblue", "rebeccapurple", "orchid", "maroon",
                      "darkorange", "palegoldenrod", "darkolivegreen", "forestgreen", "teal", "darkblue", "darkorchid",
                      "deeppink", "red", "yellow", "darkseagreen", "azure", "lightsteelblue", "midnightblue", "plum",
                      "sienna", "chartreuse", "darkslategray", "darkmagenta", "crimson", "cornflowerblue", "chocolate",
                      "lemonchiffon", "lavenderblush", "navy", "purple"]
    cluster_gdf, land_gdf = turn_dict_into_gdf(cluster_dict, resolution / 2,
                                               cluster_colors)
    plot_regions(land_gdf, out_dir, cluster_gdf, name)


def plot_clustering_with_component_graph(cluster_dict, out_dir, resolution, name, connected_component_graph,
                                         connected_components, grid_point_to_lat_lon):
    """
    Plot the clustering
    :param grid_point_to_lat_lon:
    :param connected_components:
    :param connected_component_graph:
    :param cluster_dict:
    :param out_dir:
    :param resolution:
    :param name:
    :return:
    """
    cluster_colors = ["gold", "yellowgreen", "dodgerblue", "rebeccapurple", "orchid", "maroon",
                      "darkorange", "palegoldenrod", "darkolivegreen", "forestgreen", "teal", "darkblue", "darkorchid",
                      "deeppink", "red", "yellow", "darkseagreen", "azure", "lightsteelblue", "midnightblue", "plum",
                      "sienna", "chartreuse", "darkslategray", "darkmagenta", "crimson", "cornflowerblue", "chocolate",
                      "lemonchiffon", "lavenderblush", "navy", "purple"]
    cluster_gdf, land_gdf = turn_dict_into_gdf(cluster_dict, resolution / 2,
                                               cluster_colors)
    # calculate the middle coordinates for each connected component
    mean_points = {}
    connected_components_points_dict = {'name': [], 'geometry': [], 'color': []}
    for connected_component in connected_components.values():
        latitudes = []
        longitudes = []
        cluster_id = connected_component.cluster_id
        matching_rows = cluster_gdf[cluster_gdf['cluster_id'] == cluster_id]
        if not matching_rows.empty:
            cluster_color = matching_rows.iloc[0]['color']
        else:
            # If no matching rows are found, assign a default color
            cluster_color = "gray"
        connected_components_points_dict['color'].append(cluster_color)
        for node in connected_component.nodes:
            (lat, lon) = grid_point_to_lat_lon[node]
            longitudes.append(lon)
            latitudes.append(lat)
        # if the longitude crosses the -180/180 boundary, shift the longitudes
        # to the 0/360 format and calculate the mean, and then shift back to -180/180
        if min(longitudes) < -170 and max(longitudes) > 170:
            longitudes = [lon + 360 if lon < 0 else lon for lon in longitudes]
            mean_longitude = sum(longitudes) / len(longitudes)
            if mean_longitude > 180:
                mean_longitude -= 360
        else:
            mean_longitude = sum(longitudes) / len(longitudes)
        mean_latitude = sum(latitudes) / len(latitudes)

        mean_point = Point(mean_longitude, mean_latitude)
        mean_points[connected_component.id] = mean_point
        connected_components_points_dict['name'].append(connected_component.id)
        connected_components_points_dict['geometry'].append(mean_point)
        # create geodataframe with point in the center of the connected component
    connected_component_lines_dict = {'name': [], 'geometry': []}
    for connected_component_id in connected_components.keys():
        neighbors = connected_component_graph.neighbors(connected_component_id)
        for neighbor in neighbors:
            # check if the line crosses the longitude -180/180 boundary, if it does, draw the line to the other side
            if abs(mean_points[connected_component_id].x - mean_points[neighbor].x) > 100:
                if mean_points[connected_component_id].x < 0:
                    current_line = shapely.geometry.LineString(
                        [mean_points[connected_component_id],
                         shapely.geometry.Point(-178, mean_points[connected_component_id].y)])
                elif mean_points[connected_component_id].x >= 0:
                    current_line = shapely.geometry.LineString(
                        [mean_points[connected_component_id],
                         shapely.geometry.Point(178, mean_points[connected_component_id].y)])
            else:
                current_line = shapely.geometry.LineString([mean_points[neighbor], mean_points[connected_component_id]])
            connected_component_lines_dict['name'].append(connected_component_id)
            connected_component_lines_dict['geometry'].append(current_line)
    connected_component_graph_nodes_gdf = geopandas.GeoDataFrame(connected_components_points_dict, geometry='geometry')
    connected_component_graph_gdf = geopandas.GeoDataFrame(connected_component_lines_dict, geometry='geometry')
    plot_regions_with_component_graph(land_gdf, out_dir, cluster_gdf, name, connected_component_graph_gdf,
                                      connected_component_graph_nodes_gdf)


def plot_regions_with_component_graph(land_gdf: geopandas.GeoDataFrame, output_path: str,
                                      clusters_gdf: geopandas.GeoDataFrame, name: str, connected_component_graph_gdf,
                                      connected_component_graph_nodes_gdf):
    """
    Plot the regions on a map
    :param connected_component_graph_nodes_gdf:
    :param connected_component_graph_gdf:
    :param name:
    :param land_gdf:
    :param output_path:
    :param clusters_gdf:
    :return:
    """

    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    clusters_gdf.plot(ax=ax, color=clusters_gdf["color"], zorder=4, linewidth=4)
    # clusters_gdf.boundary.plot(ax=ax, color=clusters_gdf["color"], zorder=5, linewidth=0.5)
    connected_component_graph_gdf.plot(ax=ax, color="black", zorder=5, linewidth=0.5)
    connected_component_graph_nodes_gdf.plot(ax=ax, marker='o',
                                             facecolor=connected_component_graph_nodes_gdf["color"],
                                             edgecolor="black", zorder=6, markersize=8)
    plt.xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.yticks([-90, -45, 0, 45, 90])
    handles = [mpatches.Patch(color=color, label=f"Cluster {cluster_id}")
               for cluster_id, color in
               zip(clusters_gdf["cluster_id"].unique(), clusters_gdf["color"].unique())]
    ax.legend(handles=handles, title="Clusters")
    plt.savefig(os.path.join(output_path, f"{name}.svg"))
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    plt.close()


def plot_graph_on_clustering_map(cluster_dict, grid_graph, grid_point_to_lat_lon, resolution, out_dir,
                                 name):
    """
    Plot the grid graph on the clustering map
    :param grid_point_to_lat_lon:
    :param out_dir:
    :param cluster_dict:
    :param grid_graph:
    :param resolution:
    :param name:
    :return:
    """
    cluster_colors = ["gold", "yellowgreen", "dodgerblue", "rebeccapurple", "orchid", "maroon",
                      "darkorange", "palegoldenrod", "darkolivegreen", "forestgreen", "teal", "darkblue", "darkorchid",
                      "deeppink", "red", "yellow", "darkseagreen", "azure", "lightsteelblue", "midnightblue", "plum",
                      "sienna", "chartreuse", "darkslategray", "darkmagenta", "crimson", "cornflowerblue", "chocolate",
                      "lemonchiffon", "lavenderblush", "navy", "purple"]
    cluster_gdf, land_gdf = turn_dict_into_gdf(cluster_dict, resolution / 2,
                                               cluster_colors)
    lines_dict = {'name': [], 'geometry': []}
    nodes_dict = {'name': [], 'geometry': [], 'color': []}
    for node in grid_graph.nodes:
        lat, lon = grid_point_to_lat_lon[node]
        nodes_dict['name'].append(node)
        nodes_dict['geometry'].append(Point(lon, lat))
        nodes_dict['color'].append('black')
        for neighbors in grid_graph.neighbors(node):
            lat2, lon2 = grid_point_to_lat_lon[neighbors]
            # check if the line crosses the longitude -180/180 boundary, if it does, do not draw the line
            if (lon < - 170 and lon2 > 170) or (lon > 170 and lon2 < -170):
                continue
            # create a line between the two points
            line = shapely.geometry.LineString([(lon, lat), (lon2, lat2)])
            lines_dict['geometry'].append(line)
            lines_dict['name'].append(node)
    edges_gdf = geopandas.GeoDataFrame(lines_dict, geometry='geometry')
    nodes_gdf = geopandas.GeoDataFrame(nodes_dict, geometry='geometry')
    plot_regions_with_component_graph(land_gdf, out_dir, cluster_gdf, name, edges_gdf, nodes_gdf)


def plot_with_highlighting_of_component(clustering, smallest_component, neighbors, out_dir, name, resolution,
                                        connected_components, grid_point_to_lat_lon):
    """
    Plot the clustering with highlighting of the smallest component and its neighbors
    :param connected_components:
    :param clustering:
    :param smallest_component:
    :param neighbors:
    :param out_dir:
    :param name:
    :param resolution:
    :return:
    """
    cluster_colors = ["gold", "yellowgreen", "dodgerblue", "rebeccapurple", "orchid", "maroon",
                      "darkorange", "palegoldenrod", "darkolivegreen", "forestgreen", "teal", "darkblue", "darkorchid",
                      "deeppink", "red", "yellow", "darkseagreen", "azure", "lightsteelblue", "midnightblue", "plum",
                      "sienna", "chartreuse", "darkslategray", "darkmagenta", "crimson", "cornflowerblue", "chocolate",
                      "lemonchiffon", "lavenderblush", "navy", "purple"]
    clusters_gdf, land_gdf = turn_dict_into_gdf(clustering, resolution / 2,
                                                cluster_colors)
    latitudes = []
    longitudes = []
    smallest_component_dict = {'name': [], 'geometry': [], 'color': []}
    for node in smallest_component.nodes:
        latitudes.append(grid_point_to_lat_lon[node][0])
        longitudes.append(grid_point_to_lat_lon[node][1])
    mean_coordinates_smallest_component = (sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes))
    smallest_component_dict['name'].append(smallest_component.id)
    smallest_component_dict['geometry'].append(Point(mean_coordinates_smallest_component[1],
                                                     mean_coordinates_smallest_component[0]))
    # extract the color of the smallest component
    matching_rows = clusters_gdf[clusters_gdf['cluster_id'] == smallest_component.cluster_id]
    if not matching_rows.empty:
        cluster_color = matching_rows.iloc[0]['color']
    else:
        # If no matching rows are found, assign a default color
        cluster_color = "gray"
    smallest_component_dict['color'].append(cluster_color)
    smallest_component_gdf = geopandas.GeoDataFrame(smallest_component_dict, geometry='geometry')

    neighbors_dict = {'name': [], 'geometry': [], 'color': []}
    for neighbor in neighbors:
        neighbor_component = connected_components[neighbor]
        latitudes = []
        longitudes = []
        for node in neighbor_component.nodes:
            latitudes.append(grid_point_to_lat_lon[node][0])
            longitudes.append(grid_point_to_lat_lon[node][1])
        # shift longitudes to 0/360 format if they cross the -180/180 boundary and back to -180/180 format
        if min(longitudes) < -170 and max(longitudes) > 170:
            longitudes = [lon + 360 if lon < 0 else lon for lon in longitudes]
            mean_longitude = sum(longitudes) / len(longitudes)
            if mean_longitude > 180:
                mean_longitude -= 360
        else:
            mean_longitude = sum(longitudes) / len(longitudes)
        mean_coordinates_neighbor = (sum(latitudes) / len(latitudes), mean_longitude)
        # extract the color of the neighbor component
        matching_rows = clusters_gdf[clusters_gdf['cluster_id'] == neighbor_component.cluster_id]
        if not matching_rows.empty:
            cluster_color = matching_rows.iloc[0]['color']
        else:
            cluster_color = "gray"
        neighbors_dict['color'].append(cluster_color)
        neighbors_dict['name'].append(neighbor_component.id)
        neighbors_dict['geometry'].append(Point(mean_coordinates_neighbor[1], mean_coordinates_neighbor[0]))
    neighbors_gdf = geopandas.GeoDataFrame(neighbors_dict, geometry='geometry')

    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    clusters_gdf.plot(ax=ax, color=clusters_gdf["color"], zorder=4, linewidth=4)
    # clusters_gdf.boundary.plot(ax=ax, color=clusters_gdf["color"], zorder=5, linewidth=0.5)
    smallest_component_gdf.plot(ax=ax, marker='o', facecolor=smallest_component_gdf['color'], edgecolor='red',
                                zorder=5, markersize=10)
    neighbors_gdf.plot(ax=ax, marker='o', facecolor=neighbors_gdf['color'], edgecolor="black", zorder=6, markersize=10)
    plt.xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.yticks([-90, -45, 0, 45, 90])
    handles = [mpatches.Patch(color=color, label=f"Cluster {cluster_id}")
               for cluster_id, color in
               zip(clusters_gdf["cluster_id"].unique(), clusters_gdf["color"].unique())]
    ax.legend(handles=handles, title="Clusters")
    plt.savefig(os.path.join(out_dir, f"{name}.png"))
    plt.close()
