import uuid

import networkx as nx
import numpy as np
from loguru import logger
from tqdm import tqdm

from src.clustering.cluster_entities.connected_component import ConnectedComponent


def find_neighbors(sea_level_anomaly_data, nan_mask, lat_lon_to_grid_point_id):
    """
    Find neighbors for each grid point in the latitudes and longitudes arrays.

    :param sea_level_anomaly_data:
    :param lat_lon_to_grid_point_id:
    :param nan_mask: Mask indicating NaN values in the data array.
    :return:
    """
    # Initialize variables

    lat_range = sea_level_anomaly_data["latitude"].shape[0]
    latitudes = sea_level_anomaly_data.latitude.values
    long_range = sea_level_anomaly_data["longitude"].shape[0]
    longitudes = sea_level_anomaly_data.longitude.values
    data = sea_level_anomaly_data["sla"].values
    lat_range = len(latitudes)
    long_range = len(longitudes)

    first_longitude, last_longitude, lat_for_first_longitude, lat_for_last_longitude = find_first_last_longitude(
        lat_range, long_range, nan_mask)
    neighbors = iteratively_find_neighbors(data, latitudes, longitudes, lat_range, long_range, first_longitude,
                                           last_longitude,
                                           nan_mask, lat_lon_to_grid_point_id)
    return neighbors


def iteratively_find_neighbors(data, latitudes, longitudes, lat_range, long_range, first_longitude, last_longitude,
                               nan_mask, lat_lon_to_grid_point_id):
    """

    :return:
    """
    neighbors = {}  # {grid_point_id: {neighbor_grid_point1, neighbor_grid_point2, ...}}
    # iterate through latitudes and longitudes and find neighbors for each grid point
    for i in tqdm(range(lat_range)):
        for j in (range(long_range)):
            if nan_mask[i, j]:  # points without valid data can be skipped
                continue
            neighbors[
                lat_lon_to_grid_point_id[latitudes[i], longitudes[j]]] = set()  # set of neighbors for each grid point
            # direct neighbors
            neighbor_positions = [
                (i - 1, j),  # North
                (i + 1, j),  # South
                (i, (j - 1) % long_range),  # West (wraps around)
                (i, (j + 1) % long_range),  # East (wraps around)
            ]
            # check if the grid point is on the edge of the grid, because of the interpolation there might be nan values at the edges
            if j == last_longitude:
                neighbor_positions.extend([(i, first_longitude), (i - 1, first_longitude), (i + 1, first_longitude)])
            if j == first_longitude:
                neighbor_positions.extend([(i, last_longitude), (i - 1, last_longitude), (i + 1, last_longitude)])
            # diagonal neighbors
            neighbor_positions.extend([
                ((i - 1), (j - 1) % long_range),  # Northwest
                ((i - 1), (j + 1) % long_range),  # Northeast
                ((i + 1), (j - 1) % long_range),  # Southwest
                ((i + 1), (j + 1) % long_range),  # Southeast
            ])
            # Handle out-of-bounds positions
            neighbor_positions_without_out_of_bounds = [
                (pos[0], pos[1]) if 0 <= pos[0] < lat_range else None
                for pos in neighbor_positions
            ]
            valid_neighbor_positions = [(pos[0], pos[1]) for pos in neighbor_positions_without_out_of_bounds if
                                        not nan_mask[pos[0], pos[1]]]
            # add valid neighbors to the set of neighbors
            grid_point1 = lat_lon_to_grid_point_id[latitudes[i], longitudes[j]]
            for pos in valid_neighbor_positions:
                if pos is not None and not nan_mask[pos[0], pos[1]]:
                    grid_point2 = lat_lon_to_grid_point_id[latitudes[pos[0]], longitudes[pos[1]]]
                    neighbors[grid_point1].add(grid_point2)

    neighbors = ensure_bidirectional_neighbors(neighbors)
    return neighbors


def find_first_last_longitude(lat_range, long_range, nan_mask):
    """
    Find the first and last longitude that has data
    :param lat_range:
    :param long_range:
    :param nan_mask:
    :return:
    """
    # find first and last longitude that has data
    first_longitude = np.inf
    lat_for_first_longitude = np.inf
    last_longitude = 0
    lat_for_last_longitude = 0
    for i in range(long_range):
        for j in range(lat_range):
            if not nan_mask[j, i]:
                if i < first_longitude:
                    first_longitude = i
                    lat_for_first_longitude = j
                continue
    for i in reversed(range(long_range)):
        for j in range(lat_range):
            if not nan_mask[j, i]:
                if i > last_longitude:
                    last_longitude = i
                    lat_for_last_longitude = j
                continue
    return first_longitude, last_longitude, lat_for_first_longitude, lat_for_last_longitude


def ensure_bidirectional_neighbors(neighbors: {}):
    """
    Ensure that neighbor-relationships are bidirectional
    :param neighbors:
    :return:
    """
    for key, value in neighbors.items():
        for neighbor in value:
            if key not in neighbors[neighbor]:
                neighbors[neighbor].append(key)
    return neighbors


def generate_grid_graph(lat_lon_to_grid_point_id, nan_mask, sea_level_anomaly_data):
    """
    Generate a grid graph from the sea level anomaly data to capture the neighborhood information
    :param lat_lon_to_grid_point_id:
    :param nan_mask:
    :param sea_level_anomaly_data:
    :return:
    """
    # create grid graph that contains neighborhood information
    grid_graph = nx.Graph()
    neighbors = find_neighbors(sea_level_anomaly_data, nan_mask, lat_lon_to_grid_point_id)
    grid_graph.add_nodes_from(neighbors.keys())
    counter = 0
    for neighbor in neighbors:
        # # check if neighbor is valid point
        # if not nan_mask[neighbor[0], neighbor[1]]:
        #     continue
        for neighbor2 in neighbors[neighbor]:
            # # check if neighbor2 is valid point
            # if not nan_mask[neighbor2[0], neighbor2[1]]:
            #     continue
            if neighbor != neighbor2:
                counter += 1
                try:
                    grid_graph.add_edge(neighbor, neighbor2)
                except nx.NetworkXError:
                    logger.warning(f"Edge between {neighbor} and {neighbor2} already exists or is invalid")
    return grid_graph


def generate_cluster_graph(clustering, grid_graph, lat_lon_to_grid_point_id, nan_mask):
    """
    Generate a graph from the clustering data and the grid graph, in which there is an edge between two nodes if they belong to the same cluster and are neighbors in the grid graph
    :param clustering:
    :param grid_graph:
    :param lat_lon_to_grid_point_id:
    :return:
    """
    # map lat/lon to grid point id in clustering
    clustering_with_grid_points = {}
    for cluster in clustering.keys():
        cluster_id = cluster
        clustering_with_grid_points[cluster_id] = []
        for lat_lon in clustering[cluster]:
            grid_point_id = lat_lon_to_grid_point_id[lat_lon]
            clustering_with_grid_points[cluster_id].append(grid_point_id)
    # create cluster graph where a pair of nodes is connected if they belong to the same cluster and have an edge in the grid graph
    cluster_graph = nx.Graph()
    for cluster in clustering_with_grid_points.keys():
        cluster_id = cluster
        for grid_point in clustering_with_grid_points[cluster]:
            cluster_graph.add_node(grid_point, cluster_id=cluster_id)
            neighbors = grid_graph.neighbors(grid_point)  # get potential neighbors from grid graph
            for neighbor in neighbors:
                if neighbor in clustering_with_grid_points[
                    cluster]:  # check if neighbor is in the same cluster and is already in the cluster graph
                    if cluster_graph.has_node(neighbor):
                        cluster_graph.add_edge(grid_point, neighbor)
    return cluster_graph


def generate_connected_component_graph(cluster_graph, grid_graph):
    """
    Generate a connected component graph from the cluster graph, in which each node is a connected component and there
    is an edge between nodes if any of their nodes are neighbors in the grid graph
    :param cluster_graph:
    :param grid_graph:
    :return:
    """
    connected_component_graph = nx.Graph()
    counter = 0
    connected_components = {}
    for connected_component in nx.connected_components(cluster_graph):
        counter += 1
        first_node = next(iter(connected_component))
        cluster_id = cluster_graph.nodes[first_node]["cluster_id"]
        current_component = ConnectedComponent(uuid.uuid4(), connected_component, cluster_id, len(connected_component))
        connected_component_graph.add_node(current_component.id, component=current_component)
        connected_components[current_component.id] = current_component
    # there should be an edge between each pair of connected components in the graph, if any of their nodes are neighbors in the grid graph
    for connected_component_1 in connected_components.values():
        for connected_component_2 in connected_components.values():
            for node in connected_component_1.nodes:
                neighbors = grid_graph.neighbors(node)
                for neighbor in neighbors:
                    if neighbor in connected_component_2.nodes:
                        connected_component_graph.add_edge(connected_component_1.id, connected_component_2.id)
                        break
    return connected_component_graph, connected_components
