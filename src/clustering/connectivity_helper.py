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
    neighbors = iteratively_find_neighbors(latitudes, longitudes, lat_range, long_range, first_longitude,
                                           last_longitude,
                                           nan_mask, lat_lon_to_grid_point_id)
    return neighbors


def iteratively_find_neighbors(latitudes, longitudes, lat_range, long_range, first_longitude, last_longitude,
                               nan_mask, lat_lon_to_grid_point_id):
    """
    :param latitudes:
    :param longitudes:
    :param lat_range:
    :param long_range:
    :param first_longitude:
    :param last_longitude:
    :param nan_mask:
    :param lat_lon_to_grid_point_id:
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
    # create a grid graph that contains neighborhood information
    grid_graph = nx.Graph()
    neighbors = find_neighbors(sea_level_anomaly_data, nan_mask, lat_lon_to_grid_point_id)
    grid_graph.add_nodes_from(neighbors.keys())
    counter = 0
    for neighbor in neighbors:
        # # check if neighbor is a valid point
        # if not nan_mask[neighbor[0], neighbor[1]]:
        #     continue
        for neighbor2 in neighbors[neighbor]:
            # # check if neighbor2 is a valid point
            # if not nan_mask[neighbor2[0], neighbor2[1]]:
            #     continue
            if neighbor != neighbor2:
                counter += 1
                try:
                    grid_graph.add_edge(neighbor, neighbor2)
                except nx.NetworkXError:
                    logger.warning(f"Edge between {neighbor} and {neighbor2} already exists or is invalid")
    return grid_graph


def generate_cluster_graph(clustering, grid_graph, lat_lon_to_grid_point_id):
    """
    Generate a graph from the clustering data and the grid graph, in which there is an edge between two nodes if they
    belong to the same cluster and are neighbors in the grid graph
    :param clustering: Mapping of cluster_id to a list of (lat, lon) coordinates
    :param grid_graph: grid graph with spatial relationships
    :param lat_lon_to_grid_point_id: Mapping from (lat, lon) to grid point IDs
    :return: Cluster graph where edges connect same-cluster neighbors
    """
    # Step 1: Create a mapping of lat/lon to grid point id
    # map lat/lon to grid point id in clustering
    cluster_id_to_grid_point_set = {}
    # List to store (node_id, attribute_dict) for batch node addition.
    # Ensures each node is added once with its cluster_id.
    nodes_to_add_to_graph = []
    # Keep track of grid_points already prepared for nodes_to_add_to_graph
    processed_grid_points = set()
    for cluster_id, lat_lon_list in clustering.items():
        current_cluster_grid_points_set = set()
        for lat_lon in lat_lon_list:
            grid_point_id = lat_lon_to_grid_point_id.get(lat_lon)
            if grid_point_id is not None:
                current_cluster_grid_points_set.add(grid_point_id)
                if grid_point_id not in processed_grid_points:
                    nodes_to_add_to_graph.append((grid_point_id, {"cluster_id": cluster_id}))
                    processed_grid_points.add(grid_point_id)

            if current_cluster_grid_points_set:  # Only store if the cluster has valid grid points
                cluster_id_to_grid_point_set[cluster_id] = current_cluster_grid_points_set

    # Step 2: Create the cluster graph
    # create a cluster graph where a pair of nodes is connected if they belong to the same cluster and have an edge in the grid graph
    cluster_graph = nx.Graph()
    # Add all nodes to the graph in one batch operation
    if nodes_to_add_to_graph:
        cluster_graph.add_nodes_from(nodes_to_add_to_graph)

    edges_to_add_to_graph = set()
    for cluster_id, grid_points_in_cluster_set in cluster_id_to_grid_point_set.items():
        for grid_point in grid_points_in_cluster_set:
            neighbors = grid_graph.neighbors(grid_point)  # get potential neighbors from grid graph
            for neighbor in neighbors:
                if neighbor in grid_points_in_cluster_set:  # check if a neighbor is in the same cluster and is already in the cluster graph
                    # Add edge. Ensure consistent order to avoid (u,v) and (v,u) duplicates in the set.
                    # NetworkX's add_edges_from handles this, but for the set, it's good practice.
                    if grid_point < neighbor:
                        edges_to_add_to_graph.add((grid_point, neighbor))
                    else:
                        edges_to_add_to_graph.add((neighbor, grid_point))
    # Add all collected edges in one batch operation
    if edges_to_add_to_graph:
        cluster_graph.add_edges_from(list(edges_to_add_to_graph))
    return cluster_graph


def generate_connected_component_graph(cluster_graph, grid_graph):
    """
    Generate a connected component graph from the cluster graph, in which each node is a connected component, and there
    is an edge between nodes if any of their nodes are neighbors in the grid graph
    :param cluster_graph:
    :param grid_graph:
    :return:
    """
    connected_component_graph = nx.Graph()

    component_id_to_object_map = {}  # Stores ConnectedComponent objects by their ID
    node_to_component_id_map = {}  # Maps original grid node ID to its component's UUID

    # Step 1: Identify components, create ConnectedComponent objects, and populate maps
    for node_set_for_component in nx.connected_components(cluster_graph):
        if not node_set_for_component:  # Should not happen with non-empty components
            logger.warning("Empty node set found in connected components. Skipping...")
            continue
        first_node = next(iter(node_set_for_component))
        cluster_id = cluster_graph.nodes[first_node]["cluster_id"]
        # ConnectedComponent class takes (id, nodes_set, original_cluster_attr, size)
        current_component = ConnectedComponent(uuid.uuid4(), node_set_for_component, cluster_id,
                                               len(node_set_for_component))
        connected_component_graph.add_node(current_component.id, component=current_component)
        component_id_to_object_map[current_component.id] = current_component
        for node in node_set_for_component:
            node_to_component_id_map[node] = current_component.id

    # Step 2: Add edges between components if their constituent nodes are neighbors in the grid
    for edge in grid_graph.edges():
        node1 = edge[0]
        node2 = edge[1]
        if node1 in node_to_component_id_map and node2 in node_to_component_id_map:
            connected_component_1_id = node_to_component_id_map[node1]
            connected_component_2_id = node_to_component_id_map[node2]
            if connected_component_1_id != connected_component_2_id:
                connected_component_graph.add_edge(connected_component_1_id, connected_component_2_id)

    return connected_component_graph, component_id_to_object_map
