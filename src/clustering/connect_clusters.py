import time

import numpy
import numpy as np
import xarray
from loguru import logger

from src.clustering.connectivity_helper import generate_grid_graph, generate_cluster_graph, \
    generate_connected_component_graph
from src.distance import subspace_timeseries_distance_calculation
from src.plotting import plot_graph_on_clustering_map, plot_with_highlighting_of_component, \
    plot_clustering_with_component_graph


def reestablish_connectivity(sea_level_anomaly_data: xarray.Dataset, clustering, subspaces,
                             iteration_count: int,
                             out_dir, cluster_id_to_color, number_of_clusters: int):
    """
    Reestablish connectivity in the clusters
    TODO: change grid-point-id to lat, lon?
    :param number_of_clusters:
    :param sea_level_anomaly_data: xarray dataset with sea level anomaly data
    :param clustering: dictionary with cluster ids as keys and list of grid points as values
    :param subspaces: dictionary with cluster ids as keys and tuple of (subspace, mean) as values
    :param iteration_count: iteration number for logging purposes
    :param out_dir: output directory to save plots
    :param cluster_id_to_color: dictionary with cluster ids as keys and colors as values
    :return:
    """
    # # use profiler to find bottlenecks
    # profiler = cProfile.Profile()
    # profiler.enable()
    current_time = time.time()
    data = sea_level_anomaly_data["sla"].values
    lat_lon_to_grid_point_id = {}  # {lat, lon: grid_point_id}
    latitudes = sea_level_anomaly_data.latitude.values
    longitudes = sea_level_anomaly_data.longitude.values
    lat_range = len(latitudes)
    long_range = len(longitudes)
    resolution = latitudes[1] - latitudes[0]
    nan_mask = numpy.isnan(data).any(axis=0)
    for i in (range(lat_range)):
        for j in (range(long_range)):
            if nan_mask[i, j]:  # points without valid data can be skipped
                continue
            lat_lon_to_grid_point_id[(latitudes[i], longitudes[j])] = (i, j)
    grid_point_to_lat_lon = {v: k for k, v in lat_lon_to_grid_point_id.items()}

    grid_graph = generate_grid_graph(lat_lon_to_grid_point_id, nan_mask, sea_level_anomaly_data)

    cluster_graph = generate_cluster_graph(clustering, grid_graph,
                                           lat_lon_to_grid_point_id)  # graph with grid points as nodes and edges
    # between points in the same cluster

    connected_component_graph, connected_components = generate_connected_component_graph(cluster_graph,
                                                                                         grid_graph)  # connected
    # component graph with connected components (i.e. clusters) as nodes and edges between connected components that
    # are neighbors
    # plot each graph
    plot_graph_on_clustering_map(clustering, cluster_graph, grid_point_to_lat_lon, resolution, out_dir,
                                 "cluster_graph_initial",
                                 cluster_id_to_color)

    plot_clustering_with_component_graph(clustering, out_dir, resolution, "component_graph_initial",
                                         connected_component_graph, connected_components, grid_point_to_lat_lon,
                                         cluster_id_to_color)

    counter = 0
    while len(connected_components) > number_of_clusters:
        counter += 1
        current_number_of_components = len(connected_components)
        # recalculate connected components & connected component graph and start again
        cluster_graph = generate_cluster_graph(clustering, grid_graph, lat_lon_to_grid_point_id)
        connected_component_graph, connected_components = generate_connected_component_graph(cluster_graph, grid_graph)

        # extract the smallest component
        sorted_connected_components_list = sorted(connected_components.values(), key=lambda c: c.size)
        smallest_connected_component = sorted_connected_components_list[0]
        neighbors = list(connected_component_graph.neighbors(smallest_connected_component.id))
        # plot_with_highlighting_of_component(clustering, smallest_connected_component, neighbors, out_dir, counter,
        #                                     resolution, connected_components, grid_point_to_lat_lon,
        #                                     cluster_id_to_color)
        best_neighbor = None
        neighbor_count = {}
        for neighbor in neighbors:
            subspace_id = connected_components[neighbor].cluster_id
            neighbor_count[subspace_id] = 0
        for node in smallest_connected_component.nodes:
            time_series = data[:, node[0], node[1]]
            # find the closest subspace
            min_error = np.inf
            closest_cluster = None
            for subspace_id in neighbor_count:
                if subspace_id not in subspaces:
                    logger.warning(f"subspace {subspace_id} not found in subspaces")
                    continue
                subspace, mean = subspaces[subspace_id]
                distance = subspace_timeseries_distance_calculation([], time_series, mean, subspace)
                if distance < min_error:
                    min_error = distance
                    closest_cluster = subspace_id
            if closest_cluster is None:
                logger.warning(f"no closest cluster found for node {node}")
                continue
            neighbor_count[closest_cluster] += 1
        # assign the current connected component to the neighbor that it is most similar to
        if not neighbor_count:
            # plot the image with the smallest connected component highlighted
            plot_with_highlighting_of_component(clustering, smallest_connected_component, neighbors,
                                                f"{out_dir}/deleted_nodes",
                                                f"{iteration_count}smallest_component_{counter}_error", resolution,
                                                connected_components, grid_point_to_lat_lon, cluster_id_to_color)
            # if there are no neighbors, it means that the smallest connected component is isolated, then it can be
            # removed and ignored in further iterations
            grid_graph.remove_nodes_from(smallest_connected_component.nodes)
            # remove the smallest connected component from the clustering
            for node in smallest_connected_component.nodes:
                lat, lon = grid_point_to_lat_lon[node]
                if smallest_connected_component.cluster_id in clustering:
                    if (lat, lon) in clustering[smallest_connected_component.cluster_id]:
                        # remove the node from the cluster
                        clustering[smallest_connected_component.cluster_id].remove((lat, lon))
                    else:
                        logger.warning(f"node {node} not found in cluster {smallest_connected_component.cluster_id}")
            continue

        best_neighbor = max(neighbor_count, key=neighbor_count.get)
        # assign the cluster id of the best neighbor to all points in the smallest connected component

        if not len(connected_components) < current_number_of_components and counter > 1:
            logger.warning(f"did not reduce number of components")
            # plot edges that smallest component has to neighbors
            print(f"neighbors of smallest connected component: {smallest_connected_component.id} - {neighbors}")
            print(f"nodes of smallest connected component: {smallest_connected_component.nodes}")
            for neighbor in neighbors:
                print(f"neighbor nodes: {connected_components[neighbor].nodes}")
            plot_with_highlighting_of_component(clustering, smallest_connected_component, neighbors, out_dir,
                                                f"{iteration_count}smallest_component_{counter}_error", resolution,
                                                connected_components,
                                                grid_point_to_lat_lon, cluster_id_to_color)
            # plot clustergraph
            plot_graph_on_clustering_map(clustering, cluster_graph, grid_point_to_lat_lon, resolution, out_dir,
                                         f"{iteration_count}cluster_graph_{counter}_error", cluster_id_to_color)
            exit()

        # if counter % 100 == 0:
        #     print(
        #         f"number of connected components in cluster graph: {nx.number_connected_components(cluster_graph)}
        #         and are in the component graph: {len(connected_component_graph.nodes)}")

        for node in smallest_connected_component.nodes:
            # # change assignment in cluster array
            # cluster_array[node[0], node[1]] = best_neighbor
            # remove node from original cluster and add it to the best neighbor
            (lat, lon) = grid_point_to_lat_lon[node]
            clustering[smallest_connected_component.cluster_id].remove((lat, lon))
            clustering[best_neighbor].append((lat, lon))

    all_cluster_ids = set(range(number_of_clusters))
    empty_cluster_ids = set()
    cluster_ids = set()
    for connected_component in connected_components.values():
        cluster_ids.add(connected_component.cluster_id)
    for cluster in clustering.keys():
        if cluster not in cluster_ids:
            empty_cluster_ids.add(cluster)
    # if there are less clusters than expected, add from the cluster ids that should be there
    if len(cluster_ids) < number_of_clusters:
        empty_cluster_ids = empty_cluster_ids.union(all_cluster_ids - cluster_ids)

    new_clustering = {}
    new_clustering_with_lat_lon = {}

    counter = 0

    for connected_component in connected_components.values():
        current_cluster_id = None
        if connected_component.cluster_id in cluster_ids:
            current_cluster_id = connected_component.cluster_id
            cluster_ids.remove(current_cluster_id)
        else:
            if not empty_cluster_ids:
                # new id for  cluster
                current_cluster_id = len(cluster_ids) + counter
                counter += 1
            else:
                current_cluster_id = empty_cluster_ids.pop()
        new_clustering[current_cluster_id] = []
        new_clustering_with_lat_lon[current_cluster_id] = []
        for node in connected_component.nodes:
            grid_point = tuple(node)
            lat_lon = grid_point_to_lat_lon[grid_point]
            new_clustering[current_cluster_id].append(grid_point)
            new_clustering_with_lat_lon[current_cluster_id].append(lat_lon)

    # plot clustering
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
    # plot_clustering(new_clustering_with_lat_lon, out_dir, resolution,
    #                 f"reestablished_connectivity_after {counter} iterations",
    #                 cluster_id_to_color)
    # exit()
    return new_clustering
