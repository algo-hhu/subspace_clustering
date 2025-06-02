from unittest import TestCase

import networkx as nx

from src.clustering.connectivity_helper import map_locations_to_ids, build_cluster_graph_nodes, \
    build_cluster_graph_edges, generate_cluster_graph


class Test(TestCase):
    def test_map_locations_to_ids_basic(self):
        # cluster 1 has two known locations, cluster 2 has none in the map
        cluster_to_locations = {
            1: [(0.0, 0.0), (1.0, 1.0)],
            2: [(2.0, 2.0)]
        }
        location_to_id = {
            (0.0, 0.0): 10,
            (1.0, 1.0): 11
        }
        expected = {1: {10, 11}}
        result = map_locations_to_ids(cluster_to_locations, location_to_id)
        assert result == expected

    def test_build_cluster_graph_nodes_order_irrelevant(self):
        # nodes should carry the cluster_id attribute
        cluster_to_grid_ids = {
            1: {10, 11},
            2: {20}
        }
        nodes = build_cluster_graph_nodes(cluster_to_grid_ids)
        # turn [(node, {"cluster_id": cid}), …] into {(node, cid), …}
        actual_node_cluster_pairs = {
            (node_id, attrs["cluster_id"])
            for node_id, attrs in nodes
        }

        expected_node_cluster_pairs = {
            (10, 1),
            (11, 1),
            (20, 2),
        }

        assert actual_node_cluster_pairs == expected_node_cluster_pairs

    def test_build_cluster_graph_edges_edge_selection(self):
        # Only edges between same‐cluster neighbors
        # Build a grid graph with edges (1-2), (2-3), (3-4)
        grid = nx.Graph()
        grid.add_edges_from([(1, 2), (2, 3), (3, 4)])
        # Suppose cluster 1 contains {1,2,3}, cluster 2 contains {4}
        cluster_to_ids = {1: {1, 2, 3}, 2: {4}}
        edges = build_cluster_graph_edges(cluster_to_ids, grid)
        # Expect only (1,2) and (2,3)
        assert edges == {(1, 2), (2, 3)}

    def test_generate_cluster_graph_end_to_end(self):
        # Define clusters by lat/lon
        cluster_to_locations = {
            100: [(0.0, 0.0), (0.0, 1.0)],
            200: [(1.0, 0.0)]
        }
        # Map each location to a unique integer ID
        location_to_id = {
            (0.0, 0.0): 1,
            (0.0, 1.0): 2,
            (1.0, 0.0): 3
        }
        # Create a grid graph connecting 1–2, 2–3, 1–3
        grid = nx.Graph()
        grid.add_edges_from([(1, 2), (2, 3), (1, 3)])

        G = generate_cluster_graph(cluster_to_locations, grid, location_to_id)

        # Check nodes & their cluster_id attrs
        assert set(G.nodes) == {1, 2, 3}
        assert G.nodes[1]["cluster_id"] == 100
        assert G.nodes[2]["cluster_id"] == 100
        assert G.nodes[3]["cluster_id"] == 200

        # Only 1–2 is a same‐cluster edge
        assert set(G.edges) == {(1, 2)}

    def test_empty_clusters_and_locations(self):
        # Edge cases: no clusters, no locations
        assert map_locations_to_ids({}, {}) == {}
        assert build_cluster_graph_nodes({}) == []
        empty_graph = nx.Graph()
        assert build_cluster_graph_edges({}, empty_graph) == set()
        # generate_cluster_graph on empty inputs yields empty graph
        G = generate_cluster_graph({}, empty_graph, {})
        assert len(G.nodes) == 0
        assert len(G.edges) == 0
