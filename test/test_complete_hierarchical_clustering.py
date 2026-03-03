import uuid
from unittest import TestCase

import numpy as np

from src.clustering.complete_hierarchical_clustering import GridPoint, Cluster, CompleteHierarchicalClustering
from src.distance import euclidean_distance


class Test(TestCase):
    def test_hierarchical_clustering(self):
        grid_point1 = GridPoint(id=uuid.uuid4(), latitude=0, longitude=0, 
        timeseries= np.array([0, 4, 5]))
        grid_point2 = GridPoint(id=uuid.uuid4(), latitude=0, longitude=1, timeseries=np.array([4, 2, 3]))
        grid_point3 = GridPoint(id=uuid.uuid4(), latitude=1, longitude=0, timeseries=np.array([5, 2, 4]))
        cluster1 = Cluster(id=0, grid_points=[grid_point1])
        cluster2 = Cluster(id=1, grid_points=[grid_point2])
        cluster3 = Cluster(id=2, grid_points=[grid_point3])
        clusters = {0: cluster1, 1: cluster2, 2: cluster3}
        distances = np.full((len(clusters.values()), len(clusters.values())), np.nan)
        distances[0, 1] = 1
        distances[1, 0] = 1
        distances[0, 2] = 2
        distances[2, 0] = 2
        distances[1, 2] = 3
        distances[2, 1] = 3
        sea_level_anomaly_data = None
        number_of_clusters = len(clusters.values())
        k = [1]
        hierarchical_clustering = CompleteHierarchicalClustering(sea_level_anomaly_data,k , euclidean_distance, "", 0.0 )
        clusterings = hierarchical_clustering.clustering(distances, clusters)
        self.assertEqual(1, len(clusterings))
        self.assertEqual(1, len(clusterings[1]))
        self.assertEqual(3, len(clusterings[1][0].grid_points))
        assert grid_point1 in clusterings[1][0].grid_points
        assert grid_point2 in clusterings[1][0].grid_points
        assert grid_point3 in clusterings[1][0].grid_points

    def test_hierarchical_clusterin_with_20_grid_points(self):
        grid_point0 = GridPoint(id=uuid.uuid4(), latitude=-1, longitude=-1, timeseries=np.array([0, 4, 5]))
        grid_point1 = GridPoint(id=uuid.uuid4(), latitude=0, longitude=0, timeseries=np.array([0, 4, 5]))
        grid_point2 = GridPoint(id=uuid.uuid4(), latitude=0, longitude=1, timeseries=np.array([4, 2, 3]))
        grid_point3 = GridPoint(id=uuid.uuid4(), latitude=1, longitude=0, timeseries=np.array([5, 2, 4]))
        grid_point4 = GridPoint(id=uuid.uuid4(), latitude=1, longitude=1, timeseries=np.array([5, 2, 4]))
        grid_point5 = GridPoint(id=uuid.uuid4(), latitude=2, longitude=0, timeseries=np.array([5, 2, 4]))
        grid_point6 = GridPoint(id=uuid.uuid4(), latitude=2, longitude=1, timeseries=np.array([5, 2, 4]))
        grid_point7 = GridPoint(id=uuid.uuid4(), latitude=3, longitude=0, timeseries=np.array([5, 2, 4]))
        grid_point8 = GridPoint(id=uuid.uuid4(), latitude=3, longitude=1, timeseries=np.array([5, 2, 4]))
        grid_point9 = GridPoint(id=uuid.uuid4(), latitude=4, longitude=0, timeseries=np.array([5, 2, 4]))
        grid_point10 = GridPoint(id=uuid.uuid4(), latitude=4, longitude=1, timeseries=np.array([5, 2, 4]))
        grid_point11 = GridPoint(id=uuid.uuid4(), latitude=5, longitude=0, timeseries=np.array([5, 2, 4]))
        grid_point12 = GridPoint(id=uuid.uuid4(), latitude=5, longitude=1, timeseries=np.array([5, 2, 4]))
        grid_point13 = GridPoint(id=uuid.uuid4(), latitude=6, longitude=0, timeseries=np.array([5, 2, 4]))
        grid_point14 = GridPoint(id=uuid.uuid4(), latitude=6, longitude=1, timeseries=np.array([5, 2, 4]))
        grid_point15 = GridPoint(id=uuid.uuid4(), latitude=7, longitude=0, timeseries=np.array([5, 2, 4]))
        grid_point16 = GridPoint(id=uuid.uuid4(), latitude=7, longitude=1, timeseries=np.array([5, 2, 4]))
        grid_point17 = GridPoint(id=uuid.uuid4(), latitude=8, longitude=0, timeseries=np.array([5, 2, 4]))
        grid_point18 = GridPoint(id=uuid.uuid4(), latitude=8, longitude=1, timeseries=np.array([5, 2, 4]))
        grid_point19 = GridPoint(id=uuid.uuid4(), latitude=9, longitude=0, timeseries=np.array([5, 2, 4]))
        cluster0 = Cluster(id=0, grid_points=[grid_point0])
        cluster1 = Cluster(id=1, grid_points=[grid_point1])
        cluster2 = Cluster(id=2, grid_points=[grid_point2])
        cluster3 = Cluster(id=3, grid_points=[grid_point3])
        cluster4 = Cluster(id=4, grid_points=[grid_point4])
        cluster5 = Cluster(id=5, grid_points=[grid_point5])
        cluster6 = Cluster(id=6, grid_points=[grid_point6])
        cluster7 = Cluster(id=7, grid_points=[grid_point7])
        cluster8 = Cluster(id=8, grid_points=[grid_point8])
        cluster9 = Cluster(id=9, grid_points=[grid_point9])
        cluster10 = Cluster(id=10, grid_points=[grid_point10])
        cluster11 = Cluster(id=11, grid_points=[grid_point11])
        cluster12 = Cluster(id=12, grid_points=[grid_point12])
        cluster13 = Cluster(id=13, grid_points=[grid_point13])
        cluster14 = Cluster(id=14, grid_points=[grid_point14])
        cluster15 = Cluster(id=15, grid_points=[grid_point15])
        cluster16 = Cluster(id=16, grid_points=[grid_point16])
        cluster17 = Cluster(id=17, grid_points=[grid_point17])
        cluster18 = Cluster(id=18, grid_points=[grid_point18])
        cluster19 = Cluster(id=19, grid_points=[grid_point19])
        clusters = {0: cluster0, 1: cluster1, 2: cluster2, 3: cluster3, 4: cluster4, 5: cluster5, 6: cluster6,
                    7: cluster7,
                    8: cluster8, 9: cluster9, 10: cluster10, 11: cluster11, 12: cluster12, 13: cluster13, 14: cluster14,
                    15: cluster15, 16: cluster16, 17: cluster17, 18: cluster18, 19: cluster19}
        distances = np.full((len(clusters.values()), len(clusters.values())), np.nan)
        for i in range(len(clusters.values())):
            for j in range(len(clusters.values())):
                if i != j:
                    distances[i, j] = i + j
        sea_level_anomaly_data = None
        number_of_clusters = len(clusters.values())
        k = [3]
        hierarchical_clustering = CompleteHierarchicalClustering(sea_level_anomaly_data, k, euclidean_distance, "", 0.0)
        clusterings = hierarchical_clustering.clustering(distances, clusters)
        self.assertEqual(1, len(clusterings))
        self.assertEqual(3, len(clusterings[3]))
