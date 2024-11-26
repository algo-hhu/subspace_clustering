from sqlalchemy import Table, Column, ForeignKey, Integer

from src.grid_point import Base

# Join table for clusters and grid points
cluster_grid_points = Table(
    "cluster_grid_points",
    Base.metadata,
    Column("cluster_id", Integer, ForeignKey("clusters.id"), primary_key=True),
    Column("grid_point_id", Integer, ForeignKey("grid_points.id"), primary_key=True)
)
