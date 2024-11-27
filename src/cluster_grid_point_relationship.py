# src/cluster_grid_point_relationship.py
from sqlalchemy import Table, Column, Integer, ForeignKey

from src.base import Base

cluster_grid_points = Table(
    "cluster_grid_points", Base.metadata,
    Column("cluster_id", Integer, ForeignKey("clusters.id"), primary_key=True),
    Column("grid_point_id", Integer, ForeignKey("grid_points.id"), primary_key=True)
)
