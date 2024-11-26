from sqlalchemy import Integer, ARRAY, Float, Column
from sqlalchemy.orm import declarative_base, relationship

from src.cluster_grid_point_relationship import cluster_grid_points

Base = declarative_base()


class GridPoint(Base):
    __tablename__ = "grid_points"
    id = Column(Integer, primary_key=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    sea_level_anomaly_values = Column(ARRAY(Float), nullable=False)
    clusters = relationship("Cluster", secondary=cluster_grid_points, back_populates="grid_points")
