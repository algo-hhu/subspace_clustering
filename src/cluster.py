from loguru import logger
from sqlalchemy import Column, Integer
from sqlalchemy.orm import relationship

from src.cluster_grid_point_relationship import cluster_grid_points
from src.grid_point import Base


class Cluster(Base):
    __tablename__ = "clusters"
    id = Column(Integer, primary_key=True)
    grid_points = relationship("GridPoint", secondary=cluster_grid_points, back_populates="clusters")

    def add_grid_point(self, grid_point, session):
        if grid_point not in self.grid_points:
            self.grid_points.append(grid_point)
            session.flush()  # Flushes changes to the database
        else:
            logger.warning(f"Grid point {grid_point} already in cluster {self.id}")
