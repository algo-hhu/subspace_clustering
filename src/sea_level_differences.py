from sqlalchemy import Column, Integer, ForeignKey, FLOAT, Index, UniqueConstraint
from sqlalchemy.orm import relationship

from src.grid_point import Base


# enforce that there are no cluster difference duplicates

class Difference(Base):
    __tablename__ = "differences"
    id = Column(Integer, primary_key=True)
    cluster_1_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    cluster_2_id = Column(Integer, ForeignKey("clusters.id"), nullable=False)
    difference = Column(FLOAT, nullable=False)
    cluster_1 = relationship("Cluster", foreign_keys=[cluster_1_id])
    cluster_2 = relationship("Cluster", foreign_keys=[cluster_2_id])

    __table_args__ = (
        Index("idx_cluster_pairs", "cluster_1_id", "cluster_2_id"),
        Index("idx_distance", "distance"),
        UniqueConstraint("cluster_1_id", "cluster_2_id", name="unique_cluster_pair"),
    )
