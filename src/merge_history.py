# src/merge_history.py
from sqlalchemy import Column, Integer, ForeignKey, Float

from src.base import Base


class MergeHistory(Base):
    __tablename__ = "merge_history"
    id = Column(Integer, primary_key=True)
    iteration = Column(Integer, nullable=False)  # The iteration number
    cluster_1_id = Column(Integer, ForeignKey('clusters.id'))
    cluster_2_id = Column(Integer, ForeignKey('clusters.id'))
    new_cluster_id = Column(Integer, ForeignKey('clusters.id'))
    difference = Column(Float, nullable=False)  # Distance between clusters when merged
