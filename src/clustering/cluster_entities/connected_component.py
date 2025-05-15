import uuid
from dataclasses import dataclass

from src.clustering.cluster_entities.grid_point import GridPoint


@dataclass
class ConnectedComponent:
    """
    A class to represent a component of a grid point.
    """

    nodes: [GridPoint]
    edges: [(GridPoint, GridPoint)]
    cluster_id: uuid.UUID
    size: int
