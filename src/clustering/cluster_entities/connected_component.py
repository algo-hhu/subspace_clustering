import uuid
from dataclasses import dataclass


@dataclass
class ConnectedComponent:
    """
    A class to represent a component of a grid point.
    """
    id: uuid.UUID
    nodes: set[uuid.UUID]
    cluster_id: float
    size: int
