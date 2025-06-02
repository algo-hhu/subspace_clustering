from dataclasses import dataclass


@dataclass
class ConnectedComponent:
    """
    A class to represent a component of a grid point.
    """
    id: int
    nodes: set[(float, float)]
    cluster_id: float
    size: int
