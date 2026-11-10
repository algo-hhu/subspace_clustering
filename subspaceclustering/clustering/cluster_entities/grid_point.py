import uuid
from dataclasses import dataclass


@dataclass
class GridPoint:
    """

    """
    id: uuid.UUID
    latitude: float
    longitude: float
    time_series: [float]
    connected_component_id: uuid.UUID
