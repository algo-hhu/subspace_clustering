import uuid
from dataclasses import dataclass, field

from src.clustering.cluster_entities.connected_component import ConnectedComponent


@dataclass
class Cluster:
    """
    A class to represent a cluster of grid points.
    """

    components: [ConnectedComponent]
    size: int
    id: uuid.UUID = field(default_factory=uuid.uuid4)
