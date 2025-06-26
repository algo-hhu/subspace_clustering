from abc import ABC, abstractmethod
from dataclasses import dataclass

import xarray as xr


@dataclass
class InitialClustering(ABC):
    sea_level_anomaly_data: xr.Dataset
    number_of_clusters: list[int]
    distance_function: callable
    out_dir: str

    @abstractmethod
    def start_initial_clustering(self) -> None:
        pass
