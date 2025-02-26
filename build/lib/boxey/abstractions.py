from abc import ABC
from typing import Dict

class BoxeyProcess(ABC):
    name: str

    def get_k(t: float) -> float:
        """First order rate at time `t`."""

class BoxeyModel(ABC):
    processes: Dict[str, BoxeyProcess]