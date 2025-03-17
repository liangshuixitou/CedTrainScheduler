from dataclasses import dataclass
from enum import Enum


class ComponentType(str, Enum):
    MASTER = "master"
    SCHEDULER = "scheduler"
    WORKER = "worker"


@dataclass
class ComponentInfo:
    component_type: ComponentType
    component_id: str
    component_ip: str
    component_port: int
