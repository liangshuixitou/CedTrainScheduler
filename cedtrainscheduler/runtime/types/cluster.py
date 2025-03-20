from dataclasses import dataclass
from enum import Enum


class ClusterType(str, Enum):
    CLOUD = "cloud"
    EDGE = "edge"
    TERMINAL = "terminal"


class GPUType(str, Enum):
    V100 = "V100"
    P100 = "P100"
    T4 = "T4"


@dataclass
class GPU:
    gpu_id: str
    gpu_type: GPUType
    gpu_rank: int
    node_id: str


@dataclass
class Node:
    node_id: str
    ip: str
    port: int
    cluster_id: str
    gpus: dict[str, GPU]


@dataclass
class Cluster:
    cluster_id: str
    cluster_name: str
    cluster_type: ClusterType
    nodes: dict[str, Node]
