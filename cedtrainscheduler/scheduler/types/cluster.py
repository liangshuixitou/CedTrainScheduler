from dataclasses import dataclass
from enum import Enum


class ClusterType(Enum):
    CLOUD = "cloud"
    EDGE = "edge"
    TERMINAL = "terminal"

class GPUType(Enum):
    V100 = "V100"
    T4 = "T4"
    P100 = "P100"

@dataclass
class GPU:
    gpu_id: str
    gpu_type: GPUType

@dataclass
class Node:
    node_id: str
    cpu_cores: int
    memory: int
    ip_address: str
    gpus: list[GPU]

@dataclass
class Cluster:
    cluster_id: str
    cluster_name: str
    cluster_type: ClusterType
    nodes: list[Node]
    intra_domain_bandwidth: int  # 集群内带宽(Mbps)
    inter_domain_bandwidth: int  # 集群间带宽(Mbps)


