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


class GPUPerformance(float, Enum):
    T4_PERFORMANCE = 8.1
    P100_PERFORMANCE = 9.3
    V100_PERFORMANCE = 15.7


CLUSTER_TYPE_GPU_MAP = {
    ClusterType.CLOUD: GPUType.V100,
    ClusterType.EDGE: GPUType.P100,
    ClusterType.TERMINAL: GPUType.T4,
}

GPU_PERFORMANCE_MAP = {
    GPUType.T4: GPUPerformance.T4_PERFORMANCE,
    GPUType.P100: GPUPerformance.P100_PERFORMANCE,
    GPUType.V100: GPUPerformance.V100_PERFORMANCE,
}


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
