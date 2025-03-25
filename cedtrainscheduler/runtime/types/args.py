from dataclasses import dataclass

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.types.cluster import ClusterType
from cedtrainscheduler.runtime.types.cluster import GPUType


@dataclass
class MasterArgs:
    master_info: ComponentInfo
    manager_info: ComponentInfo
    cluster_name: str
    cluster_type: ClusterType


@dataclass
class WorkerArgs:
    worker_info: ComponentInfo
    master_info: ComponentInfo
    gpu_type: GPUType


@dataclass
class ManagerArgs:
    manager_info: ComponentInfo
    scheduler_name: str
