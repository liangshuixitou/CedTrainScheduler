from dataclasses import dataclass
from enum import Enum

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.types.cluster import ClusterType
from cedtrainscheduler.runtime.types.cluster import GPUType


class ExecutorType(str, Enum):
    PYTHON = "python"
    DOCKER = "docker"


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
    executor_type: ExecutorType
    python_path: str
    gpu_ids: list[int]
    sim_gpu_num: int


@dataclass
class ManagerArgs:
    manager_info: ComponentInfo
    scheduler_name: str
    fs_config_path: str
