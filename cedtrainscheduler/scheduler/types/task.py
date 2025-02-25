from dataclasses import dataclass
from enum import Enum

from cedtrainscheduler.scheduler.types.cluster import GPUType


class TaskStatus(str, Enum):
    Pending = "pending"
    Ready = "ready"
    Running = "running"
    Finished = "finished"


class TaskInstStatus(str, Enum):
    Pending = "pending"
    Ready = "ready"
    Running = "running"
    Finished = "finished"


class TaskInstDataStatus(str, Enum):
    Pending = "pending"
    Running = "running"
    Finished = "finished"


@dataclass
class TaskInst:
    task_id: str
    inst_id: int
    inst_status: TaskStatus


@dataclass
class TaskMeta:
    # task metadata
    task_id: str
    task_name: str
    task_inst_num: int
    task_plan_cpu: float
    task_plan_mem: float
    task_plan_gpu: int
    task_status: TaskStatus
    task_runtime: dict[GPUType, float]


@dataclass
class ScheduleInfo:
    # for each inst in task
    inst_id: int
    gpu_id: str


@dataclass
class TaskWrapRuntimeInfo:
    task_meta: TaskMeta
    schedule_infos: dict[int, ScheduleInfo]
    inst_status: dict[int, TaskInstStatus]
    inst_data_status: dict[int, TaskInstDataStatus]
    task_submit_time: float
    task_start_time: float
    task_end_time: float
