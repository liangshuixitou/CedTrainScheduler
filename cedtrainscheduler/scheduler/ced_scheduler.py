from cedtrainscheduler.scheduler.factory import SchedulerType
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.executor import GPUExecutor
from cedtrainscheduler.simulator.fs import TaskDataInfo


class CEDScheduler(SchedulerBase):
    def __init__(self):
        super().__init__(SchedulerType.CED)
        self.task_queue: list[TaskMeta] = []

    def sort_task_queue(self):
        self.task_queue.sort(key=lambda x: x.task_runtime["T4"])

    def schedule(
        self,
        current_time: float,
        clusters: dict[str, Cluster],
        gpu_task_queue: dict[str, GPUExecutor],
        task_data_info: dict[str, TaskDataInfo],
        task_record: dict[str, TaskWrapRuntimeInfo],
    ) -> tuple[TaskWrapRuntimeInfo, bool]:
        pass
