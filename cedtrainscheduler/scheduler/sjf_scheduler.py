import math

from cedtrainscheduler.scheduler.factory import SchedulerType
from cedtrainscheduler.scheduler.policy.central_policy import ResourceAffinityPolicy
from cedtrainscheduler.scheduler.policy.cluster_policy import GreedyPolicy
from cedtrainscheduler.scheduler.policy.queue_policy import SFJQueuePolicy
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record


class SJFScheduler(SchedulerBase):
    def __init__(self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem):
        super().__init__(SchedulerType.SJF, cluster_manager, task_record, file_system)
        self.queue_policy = SFJQueuePolicy(cluster_manager, task_record, file_system, self.task_queue)
        self.central_policy = ResourceAffinityPolicy(cluster_manager, task_record, file_system)
        self.cluster_policy = GreedyPolicy(cluster_manager, task_record, file_system)

    def submit_task(self, task: TaskMeta):
        self.queue_policy.add_task([task])

    def schedule(
        self,
        current_time: float,
    ) -> tuple[TaskWrapRuntimeInfo, bool]:
        if len(self.task_queue) == 0:
            return None, True

        task = self.queue_policy.pop_one_task(current_time)

        cluster_ids = self.central_policy.schedule(current_time, task)
        schedule_infos = self.cluster_policy.schedule(current_time, task, cluster_ids)

        # 创建任务运行时信息
        runtime_info = TaskWrapRuntimeInfo(
            task_meta=task,
            schedule_infos=schedule_infos,
            inst_status={},
            inst_data_status={},
            task_submit_time=task.task_start_time,
            task_start_time=-math.inf,
            task_end_time=-math.inf,
        )

        return runtime_info, False
