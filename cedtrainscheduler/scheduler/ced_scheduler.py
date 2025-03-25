import math  # noqa: I001

from cedtrainscheduler.scheduler.factory import SchedulerType

from cedtrainscheduler.scheduler.policy.ced_policy import CedCentralPolicy, CedClusterPolicy
from cedtrainscheduler.scheduler.policy.ced_policy import CedQueuePolicy
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo


class CedScheduler(SchedulerBase):
    def __init__(self):
        super().__init__()
        self.scheduler_name = SchedulerType.CED
        self.queue_policy = CedQueuePolicy()
        self.central_policy = CedCentralPolicy()
        self.cluster_policy = CedClusterPolicy()

    def submit_task(self, task: TaskMeta):
        self.queue_policy.add_task([task])

    def schedule(
        self,
        scheduler_context: SchedulerContext,
    ) -> tuple[TaskWrapRuntimeInfo, bool]:
        if len(self.task_queue) == 0:
            return None, True
        scheduler_context.task_queue = self.task_queue

        task = self.queue_policy.pop_one_task(scheduler_context)
        cluster_id = self.central_policy.schedule(scheduler_context, task)
        schedule_infos = self.cluster_policy.schedule(scheduler_context, task, cluster_id)

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
