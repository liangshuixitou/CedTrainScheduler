import math

from cedtrainscheduler.scheduler.factory import SchedulerType
from cedtrainscheduler.scheduler.policy.central_policy import ResourceAffinityPolicy
from cedtrainscheduler.scheduler.policy.cluster_policy import GreedyPolicy
from cedtrainscheduler.scheduler.policy.queue_policy import SFJQueuePolicy
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo


class SJFScheduler(SchedulerBase):
    def __init__(self):
        super().__init__()
        self.scheduler_name = SchedulerType.SJF
        self.queue_policy = SFJQueuePolicy()
        self.central_policy = ResourceAffinityPolicy()
        self.cluster_policy = GreedyPolicy()

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
        cluster_ids = self.central_policy.schedule(scheduler_context, task)
        schedule_infos = self.cluster_policy.schedule(scheduler_context, task, cluster_ids)

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
