from cedtrainscheduler.scheduler.factory import SchedulerType
from cedtrainscheduler.scheduler.policy.central_policy import ResourceAffinityPolicy
from cedtrainscheduler.scheduler.policy.cluster_policy import FirstFitPolicy
from cedtrainscheduler.scheduler.policy.queue_policy import SFJQueuePolicy
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.scheduler.utils import build_task_wrap_runtime_info


class FirstFitScheduler(SchedulerBase):
    def __init__(self):
        super().__init__()
        self.scheduler_name = SchedulerType.FIRST_FIT
        self.queue_policy = SFJQueuePolicy()
        self.central_policy = ResourceAffinityPolicy()
        self.cluster_policy = FirstFitPolicy()

    def submit_task(self, scheduler_context: SchedulerContext, task: TaskMeta):
        self.queue_policy.add_task(scheduler_context, [task])

    def schedule(
        self,
        scheduler_context: SchedulerContext,
    ) -> tuple[TaskWrapRuntimeInfo, bool]:
        if len(self.task_queue) == 0:
            return None, True

        task = self.queue_policy.pop_one_task(scheduler_context)
        cluster_ids = self.central_policy.schedule(scheduler_context, task)
        schedule_infos = self.cluster_policy.schedule(scheduler_context, task, cluster_ids)

        # 创建任务运行时信息
        runtime_info = build_task_wrap_runtime_info(task, schedule_infos)

        return runtime_info, False
