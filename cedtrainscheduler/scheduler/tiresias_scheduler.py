from cedtrainscheduler.scheduler.factory import SchedulerType
from cedtrainscheduler.scheduler.policy.cluster_policy import GreedyPolicy
from cedtrainscheduler.scheduler.policy.thiresias_policy import TiresiasClusterPolicy
from cedtrainscheduler.scheduler.policy.thiresias_policy import TiresiasQueuePolicy
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.scheduler.utils import build_task_wrap_runtime_info


class TiresiasScheduler(SchedulerBase):
    def __init__(self):
        super().__init__()
        self.scheduler_name = SchedulerType.TIRESIAS
        self.queue_policy = TiresiasQueuePolicy()
        self.central_policy = TiresiasClusterPolicy()
        self.cluster_policy = GreedyPolicy()  # 使用默认的贪心策略进行具体分配

    def submit_task(self, scheduler_context: SchedulerContext, task: TaskMeta):
        self.queue_policy.add_task(scheduler_context, [task])

    def schedule(self, scheduler_context: SchedulerContext) -> tuple[TaskWrapRuntimeInfo, bool]:
        if len(self.task_queue) == 0:
            return None, True

        task = self.queue_policy.pop_one_task(scheduler_context)

        cluster_id = self.central_policy.schedule(scheduler_context, task)
        schedule_infos = self.cluster_policy.schedule(scheduler_context, task, cluster_id)

        runtime_info = build_task_wrap_runtime_info(task, schedule_infos)

        return runtime_info, False
