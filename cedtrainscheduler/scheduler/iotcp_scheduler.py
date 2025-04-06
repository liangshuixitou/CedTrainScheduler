from cedtrainscheduler.scheduler.factory import SchedulerType
from cedtrainscheduler.scheduler.policy.ced_policy import CedCentralPolicy
from cedtrainscheduler.scheduler.policy.ced_policy import CedClusterPolicy
from cedtrainscheduler.scheduler.policy.ced_policy import CedQueuePolicy
from cedtrainscheduler.scheduler.policy.central_policy import ResourceAffinityPolicy
from cedtrainscheduler.scheduler.policy.queue_policy import SFJQueuePolicy
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.scheduler.utils import build_task_wrap_runtime_info


class IOTCPScheduler(SchedulerBase):
    def __init__(self):
        super().__init__()
        self.scheduler_name = SchedulerType.IOTCP
        self.queue_policy = SFJQueuePolicy()
        self.central_policy = ResourceAffinityPolicy()
        self.cluster_policy = CedClusterPolicy()

    def submit_task(self, scheduler_context: SchedulerContext, task: TaskMeta):
        self.queue_policy.add_task(scheduler_context, [task])

    def schedule(
        self,
        scheduler_context: SchedulerContext,
    ) -> tuple[TaskWrapRuntimeInfo, bool]:
        if len(self.task_queue) == 0:
            return None, True

        task = self.queue_policy.pop_one_task(scheduler_context)
        cluster_id = self.central_policy.schedule(scheduler_context, task)
        schedule_infos = self.cluster_policy.schedule(scheduler_context, task, cluster_id)

        # 创建任务运行时信息
        runtime_info = build_task_wrap_runtime_info(task, schedule_infos)

        return runtime_info, False
