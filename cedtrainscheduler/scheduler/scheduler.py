from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo


class SchedulerBase:
    def __init__(
        self,
    ):
        self.max_task_num = 50
        self.task_queue: list[TaskMeta] = []  # submitted task

    def submit_task(self, task: TaskMeta):
        pass

    def schedule(
        self,
        scheduler_context: SchedulerContext,
    ) -> tuple[TaskWrapRuntimeInfo, bool]:
        pass

    def is_queue_full(self) -> bool:
        return len(self.task_queue) >= self.max_task_num
