from abc import ABC
from abc import abstractmethod

from cedtrainscheduler.runtime.types.cluster import Node
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo


class MasterService(ABC):
    @abstractmethod
    async def handle_worker_register(
        self, node: Node, task_insts: list[TaskInst], task_queue_map: dict[str, list[TaskInst]]
    ):
        pass

    @abstractmethod
    async def handle_task_submit(self, task_info: TaskWrapRuntimeInfo):
        pass
