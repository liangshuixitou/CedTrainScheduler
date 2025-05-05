from abc import ABC
from abc import abstractmethod

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.types.cluster import Cluster
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskMeta
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo


class ManagerService(ABC):
    @abstractmethod
    async def handle_task_submit(self, task_meta: TaskMeta):
        pass

    @abstractmethod
    async def handle_task_list(self) -> list[TaskWrapRuntimeInfo]:
        pass

    @abstractmethod
    async def handle_master_register(
        self,
        cluster: Cluster,
        task_infos: dict[str, TaskWrapRuntimeInfo],
        master_info: ComponentInfo,
        task_queue_map: dict[str, TaskInst],
    ):
        pass

    @abstractmethod
    async def handle_metrics(self) -> dict:
        pass
