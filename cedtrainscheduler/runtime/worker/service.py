from abc import ABC
from abc import abstractmethod

from cedtrainscheduler.runtime.types.task import TaskInst


class WorkerService(ABC):
    @abstractmethod
    async def handle_task_inst_submit(self, task_inst: TaskInst, gpu_id: str):
        pass

    @abstractmethod
    async def handle_task_inst_start(
        self,
        task_inst: TaskInst,
        gpu_id: str,
        task_name: str,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: int,
        plan_runtime: int,
        data_transfer_time: float,
    ):
        pass

    @abstractmethod
    async def handle_task_log(self, task_id: str, inst_id: int, gpu_id: str) -> str:
        pass
