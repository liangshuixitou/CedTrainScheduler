import asyncio
from abc import ABC
from abc import abstractmethod

from cedtrainscheduler.runtime.types.task import TaskInst


class ExecutorBackend(ABC):
    @abstractmethod
    async def execute_task(
        self,
        task_name: str,
        task_inst: TaskInst,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: int,
        plan_runtime: int,
        data_transfer_time: float,
    ) -> asyncio.subprocess.Process:
        """Execute a task using the backend implementation.

        Args:
            task_name: Name of the task to execute
            task_inst: Task instance to execute
            world_size: Total number of processes
            inst_rank: Rank of this instance
            master_addr: Master node address
            master_port: Master node port
            plan_runtime: Planned runtime in seconds
            data_transfer_time: Data transfer time in seconds

        Returns:
            asyncio.subprocess.Process: The process running the task
        """
        pass

    @abstractmethod
    async def monitor_task(self, process: asyncio.subprocess.Process, task_inst: TaskInst) -> None:
        """Monitor a running task and handle its completion.

        Args:
            process: The process to monitor
            task_inst: The task instance being monitored
        """
        pass