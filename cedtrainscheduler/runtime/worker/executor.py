import asyncio

from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.task import TaskInst


class Executor:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self.task_record: list[TaskInst] = []
        self.task_record_lock = asyncio.Lock()

    async def append_task(self, task: TaskInst):
        async with self.task_record_lock:
            self.task_record.append(task)

    async def get_task_record(self) -> list[TaskInst]:
        async with self.task_record_lock:
            return self.task_record

    