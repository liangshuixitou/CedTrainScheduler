import asyncio
import logging
import os
from typing import Optional

from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo


class Executor:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self.task_queue: list[TaskInst] = []
        self.task_queue_lock = asyncio.Lock()

        self.current_task: Optional[TaskWrapRuntimeInfo] = None
        self.current_process: Optional[asyncio.subprocess.Process] = None

        self.logger = logging.getLogger(__name__)
        self._process_task = None
        self._monitor_task = None

    async def append_task(self, task: TaskInst):
        async with self.task_queue_lock:
            self.task_queue.append(task)

    async def get_task_record(self) -> list[TaskInst]:
        async with self.task_queue_lock:
            return self.task_queue

    async def start_task(self, task_name: str, task_inst: TaskInst):
        pass
