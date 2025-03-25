import asyncio
import logging

from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskInstStatus
from cedtrainscheduler.runtime.utils.python_util import get_python_executable_path
from cedtrainscheduler.runtime.workload.script import ScriptGenerator


class Executor:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self.task_record: list[TaskInst] = []
        self.task_queue: list[TaskInst] = []
        self.task_record_lock = asyncio.Lock()

        self.logger = logging.getLogger(__name__)

    async def append_task(self, task: TaskInst):
        async with self.task_record_lock:
            self.task_record.append(task)
            self.task_queue.append(task)

    async def get_task_record(self) -> list[TaskInst]:
        async with self.task_record_lock:
            return self.task_record

    async def task_finished(self, current_task_inst: TaskInst):
        current_task_inst.inst_status = TaskInstStatus.Finished
        async with self.task_record_lock:
            self.task_queue.pop(0)
            self.task_queue[0].inst_status = TaskInstStatus.Ready

    async def start_task(
        self,
        task_name: str,
        task_inst: TaskInst,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: str,
    ):
        script = ScriptGenerator.generate_script(
            gpu_rank=self.gpu.gpu_rank,
            task_name=task_name,
            world_size=world_size,
            inst_rank=inst_rank,
            master_addr=master_addr,
            master_port=master_port,
            python_path=get_python_executable_path(),
        )


        with self.task_record_lock:
            current_task_inst = self.task_queue[0]

        if current_task_inst.inst_id != task_inst.inst_id or current_task_inst.inst_status != TaskInstStatus.Pending:
            self.logger.error(f"Task {task_name} instance {task_inst.inst_id} is not pending")
            return

        current_task_inst.inst_status = TaskInstStatus.Running

        try:
            # Start the process asynchronously
            process = await asyncio.create_subprocess_shell(
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Start a background task to monitor the process completion
            asyncio.create_task(self._monitor_task_completion(process, current_task_inst, task_name))

            self.logger.info(f"Task {task_name} started with PID {process.pid}")

        except Exception as e:
            self.logger.error(f"Error starting task {task_name}: {str(e)}")
            await self.task_finished(current_task_inst)

    async def _monitor_task_completion(self, process: asyncio.subprocess.Process, task_inst: TaskInst, task_name: str):
        """Monitor the task process and handle its completion."""
        try:
            stdout, stderr = await process.communicate()

            # Log the output
            if stdout:
                self.logger.info(f"Task {task_name} stdout: {stdout.decode()}")
            if stderr:
                self.logger.warning(f"Task {task_name} stderr: {stderr.decode()}")

            if process.returncode == 0:
                self.logger.info(f"Task {task_name} completed successfully")
            else:
                self.logger.error(f"Task {task_name} failed with exit code {process.returncode}")

            await self.task_finished(task_inst)

        except Exception as e:
            self.logger.error(f"Error monitoring task {task_name}: {str(e)}")
            await self.task_finished(task_inst)
