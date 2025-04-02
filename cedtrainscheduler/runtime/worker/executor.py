import asyncio

from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskInstStatus
from cedtrainscheduler.runtime.utils.logger import setup_logger
from cedtrainscheduler.runtime.utils.python_util import get_python_executable_path
from cedtrainscheduler.runtime.workload.script import ScriptGenerator

LOG_DIR = "/tmp/train_logs"


class Executor:
    def __init__(self, gpu: GPU):
        self.gpu = gpu
        self.task_record: list[TaskInst] = []
        self.task_queue: list[TaskInst] = []
        self.task_record_lock = asyncio.Lock()

        self.logger = setup_logger(__name__)

    async def append_task(self, task: TaskInst):
        async with self.task_record_lock:
            if len(self.task_queue) == 0:
                task.inst_status = TaskInstStatus.Ready
            self.task_record.append(task)
            self.task_queue.append(task)

    async def get_task_record(self) -> list[TaskInst]:
        async with self.task_record_lock:
            return self.task_record

    async def task_finished(self, current_task_inst: TaskInst):
        current_task_inst.inst_status = TaskInstStatus.Finished
        async with self.task_record_lock:
            self.task_queue.pop(0)
            if len(self.task_queue) > 0:
                self.task_queue[0].inst_status = TaskInstStatus.Ready

    async def simulate_data_transfer(self, task_inst: TaskInst):
        pass

    async def start_task(
        self,
        task_name: str,
        task_inst: TaskInst,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: int,
        plan_runtime: int,
        data_transfer_time: float,
    ):
        script = ScriptGenerator.generate_script(
            gpu_rank=self.gpu.gpu_rank,
            task_id=task_inst.task_id,
            task_name=task_name,
            world_size=world_size,
            inst_rank=inst_rank,
            master_addr=master_addr,
            master_port=master_port,
            plan_runtime=plan_runtime,
            data_transfer_time=int(data_transfer_time),
            python_path=get_python_executable_path(),
        )

        self.logger.info(f"Start task {task_name} with script: {script}")

        async with self.task_record_lock:
            current_task_inst = self.task_queue[0]

        if current_task_inst.inst_id != task_inst.inst_id or current_task_inst.inst_status != TaskInstStatus.Ready:
            self.logger.error(f"Task {task_name} instance {task_inst.inst_id} is not ready")
            return

        current_task_inst.inst_status = TaskInstStatus.Running

        try:
            # 不再捕获输出，因为已经重定向到文件
            process = await asyncio.create_subprocess_shell(
                script,
                stdout=None,  # 使用 None 而不是 PIPE
                stderr=None,
            )

            # 简化监控任务
            asyncio.create_task(self._monitor_task_completion(process, current_task_inst, task_name))
            self.logger.info(f"Task {task_name} started with PID {process.pid}")

        except Exception as e:
            self.logger.error(f"Error starting task {task_name}: {str(e)}")
            await self.task_finished(current_task_inst)

    async def _monitor_task_completion(self, process: asyncio.subprocess.Process, task_inst: TaskInst, task_name: str):
        """只监控进程退出状态"""
        try:
            # 直接等待进程结束
            return_code = await process.wait()

            if return_code == 0:
                self.logger.info(f"Task {task_name} completed successfully")
            else:
                self.logger.error(f"Task {task_name} failed with exit code {return_code}")

            await self.task_finished(task_inst)

        except Exception as e:
            self.logger.error(f"Error monitoring task {task_name}: {str(e)}")
            await self.task_finished(task_inst)
