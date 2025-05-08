import asyncio

from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.utils.logger import setup_logger
from cedtrainscheduler.runtime.utils.python_util import get_python_executable_path
from cedtrainscheduler.runtime.worker.backend.backend import Backend
from cedtrainscheduler.runtime.workload.script import ScriptGenerator


class PythonBackend(Backend):
    def __init__(self, gpu: GPU, executor_python_path: str):
        self.executor_python_path = executor_python_path
        self.logger = setup_logger(__name__)

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
        python_path = (
            self.executor_python_path
            if self.executor_python_path and self.executor_python_path != ""
            else get_python_executable_path()
        )

        script = ScriptGenerator.generate_python_script(
            gpu_rank=self.gpu.gpu_rank,
            task_id=task_inst.task_id,
            task_name=task_name,
            world_size=world_size,
            inst_rank=inst_rank,
            master_addr=master_addr,
            master_port=master_port,
            plan_runtime=plan_runtime,
            data_transfer_time=int(data_transfer_time),
            python_path=python_path,
        )

        self.logger.info(f"[GPU {self.gpu.gpu_id}] Start task {task_inst.task_id} with script: {script}")

        try:
            process = await asyncio.create_subprocess_shell(
                script,
                stdout=None,
                stderr=None,
            )
            self.logger.info(f"[GPU {self.gpu.gpu_id}] Task {task_inst.task_id} started with PID {process.pid}")
            return process
        except Exception as e:
            self.logger.error(f"[GPU {self.gpu.gpu_id}] Error starting task {task_inst.task_id}: {str(e)}")
            raise

    async def monitor_task(self, process: asyncio.subprocess.Process, task_inst: TaskInst) -> None:
        try:
            return_code = await process.wait()

            if return_code == 0:
                self.logger.info(f"[GPU {self.gpu.gpu_id}] Task {task_inst.task_id} completed successfully")
            else:
                self.logger.error(
                    f"[GPU {self.gpu.gpu_id}] Task {task_inst.task_id} failed with exit code {return_code}"
                )

        except Exception as e:
            self.logger.error(f"[GPU {self.gpu.gpu_id}] Error monitoring task {task_inst.task_id}: {str(e)}")
            raise
