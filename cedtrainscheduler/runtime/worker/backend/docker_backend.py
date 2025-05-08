import asyncio

from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.utils.logger import setup_logger
from cedtrainscheduler.runtime.worker.backend.backend import ExecutorBackend
from cedtrainscheduler.runtime.workload.script import ScriptGenerator


class DockerBackend(ExecutorBackend):
    def __init__(self, gpu: GPU):
        self.gpu = gpu
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
        # Generate docker command using ScriptGenerator
        docker_cmd = ScriptGenerator.generate_ix_docker_script(
            gpu_rank=self.gpu.gpu_rank,
            task_id=task_inst.task_id,
            task_name=task_name,
            world_size=world_size,
            inst_rank=inst_rank,
            master_addr=master_addr,
            master_port=master_port,
            plan_runtime=plan_runtime,
            data_transfer_time=int(data_transfer_time),
        )
        self.logger.info(f"[GPU {self.gpu.gpu_id}] Start task {task_inst.task_id} with script: {docker_cmd}")
        try:
            process = await asyncio.create_subprocess_shell(
                docker_cmd,
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
