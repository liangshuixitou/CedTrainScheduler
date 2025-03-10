import pandas as pd

from cedtrainscheduler.scheduler.types.cluster import GPUType
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskStatus
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record


class SchedulerBase:
    def __init__(
        self,
        scheduler_name: str,
        config_path: str,
        cluster_manager: ClusterManager,
        task_record: Record,
        file_system: FileSystem,
    ):
        self.scheduler_name = scheduler_name
        self.task_queue: list[TaskMeta] = []
        self.cluster_manager = cluster_manager
        self.task_record = task_record
        self.file_system = file_system

        self.load_config(config_path)

        self.task_record = self.task_record.task_record
        self.task_data_info = self.file_system.task_data_info
        self.gpu_task_queue = self.cluster_manager.gpu_task_queue
        self.clusters = self.cluster_manager.clusters

    def load_config(self, config_path: str):
        df = pd.read_csv(config_path)
        for _, row in df.iterrows():
            task_meta = TaskMeta(
                task_id=row["job_name"],
                task_name=row["task_name"],
                task_inst_num=int(row["inst_num"]),
                task_plan_cpu=float(row["plan_cpu"]),
                task_plan_mem=float(row["plan_mem"]),
                task_plan_gpu=float(row["plan_gpu"]) / 100,
                task_status=TaskStatus.Pending,
                # 创建运行时间字典
                task_runtime={
                    GPUType.T4: float(row["runtime_T4"]),
                    GPUType.P100: float(row["runtime_P100"]),
                    GPUType.V100: float(row["runtime_V100"]),
                },
            )
            self.task_queue.append(task_meta)

        self.sort_task_queue()

    def sort_task_queue(self):
        import random

        random.shuffle(self.task_queue)

    def schedule(self, current_time: float) -> tuple[TaskWrapRuntimeInfo, bool]:
        pass
