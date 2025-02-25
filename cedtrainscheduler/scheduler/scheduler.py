import pandas as pd

from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskStatus
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.executor import GPUExecutor
from cedtrainscheduler.simulator.fs import TaskDataInfo


class SchedulerBase:
    def __init__(self, scheduler_name: str):
        self.scheduler_name = scheduler_name
        self.task_queue: list[TaskMeta] = []

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
                    "T4": float(row["runtime_T4"]),
                    "P100": float(row["runtime_P100"]),
                    "V100": float(row["runtime_V100"]),
                },
            )
            self.task_queue.append(task_meta)

        self.task_queue.sort(key=lambda x: x.task_id)

    def schedule(
        self,
        current_time: float,
        clusters: dict[str, Cluster],
        gpu_task_queue: dict[str, GPUExecutor],
        task_data_info: dict[str, TaskDataInfo],
        task_record: dict[str, TaskWrapRuntimeInfo],
    ) -> tuple[TaskWrapRuntimeInfo, bool]:
        pass
