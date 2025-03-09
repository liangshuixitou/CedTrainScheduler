import math

from cedtrainscheduler.scheduler.factory import SchedulerType
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.task import ScheduleInfo
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.executor import GPUExecutor
from cedtrainscheduler.simulator.fs import TaskDataInfo


class FCFSScheduler(SchedulerBase):
    def __init__(self):
        super().__init__(SchedulerType.FCFS)
        self.task_queue: list[TaskMeta] = []

    def schedule(
        self,
        current_time: float,
        clusters: dict[str, Cluster],
        gpu_task_queue: dict[str, GPUExecutor],
        task_data_info: dict[str, TaskDataInfo],
        task_record: dict[str, TaskWrapRuntimeInfo],
    ) -> tuple[TaskWrapRuntimeInfo, bool]:
        if not self.task_queue or len(self.task_queue) == 0:
            return None, True

        task = self.task_queue[0]

        # 寻找集群内的所有GPU
        cluster_gpus = []
        for cluster in clusters.values():
            nodes = cluster.nodes
            for node in nodes:
                for gpu in node.gpus:
                    cluster_gpus.append(gpu.gpu_id)

        # 按照队列总体运行时间排序，选择运行时间最短的节点
        cluster_gpus.sort(key=lambda gpu_id: gpu_task_queue[gpu_id].queue_time(current_time, task_record))

        # 将GPU按类型分组
        gpu_groups = {}
        for gpu_id in cluster_gpus:
            gpu_type = gpu_task_queue[gpu_id].gpu_type
            if gpu_type not in gpu_groups:
                gpu_groups[gpu_type] = []
            gpu_groups[gpu_type].append(gpu_id)

        # 选择执行时间最短的GPU组
        min_execution_time = float("inf")
        selected_group = None

        for gpu_type, group in gpu_groups.items():
            # 选择所需数量的GPU
            selected_gpus = group[: task.task_inst_num]
            if len(selected_gpus) < task.task_inst_num:
                continue  # 如果当前组的GPU数量不足，跳过

            # 计算当前组的任务执行时间
            max_wait_time = max(
                gpu_task_queue[gpu_id].queue_time(current_time, task_record) for gpu_id in selected_gpus
            )
            execution_time = max_wait_time + task.task_runtime[gpu_type]

            # 更新最短执行时间的组
            if execution_time < min_execution_time:
                min_execution_time = execution_time
                selected_group = selected_gpus

        # 给inst分配等待时间最短的节点
        schedule_infos = {}
        for inst_id, gpu_id in enumerate(selected_group):
            schedule_infos[inst_id] = ScheduleInfo(
                inst_id=inst_id,
                gpu_id=gpu_id,
            )

        # 创建任务运行时信息
        runtime_info = TaskWrapRuntimeInfo(
            task_meta=task,
            schedule_infos=schedule_infos,
            inst_status={},
            inst_data_status={},
            task_submit_time=current_time,
            task_start_time=-math.inf,
            task_end_time=-math.inf,
        )

        self.task_queue.pop(0)

        return runtime_info, len(self.task_queue) == 0
