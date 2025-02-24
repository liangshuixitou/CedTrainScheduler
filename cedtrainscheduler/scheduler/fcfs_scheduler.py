from cedtrainscheduler.scheduler.factory import SchedulerType
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.task import ScheduleInfo
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.executor import GPUExecutor
from cedtrainscheduler.simulator.fs import TaskDataInfo


class FCFScheduler(SchedulerBase):
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
        # 查找所有的数据所在的节点的集群
        nodes = task_data_info[task.task_id].dataset.storage_nodes + task_data_info[task.task_id].model.storage_nodes
        cluster_ids = []
        for cluster in clusters.values():
            for node in cluster.nodes:
                if node.ip_address in nodes:
                    cluster_ids.append(cluster.cluster_id)

        # 寻找集群内的所有GPU
        cluster_gpus = []
        for cluster_id in cluster_ids:
            nodes = clusters[cluster_id].nodes
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

        # 选择等待时间最短的GPU组
        selected_group = min(
            gpu_groups.values(),
            key=lambda group: sum(gpu_task_queue[gpu_id].queue_time(current_time, task_record) for gpu_id in group),
        )

        # 给inst分配等待时间最短的节点
        schedule_infos = {}
        for inst_id in range(task.inst_count):
            plan_gpu = task.task_plan_gpu
            # 动态选择合适的GPU
            selected_gpus = selected_group[:plan_gpu]
            schedule_infos[inst_id] = ScheduleInfo(
                gpu_list=selected_gpus,
            )
            # 移除已分配的GPU，避免重复分配
            selected_group = selected_group[plan_gpu:]

        # 创建任务运行时信息
        runtime_info = TaskWrapRuntimeInfo(
            task_meta=task,
            schedule_infos=schedule_infos,
        )

        self.task_queue.pop(0)

        return runtime_info, len(self.task_queue) == 0
