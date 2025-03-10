import math

from cedtrainscheduler.scheduler.factory import SchedulerType
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.cluster import CLUSTER_TYPE_GPU_MAP
from cedtrainscheduler.scheduler.types.cluster import GPUType
from cedtrainscheduler.scheduler.types.task import ScheduleInfo
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record


class SJFDataScheduler(SchedulerBase):
    def __init__(self, config_path: str, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem):
        super().__init__(SchedulerType.SJF_DATA, config_path, cluster_manager, task_record, file_system)

    def sort_task_queue(self):
        self.task_queue.sort(key=lambda x: x.task_runtime[GPUType.T4])

    def schedule(
        self,
        current_time: float,
    ) -> tuple[TaskWrapRuntimeInfo, bool]:
        if not self.task_queue or len(self.task_queue) == 0:
            return None, True

        task = self.task_queue[0]
        # 查找所有的数据所在的节点的集群
        nodes = self.task_data_info[task.task_name].dataset.storage_nodes
        cluster_ids: set[str] = set()
        for cluster in self.clusters.values():
            for node in cluster.nodes:
                if node.node_id in nodes:
                    cluster_ids.add(cluster.cluster_id)

        # 寻找集群内的所有GPU
        cluster_gpus = []
        for cluster_id in cluster_ids:
            nodes = self.clusters[cluster_id].nodes
            for node in nodes:
                for gpu in node.gpus:
                    cluster_gpus.append(gpu.gpu_id)

        # 按照队列总体运行时间排序，选择运行时间最短的节点
        cluster_gpus.sort(key=lambda gpu_id: self.gpu_task_queue[gpu_id].queue_time(current_time, self.task_record))

        # 将GPU按照集群分组
        gpu_groups = {}
        for gpu_id in cluster_gpus:
            node_id = self.cluster_manager.gpu_node_map[gpu_id].node_id
            cluster_id = self.cluster_manager.node_cluster_map[node_id].cluster_id

            if cluster_id not in gpu_groups:
                gpu_groups[cluster_id] = []
            gpu_groups[cluster_id].append(gpu_id)

        # 选择执行时间最短的GPU组
        min_execution_time = float("inf")
        selected_group = None

        for cluster_id, group in gpu_groups.items():
            # 选择所需数量的GPU
            selected_gpus = group[: task.task_inst_num]
            if len(selected_gpus) < task.task_inst_num:
                continue  # 如果当前组的GPU数量不足，跳过

            # 计算当前组的任务执行时间
            max_wait_time = max(
                self.gpu_task_queue[gpu_id].queue_time(current_time, self.task_record) for gpu_id in selected_gpus
            )
            gpu_type = CLUSTER_TYPE_GPU_MAP[self.clusters[cluster_id].cluster_type]
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

        return runtime_info, False
