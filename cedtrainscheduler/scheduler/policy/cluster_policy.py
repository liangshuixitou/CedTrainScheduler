from cedtrainscheduler.scheduler.types.cluster import CLUSTER_TYPE_GPU_MAP
from cedtrainscheduler.scheduler.types.task import ScheduleInfo
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record


class ClusterPolicy:
    def __init__(self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem):
        self.cluster_manager = cluster_manager
        self.task_record = task_record
        self.file_system = file_system

        self.task_record = self.task_record.task_record
        self.task_data_info = self.file_system.task_data_info
        self.gpu_task_queue = self.cluster_manager.gpu_task_queue
        self.clusters = self.cluster_manager.clusters

    def schedule(self, current_time: int, task: TaskMeta, cluster_ids: list[str]) -> ScheduleInfo:
        pass


class GreedyPolicy(ClusterPolicy):
    def __init__(self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem):
        super().__init__(cluster_manager, task_record, file_system)

    def schedule(self, current_time: int, task: TaskMeta, cluster_ids: list[str]) -> ScheduleInfo:
        cluster_gpus = []
        for cluster_id in cluster_ids:
            nodes = self.clusters[cluster_id].nodes
            for node in nodes:
                for gpu in node.gpus:
                    cluster_gpus.append(gpu.gpu_id)

        cluster_gpus.sort(key=lambda gpu_id: self.gpu_task_queue[gpu_id].queue_time(current_time, self.task_record))

        gpu_groups = {}
        for gpu_id in cluster_gpus:
            node_id = self.cluster_manager.gpu_node_map[gpu_id].node_id
            cluster_id = self.cluster_manager.node_cluster_map[node_id].cluster_id

            if cluster_id not in gpu_groups:
                gpu_groups[cluster_id] = []
            gpu_groups[cluster_id].append(gpu_id)

        min_execution_time = float("inf")
        selected_group = None

        for cluster_id, group in gpu_groups.items():
            selected_gpus = group[: task.task_inst_num]
            if len(selected_gpus) < task.task_inst_num:
                continue

            max_wait_time = max(
                self.gpu_task_queue[gpu_id].queue_time(current_time, self.task_record) for gpu_id in selected_gpus
            )
            gpu_type = CLUSTER_TYPE_GPU_MAP[self.clusters[cluster_id].cluster_type]
            execution_time = max_wait_time + task.task_runtime[gpu_type]

            if execution_time < min_execution_time:
                min_execution_time = execution_time
                selected_group = selected_gpus

        schedule_infos = {}
        for inst_id, gpu_id in enumerate(selected_group):
            schedule_infos[inst_id] = ScheduleInfo(
                inst_id=inst_id,
                gpu_id=gpu_id,
            )
        return schedule_infos
