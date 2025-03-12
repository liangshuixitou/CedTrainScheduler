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

    def schedule(self, current_time: int, task: TaskMeta, cluster_id: str) -> ScheduleInfo:
        pass


class GreedyPolicy(ClusterPolicy):
    def __init__(self, cluster_manager: ClusterManager, task_record: Record, file_system: FileSystem):
        super().__init__(cluster_manager, task_record, file_system)

    def schedule(self, current_time: int, task: TaskMeta, cluster_id: str) -> ScheduleInfo:
        cluster_gpus = []
        nodes = self.clusters[cluster_id].nodes
        for node in nodes:
            for gpu in node.gpus:
                cluster_gpus.append(gpu.gpu_id)

        cluster_gpus.sort(key=lambda gpu_id: self.gpu_task_queue[gpu_id].queue_time(current_time, self.task_record))

        selected_gpus = cluster_gpus[: task.task_inst_num]

        schedule_infos = {}
        for inst_id, gpu_id in enumerate(selected_gpus):
            schedule_infos[inst_id] = ScheduleInfo(
                inst_id=inst_id,
                gpu_id=gpu_id,
            )
        return schedule_infos
