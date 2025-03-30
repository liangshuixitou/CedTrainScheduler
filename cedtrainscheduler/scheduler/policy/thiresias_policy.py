from collections import defaultdict

from cedtrainscheduler.scheduler.policy.central_policy import CentralPolicy
from cedtrainscheduler.scheduler.policy.queue_policy import QueuePolicy
from cedtrainscheduler.scheduler.types.cluster import GPUType
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskMeta


class TiresiasQueuePolicy(QueuePolicy):
    def __init__(self):
        super().__init__()
        self.is_sorted = False

    def add_task(self, scheduler_context: SchedulerContext, task_list: list[TaskMeta]):
        super().add_task(scheduler_context, task_list)
        self.is_sorted = False

    def _sort_task_queue(self, scheduler_context: SchedulerContext):
        self.task_queue.sort(
            key=lambda x: 1
            / (scheduler_context.current_time - x.task_start_time)
            * x.task_inst_num
            / x.task_runtime[GPUType.T4]
        )
        self.is_sorted = True

    def pop_one_task(self, scheduler_context: SchedulerContext) -> TaskMeta:
        self.set_scheduler_context(scheduler_context)
        if not self.is_sorted:
            self._sort_task_queue(scheduler_context)
        return self.task_queue.pop(0)


class TiresiasClusterPolicy(CentralPolicy):
    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta) -> str:
        self.set_scheduler_context(scheduler_context)
        current_time = scheduler_context.current_time

        cluster_queue_time: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0))
        for cluster_id in self.clusters.keys():
            for node in self.clusters[cluster_id].nodes:
                for gpu in node.gpus:
                    gpu_time = self.gpu_task_queue[gpu.gpu_id].queue_time(current_time, self.task_record)
                    cluster_queue_time[cluster_id] = (
                        cluster_queue_time[cluster_id][0] + 1,
                        cluster_queue_time[cluster_id][1] + gpu_time,
                    )
        avg_queue_times: dict[str, float] = {}

        for cluster_id, (gpu_count, total_queue_time) in cluster_queue_time.items():
            avg_queue_times[cluster_id] = total_queue_time / gpu_count

        # 按平均队列等待时间排序
        sorted_clusters = sorted(avg_queue_times.items(), key=lambda x: x[1])

        # 选择平均队列等待时间最少的前1个集群
        selected_cluster = sorted_clusters[0][0]
        return selected_cluster
