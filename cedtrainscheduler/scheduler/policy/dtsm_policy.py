import math
from collections import defaultdict

from cedtrainscheduler.scheduler.policy.central_policy import CentralPolicy
from cedtrainscheduler.scheduler.policy.queue_policy import QueuePolicy
from cedtrainscheduler.scheduler.types.cluster import CLUSTER_TYPE_GPU_MAP
from cedtrainscheduler.scheduler.types.cluster import GPU_PERFORMANCE_MAP
from cedtrainscheduler.scheduler.types.cluster import GPUType
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskMeta

# global variable

comprehensive_affinity_dict: dict[str, dict[str, float]] = {}
cluster_queue_time_dict: dict[str, tuple[int, float]] = {}
cluster_data_arrival_time_dict: dict[str, dict[str, float]] = {}


class DTSMQueuePolicy(QueuePolicy):
    def __init__(self):
        super().__init__()

    def pop_one_task(self, scheduler_context: SchedulerContext) -> TaskMeta:
        self.set_scheduler_context(scheduler_context)

        comprehensive_affinity_dict.clear()
        task_priority_dict: dict[str, float] = {}
        cluster_queue_time_dict = self.get_all_queue_time()

        for task in self.task_queue:
            resource_affinity = self.get_resource_affinity(task, cluster_queue_time_dict)

            comprehensive_affinity: dict[str, float] = {}
            for cluster_id in self.clusters.keys():
                comprehensive_affinity[cluster_id] = resource_affinity[cluster_id]
            comprehensive_affinity_dict[task.task_id] = comprehensive_affinity

            max_cluster_id = max(comprehensive_affinity, key=comprehensive_affinity.get)
            comprehensive_affinity.pop(max_cluster_id)

            task_priority_dict[task.task_id] = (1 / task.task_runtime[GPUType.T4])

        max_task_id = max(task_priority_dict, key=task_priority_dict.get)
        for task in self.task_queue:
            if task.task_id == max_task_id:
                selected_task = task
                break

        self.task_queue.remove(selected_task)
        return selected_task

    def get_resource_affinity(
        self, task: TaskMeta, cluster_queue_time_dict: dict[str, tuple[int, float]]
    ) -> dict[str, float]:
        # return the resource affinity of the task to each cluster
        # 计算每个集群的资源使用情况,仅考虑GPU,总时间/GPU卡数
        cluster_resource_affinity: dict[str, float] = defaultdict(float)

        total_gpu_time = 0
        total_gpu_num = 0
        for _, (gpu_num, gpu_time) in cluster_queue_time_dict.items():
            total_gpu_time += gpu_time
            total_gpu_num += gpu_num
        all_avg_gpu_time = total_gpu_time / total_gpu_num

        # 计算所有集群的负载偏离度
        deviations = {}
        for cluster_id, (cluster_gpu_num, cluster_gpu_time) in cluster_queue_time_dict.items():
            cluster_gpu_type = CLUSTER_TYPE_GPU_MAP[self.clusters[cluster_id].cluster_type]
            deviations[cluster_id] = abs(
                (cluster_gpu_time + task.task_runtime[cluster_gpu_type]) / cluster_gpu_num - all_avg_gpu_time
            )

        # 找出最大偏离度用于归一化
        max_deviation = max(deviations.values()) if deviations else 1.0

        # 计算归一化后的资源亲和度
        for cluster_id in self.clusters.keys():
            if cluster_id in deviations:
                if max_deviation > 0:
                    # 归一化偏离度到0-1区间
                    normalized_deviation = deviations[cluster_id] / max_deviation
                    # 将偏离度转换为亲和度（反向关系）
                    cluster_resource_affinity[cluster_id] = 1 - normalized_deviation
                else:
                    # 如果所有偏离度为0，则亲和度为1
                    cluster_resource_affinity[cluster_id] = 1.0
            else:
                # 对于没有在队列中的集群，设置默认亲和度
                cluster_resource_affinity[cluster_id] = 1.0

        return cluster_resource_affinity

    def get_data_affinity(
        self, task: TaskMeta, cluster_data_arrival_time_dict: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        # return the data affinity of the task to each cluster
        avg_data_arrival_time = self.file_system.avg_data_arrival_time

        # 首先计算所有集群的原始数据亲和度
        raw_data_affinity = {}
        for cluster_id in self.clusters.keys():
            raw_data_affinity[cluster_id] = math.exp(
                cluster_data_arrival_time_dict[task.task_name][cluster_id] * -math.log(2) / avg_data_arrival_time
            )

        # 找出最大和最小亲和度值用于归一化
        max_affinity = max(raw_data_affinity.values()) if raw_data_affinity else 1.0
        min_affinity = min(raw_data_affinity.values()) if raw_data_affinity else 0.0

        # 计算归一化后的数据亲和度
        cluster_data_affinity: dict[str, float] = defaultdict(float)
        for cluster_id in self.clusters.keys():
            if max_affinity > min_affinity:
                # 标准归一化公式
                cluster_data_affinity[cluster_id] = (raw_data_affinity[cluster_id] - min_affinity) / (
                    max_affinity - min_affinity
                )
            else:
                # 如果所有值相同，则均设为1.0
                cluster_data_affinity[cluster_id] = 1.0

        return cluster_data_affinity

    def get_all_queue_time(self) -> dict[str, tuple[int, float]]:
        # return the gpu num and gpu queue time of the task to each cluster
        # dict[cluster_id, (gpu_num, gpu_time)]
        cluster_queue_time: dict[str, tuple[int, float]] = defaultdict(lambda: (0, 0))
        for cluster_id in self.clusters.keys():
            for node in self.clusters[cluster_id].nodes:
                for gpu in node.gpus:
                    gpu_time = self.gpu_task_queue[gpu.gpu_id].queue_time(self.current_time, self.task_record)
                    t4_time = gpu_time * GPU_PERFORMANCE_MAP[GPUType.T4] / GPU_PERFORMANCE_MAP[gpu.gpu_type]
                    cluster_queue_time[cluster_id] = (
                        cluster_queue_time[cluster_id][0] + 1,
                        cluster_queue_time[cluster_id][1] + t4_time,
                    )
        return cluster_queue_time

    def get_all_data_arrival_time(self) -> dict[str, dict[str, float]]:
        # return the data arrival time of the task to each cluster
        # dict[task_name, dict[cluster_id, data_arrival_time]]
        cluster_data_arrival_time: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        task_name_set = set()
        for task in self.task_queue:
            task_name_set.add(task.task_name)

        for task_name in task_name_set:
            task_data_info = self.file_system.get_task_data_info(task_name)
            for cluster_id in self.clusters.keys():
                cluster_data_arrival_time[task_name][cluster_id] = self.file_system.get_data_arrival_time(
                    task_data_info, cluster_id, self.cluster_manager
                )
        return cluster_data_arrival_time


class DTSMCentralPolicy(CentralPolicy):
    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta) -> str:
        self.set_scheduler_context(scheduler_context)

        # 获取任务对各集群的亲和度
        affinities = comprehensive_affinity_dict[task.task_id]

        # 按亲和度降序排序集群
        sorted_clusters = sorted(affinities.items(), key=lambda x: x[1], reverse=True)

        # 选择前1名集群（或者所有集群，如果集群数量少于5）
        top_cluster = sorted_clusters[0][0]

        return top_cluster
