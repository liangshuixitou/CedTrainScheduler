import math
from collections import defaultdict

from cedtrainscheduler.scheduler.policy.central_policy import CentralPolicy
from cedtrainscheduler.scheduler.policy.cluster_policy import ClusterPolicy
from cedtrainscheduler.scheduler.policy.queue_policy import QueuePolicy
from cedtrainscheduler.scheduler.types.cluster import CLUSTER_TYPE_GPU_MAP
from cedtrainscheduler.scheduler.types.cluster import GPU_PERFORMANCE_MAP
from cedtrainscheduler.scheduler.types.cluster import GPUType
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import ScheduleInfo
from cedtrainscheduler.scheduler.types.task import TaskMeta

# global variable

comprehensive_affinity_dict: dict[str, dict[str, float]] = {}
cluster_queue_time_dict: dict[str, tuple[int, float]] = {}
cluster_data_arrival_time_dict: dict[str, dict[str, float]] = {}


class CedQueuePolicy(QueuePolicy):
    def __init__(self):
        super().__init__()
        self.weight_resource = 0.85
        self.weight_max_affinity = 0.10

    def pop_one_task(self, scheduler_context: SchedulerContext) -> TaskMeta:
        self.set_scheduler_context(scheduler_context)

        comprehensive_affinity_dict.clear()
        task_priority_dict: dict[str, float] = {}
        cluster_queue_time_dict = self.get_all_queue_time()
        cluster_data_arrival_time_dict = self.get_all_data_arrival_time()

        for task in self.task_queue:
            resource_affinity = self.get_resource_affinity(task, cluster_queue_time_dict)

            data_affinity = self.get_data_affinity(task, cluster_data_arrival_time_dict)

            comprehensive_affinity: dict[str, float] = {}
            for cluster_id in self.clusters.keys():
                comprehensive_affinity[cluster_id] = (
                    self.weight_resource * resource_affinity[cluster_id]
                    + (1 - self.weight_resource) * data_affinity[cluster_id]
                )
                # print(f"task_id: {task.task_id}, "
                #       f"cluster_id: {cluster_id}, "
                #       f"resource:{resource_affinity[cluster_id]}, "
                #       f"data:{data_affinity[cluster_id]}, "
                #       f"com:{comprehensive_affinity[cluster_id]}")
            comprehensive_affinity_dict[task.task_id] = comprehensive_affinity

            max_affinity = max(comprehensive_affinity.values())
            max_cluster_id = max(comprehensive_affinity, key=comprehensive_affinity.get)
            comprehensive_affinity.pop(max_cluster_id)
            second_max_affinity = max(comprehensive_affinity.values())
            comprehensive_affinity[max_cluster_id] = max_affinity

            task_priority_dict[task.task_id] = (1 / task.task_runtime[GPUType.T4]) * (
                self.weight_max_affinity * max_affinity + (1 - self.weight_max_affinity) * second_max_affinity
            )

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
            deviations[cluster_id] = ((cluster_gpu_time)
                                      / cluster_gpu_num - all_avg_gpu_time) / GPU_PERFORMANCE_MAP[cluster_gpu_type]
        # 找出最大偏离度用于归一化
        max_deviation = max(deviations.values())
        min_deviation = min(deviations.values())

        # 计算归一化后的资源亲和度
        for cluster_id in self.clusters.keys():
            if cluster_id in deviations:
                if max_deviation > 0:
                    # 归一化偏离度到0-1区间
                    normalized_deviation = (deviations[cluster_id] - min_deviation) / (max_deviation - min_deviation)
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
                    cluster_queue_time[cluster_id] = (
                        cluster_queue_time[cluster_id][0] + 1,
                        cluster_queue_time[cluster_id][1] + gpu_time,
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


class CedCentralPolicy(CentralPolicy):
    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta) -> str:
        self.set_scheduler_context(scheduler_context)

        # 获取任务对各集群的亲和度
        affinities = comprehensive_affinity_dict[task.task_id]
        # print(f"task id: {task.task_id}, affinity: {affinities}")

        # 按亲和度降序排序集群
        sorted_clusters = sorted(affinities.items(), key=lambda x: x[1], reverse=True)

        # print(f"sorted_clusters: {sorted_clusters}")
        # 选择前1名集群（或者所有集群，如果集群数量少于5）
        top_cluster = sorted_clusters[0][0]

        return top_cluster


class CedClusterPolicy(ClusterPolicy):
    def _is_large_task(self, task: TaskMeta) -> bool:
        return task.task_inst_num >= 2

    def _calculate_group_score(self, group: list[str], current_time: float, task: TaskMeta, cluster_type: str) -> float:
        """多维度评估GPU组得分（执行时间 + 负载均衡 + 数据亲和性）"""
        # 基础执行时间
        max_wait_time = max(self.gpu_task_queue[gpu_id].queue_time(current_time, self.task_record) for gpu_id in group)
        exec_time = max_wait_time + task.task_runtime[CLUSTER_TYPE_GPU_MAP[cluster_type]]

        # 负载均衡因子（避免节点过载）
        node_counts = defaultdict(int)
        for gpu_id in group:
            node_id = self.cluster_manager.gpu_node_map[gpu_id].node_id
            node_counts[node_id] += 1
        load_balance = max(node_counts.values()) / len(group)  # 越低越好

        # 综合评分（权重可调）
        return exec_time * 0.7 + load_balance * 0.2

    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta, cluster_id: str) -> dict[int, ScheduleInfo]:
        self.set_scheduler_context(scheduler_context)

        current_time = scheduler_context.current_time

        is_large_task = self._is_large_task(task)
        best_group = None
        min_score = float("inf")
        # 第一优先级：大任务优先同节点分配
        if is_large_task:
            cluster = self.clusters[cluster_id]
            for node in cluster.nodes:
                available_gpus = [gpu.gpu_id for gpu in node.gpus]
                if len(available_gpus) < task.task_inst_num:
                    continue

                # 按等待时间排序并取前N个
                available_gpus.sort(key=lambda gid: self.gpu_task_queue[gid].queue_time(current_time, self.task_record))
                candidate_group = available_gpus[: task.task_inst_num]

                # 计算综合评分
                score = self._calculate_group_score(candidate_group, current_time, task, cluster.cluster_type)
                if score < min_score:
                    min_score = score
                    best_group = candidate_group

        # 第二优先级：跨节点分配（含小任务逻辑）
        if best_group is None:
            cluster = self.clusters[cluster_id]
            all_gpus = []
            for node in cluster.nodes:
                all_gpus.extend(gpu.gpu_id for gpu in node.gpus)

            # 优化排序：同时考虑等待时间和节点分布
            all_gpus.sort(
                key=lambda gid: (
                    self.gpu_task_queue[gid].queue_time(current_time, self.task_record),
                    self.cluster_manager.gpu_node_map[gid].node_id,
                )
            )

            # 生成候选组（尽量分散到不同节点）
            candidate_group = []
            node_used = defaultdict(int)
            for gpu_id in all_gpus:
                node_id = self.cluster_manager.gpu_node_map[gpu_id].node_id
                if node_used[node_id] < 2:  # 每个节点最多取2个GPU
                    candidate_group.append(gpu_id)
                    node_used[node_id] += 1
                    if len(candidate_group) == task.task_inst_num:
                        break

            if len(candidate_group) == task.task_inst_num:
                score = self._calculate_group_score(candidate_group, current_time, task, cluster.cluster_type)
                if score < min_score:
                    min_score = score
                    best_group = candidate_group

        # 第三优先级：纯等待时间排序
        if best_group is None:
            all_gpus = []
            cluster = self.clusters[cluster_id]
            for node in cluster.nodes:
                all_gpus.extend(gpu.gpu_id for gpu in node.gpus)

            all_gpus.sort(key=lambda gid: self.gpu_task_queue[gid].queue_time(current_time, self.task_record))
            best_group = all_gpus[: task.task_inst_num]

        # 构造返回结果
        return {inst_id: ScheduleInfo(inst_id=inst_id, gpu_id=gpu_id) for inst_id, gpu_id in enumerate(best_group)}
