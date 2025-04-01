import networkx as nx

from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import ScheduleInfo
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.executor import GPUExecutor
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.fs import TaskDataInfo
from cedtrainscheduler.simulator.manager import ClusterManager


class ClusterPolicy:
    def __init__(self):
        self.cluster_manager: ClusterManager = None
        self.task_record: dict[str, TaskWrapRuntimeInfo] = {}
        self.file_system: FileSystem = None
        self.task_queue: list[TaskMeta] = []

        self.task_data_info: dict[str, TaskDataInfo] = {}
        self.gpu_task_queue: dict[str, GPUExecutor] = {}
        self.clusters: dict[str, Cluster] = {}

    def set_scheduler_context(self, scheduler_context: SchedulerContext):
        self.cluster_manager = scheduler_context.cluster_manager
        self.file_system = scheduler_context.file_system
        self.task_record = scheduler_context.task_record
        self.task_queue = scheduler_context.task_queue

        self.task_data_info = self.file_system.task_data_info
        self.gpu_task_queue = self.cluster_manager.gpu_task_queue
        self.clusters = self.cluster_manager.clusters

    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta, cluster_id: str) -> dict[int, ScheduleInfo]:
        pass


class FirstFitPolicy(ClusterPolicy):
    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta, cluster_id: str) -> dict[int, ScheduleInfo]:
        self.set_scheduler_context(scheduler_context)

        cluster_gpus = []
        nodes = self.clusters[cluster_id].nodes
        for node in nodes:
            for gpu in node.gpus:
                cluster_gpus.append(gpu.gpu_id)

        selected_gpus = cluster_gpus[: task.task_inst_num]

        schedule_infos = {}
        for inst_id, gpu_id in enumerate(selected_gpus):
            schedule_infos[inst_id] = ScheduleInfo(
                inst_id=inst_id,
                gpu_id=gpu_id,
            )
        return schedule_infos


class WorstFitPolicy(ClusterPolicy):
    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta, cluster_id: str) -> dict[int, ScheduleInfo]:
        self.set_scheduler_context(scheduler_context)
        current_time = scheduler_context.current_time

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


class MinCostMatchingPolicy(ClusterPolicy):
    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta, cluster_id: str) -> dict[int, ScheduleInfo]:
        self.set_scheduler_context(scheduler_context)
        current_time = scheduler_context.current_time

        # 构建二分图
        G = nx.Graph()

        # 添加任务实例节点 (左侧)
        task_nodes = [f"task_{i}" for i in range(task.task_inst_num)]
        G.add_nodes_from(task_nodes, bipartite=0)

        # 添加GPU节点 (右侧)
        gpu_nodes = []
        for node in self.clusters[cluster_id].nodes:
            for gpu in node.gpus:
                gpu_nodes.append(gpu.gpu_id)
        G.add_nodes_from(gpu_nodes, bipartite=1)

        # 添加边和权重 (使用GPU的队列时间作为权重)
        for _, task_node in enumerate(task_nodes):
            for gpu_id in gpu_nodes:
                weight = self.gpu_task_queue[gpu_id].queue_time(current_time, self.task_record)
                G.add_edge(task_node, gpu_id, weight=weight)

        # 使用最小权二分匹配算法求解
        matching = nx.bipartite.minimum_weight_full_matching(G, top_nodes=task_nodes, weight="weight")

        schedule_infos = {}
        for task_node, gpu_id in matching.items():
            if task_node.startswith("task_"):
                inst_id = int(task_node.split("_")[1])
                schedule_infos[inst_id] = ScheduleInfo(
                    inst_id=inst_id,
                    gpu_id=gpu_id,
                )

        return schedule_infos


class ChronusPolicy(ClusterPolicy):
    def schedule(self, scheduler_context: SchedulerContext, task: TaskMeta, cluster_id: str) -> dict[int, ScheduleInfo]:
        self.set_scheduler_context(scheduler_context)
        current_time = scheduler_context.current_time

        # 获取所有节点的GPU并计算节点的平均队列时间
        node_gpus = {}  # 存储节点及其GPU
        node_queue_times = {}  # 存储节点的平均队列时间

        for node in self.clusters[cluster_id].nodes:
            node_gpus[node.node_id] = []
            total_queue_time = 0

            for gpu in node.gpus:
                node_gpus[node.node_id].append(gpu.gpu_id)
                total_queue_time += self.gpu_task_queue[gpu.gpu_id].queue_time(current_time, self.task_record)

            # 计算该节点的平均队列时间
            node_queue_times[node.node_id] = total_queue_time / len(node.gpus)

        # 按照平均队列时间排序节点（空闲程度从高到低）
        sorted_nodes = sorted(node_queue_times.keys(), key=lambda node_id: node_queue_times[node_id])

        # 分配任务实例到GPU
        schedule_infos = {}
        assigned_count = 0

        # 遍历排序后的节点进行分配
        for node_id in sorted_nodes:
            # 如果已经分配完所有实例，退出循环
            if assigned_count >= task.task_inst_num:
                break

            # 获取当前节点的GPU列表
            gpus = node_gpus[node_id]

            # 对节点内的GPU按队列时间排序
            sorted_gpus = sorted(
                gpus, key=lambda gpu_id: self.gpu_task_queue[gpu_id].queue_time(current_time, self.task_record)
            )

            # 分配GPU给任务实例
            for gpu_id in sorted_gpus:
                if assigned_count >= task.task_inst_num:
                    break

                schedule_infos[assigned_count] = ScheduleInfo(inst_id=assigned_count, gpu_id=gpu_id)
                assigned_count += 1

        return schedule_infos
