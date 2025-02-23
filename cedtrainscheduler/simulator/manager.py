import json
import uuid

from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.cluster import GPU
from cedtrainscheduler.scheduler.types.cluster import Node
from cedtrainscheduler.simulator.executor import GPUExecutor


class ClusterManager:
    def __init__(self, config_path: str):
        self.clusters: dict[str, Cluster] = {}
        self.init_cluster(config_path)

        self.gpu_node_map: dict[str, Node] = self.init_gpu_node_map()
        self.gpu_task_queue: dict[str, GPUExecutor] = self.init_gpu_task_queue()

    def init_gpu_queues(self):
        gpu_queues = {}
        for cluster in self.clusters.values():
            for node in cluster.nodes:
                for gpu in node.gpus:
                    gpu_queues[gpu.gpu_id] = GPUExecutor(gpu.gpu_id)
        return gpu_queues

    def init_gpu_node_map(self):
        gpu_node_map = {}
        for cluster in self.clusters.values():
            for node in cluster.nodes:
                for gpu in node.gpus:
                    gpu_node_map[gpu.gpu_id] = node
        return gpu_node_map

    def init_cluster(self, config_path: str):
        # 读取 JSON 配置文件
        with open(config_path) as file:
            config_data = json.load(file)

        # 遍历每个集群类型
        for _cluster_type, clusters in config_data["clusters"].items():
            for cluster_info in clusters:
                # 初始化节点和GPU信息
                nodes: list[Node] = []
                for node_info in cluster_info["nodes"]:
                    gpus: list[GPU] = []
                    for _ in range(node_info["gpu_count"]):
                        # 为每个GPU生成唯一ID
                        gpu_id = str(uuid.uuid4())
                        gpus.append(GPU(gpu_id=gpu_id, gpu_type=node_info["gpu_model"]))
                    nodes.append(node_info)

                # 创建 Cluster 对象
                cluster = Cluster(
                    cluster_id=cluster_info["cluster_id"],
                    cluster_name=cluster_info["cluster_name"],
                    cluster_type=cluster_info["cluster_type"],
                    nodes=nodes,
                )
                # 将 Cluster 对象添加到字典中
                self.clusters[cluster.cluster_id] = cluster
