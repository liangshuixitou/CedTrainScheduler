import json

from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.cluster import GPU
from cedtrainscheduler.scheduler.types.cluster import Node
from cedtrainscheduler.simulator.executor import GPUExecutor


class ClusterManager:
    def __init__(self, config_path: str = None):
        self.clusters: dict[str, Cluster] = {}
        self.node_cluster_map: dict[str, Cluster] = {}
        self.gpu_node_map: dict[str, Node] = {}
        self.gpu_task_queue: dict[str, GPUExecutor] = {}

        if config_path:
            self.init_cluster(config_path)
            self.init_node_cluster_map()
            self.init_gpu_node_map()
            self.init_gpu_task_queues()

    @classmethod
    def from_clusters(cls, clusters: dict[str, Cluster]) -> "ClusterManager":
        cluster_manager = cls()
        cluster_manager.clusters = clusters
        cluster_manager.init_node_cluster_map()
        cluster_manager.init_gpu_node_map()
        cluster_manager.init_gpu_task_queues()
        return cluster_manager

    def init_gpu_task_queues(self):
        for cluster in self.clusters.values():
            for node in cluster.nodes:
                for gpu in node.gpus:
                    self.gpu_task_queue[gpu.gpu_id] = GPUExecutor(gpu.gpu_id, gpu.gpu_type)

    def init_node_cluster_map(self):
        for cluster in self.clusters.values():
            for node in cluster.nodes:
                self.node_cluster_map[node.node_id] = cluster

    def init_gpu_node_map(self):
        for cluster in self.clusters.values():
            for node in cluster.nodes:
                for gpu in node.gpus:
                    self.gpu_node_map[gpu.gpu_id] = node

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
                    for gpu_info in node_info["gpus"]:
                        gpus.append(GPU(gpu_id=gpu_info["gpu_id"], gpu_type=gpu_info["gpu_type"]))
                    nodes.append(
                        Node(
                            node_id=node_info["node_id"],
                            cpu_cores=node_info["cpu_cores"],
                            memory=node_info["memory"],
                            ip_address=node_info["ip_address"],
                            gpus=gpus,
                        )
                    )

                # 创建 Cluster 对象
                cluster = Cluster(
                    cluster_id=cluster_info["cluster_id"],
                    cluster_name=cluster_info["cluster_name"],
                    cluster_type=cluster_info["cluster_type"],
                    nodes=nodes,
                    intra_domain_bandwidth=cluster_info["intra_domain_bandwidth"],
                    inter_domain_bandwidth=cluster_info["inter_domain_bandwidth"],
                )
                # 将 Cluster 对象添加到字典中
                self.clusters[cluster.cluster_id] = cluster

    def get_bandwidth(self, node_id: str, target_cluster_id: str) -> float:
        """获取节点之间的带宽"""
        # 获取节点所在的集群
        node_cluster = self.node_cluster_map[node_id]
        target_cluster = self.clusters[target_cluster_id]

        # 如果节点在同一个集群内，则直接返回集群内带宽
        if node_cluster == target_cluster:
            return node_cluster.intra_domain_bandwidth

        # 如果节点在不同集群内，则返回最小的域间带宽
        return min(node_cluster.inter_domain_bandwidth, target_cluster.inter_domain_bandwidth)
