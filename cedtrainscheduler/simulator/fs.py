import json
from dataclasses import dataclass
from typing import Optional

from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.manager import ClusterManager


@dataclass
class DatasetInfo:
    """数据集信息"""

    size_mb: int  # 数据集大小(MB)
    storage_nodes: list[str]  # 存储节点id列表


@dataclass
class ModelInfo:
    """模型信息"""

    size_mb: int  # 模型大小(MB)
    storage_nodes: list[str]  # 存储节点id列表


@dataclass
class TaskDataInfo:
    task_name: str  # 任务名称
    dataset: DatasetInfo  # 数据集信息
    model: ModelInfo  # 模型信息


class FileSystem:
    """文件系统模拟器"""

    def __init__(self, config_path: str, cluster_manager: ClusterManager):
        self.task_data_info: dict[str, TaskDataInfo] = {}  # 任务名称 -> 任务信息
        self.init_task_data(config_path)
        self.avg_data_arrival_time = self.get_avg_data_arrival_time(cluster_manager)

    def init_task_data(self, config_path: str) -> None:
        """从配置文件加载训练任务信息"""
        with open(config_path) as f:
            config = json.load(f)

        for task in config["tasks"]:
            dataset_info = DatasetInfo(
                size_mb=task["dataset"]["size_mb"], storage_nodes=task["dataset"]["storage_nodes"]
            )

            model_info = ModelInfo(size_mb=task["model"]["size_mb"], storage_nodes=task["model"]["storage_nodes"])

            task_data_info = TaskDataInfo(task_name=task["model_name"], dataset=dataset_info, model=model_info)

            self.task_data_info[task["model_name"]] = task_data_info

    def get_task_data_info(self, task_name: str) -> Optional[TaskDataInfo]:
        """获取指定任务的数据信息"""
        return self.task_data_info.get(task_name)

    def get_dataset_locations(self, model_name: str) -> list[str]:
        """获取指定模型的数据集存储位置"""
        task = self.get_task_data_info(model_name)
        if task:
            return task.dataset.storage_nodes
        return []

    def get_model_locations(self, model_name: str) -> list[str]:
        """获取指定模型的模型文件存储位置"""
        task = self.get_task_data_info(model_name)
        if task:
            return task.model.storage_nodes
        return []

    def get_dataset_size(self, model_name: str) -> int:
        """获取指定模型的数据集大小(MB)"""
        task = self.get_task_data_info(model_name)
        if task:
            return task.dataset.size_mb
        return 0

    def get_model_size(self, model_name: str) -> int:
        """获取指定模型的模型文件大小(MB)"""
        task = self.get_task_data_info(model_name)
        if task:
            return task.model.size_mb
        return 0

    def get_all_models(self) -> list[str]:
        """获取所有模型名称列表"""
        return list(self.task_data_info.keys())

    def get_avg_data_arrival_time(self, cluster_manager: ClusterManager) -> float:
        """获取所有模型数据集到达时间"""
        data_arrival_time_list = []

        for task_name in self.get_all_models():
            task_data_info = self.get_task_data_info(task_name)
            for cluster_id in cluster_manager.clusters.keys():
                data_arrival_time_list.append(
                    self.get_data_arrival_time(task_data_info, cluster_id, cluster_manager, "")
                )

        return sum(data_arrival_time_list) / len(data_arrival_time_list)

    def get_data_arrival_time(
        self, task_data_info: TaskDataInfo, target_cluster_id: str, cluster_manager: ClusterManager, scheduler_name: str
    ) -> float:
        # 获取模型和数据集的存储节点
        model_nodes = task_data_info.model.storage_nodes
        model_size = task_data_info.model.size_mb
        dataset_nodes = task_data_info.dataset.storage_nodes
        dataset_size = task_data_info.dataset.size_mb

        # 初始化最大带宽和对应的节点
        max_model_bandwidth = 0
        max_dataset_bandwidth = 0

        # 遍历模型存储节点，寻找带宽最大的节点
        for node_id in model_nodes:
            bandwidth = cluster_manager.get_bandwidth(node_id, target_cluster_id)
            if bandwidth > max_model_bandwidth:
                max_model_bandwidth = bandwidth

        # 遍历数据集存储节点，寻找带宽最大的节点
        for node_id in dataset_nodes:
            bandwidth = cluster_manager.get_bandwidth(node_id, target_cluster_id)
            if bandwidth > max_dataset_bandwidth:
                max_dataset_bandwidth = bandwidth

        # 计算模型数据到达时间
        model_arrival_time = float("inf")
        if max_model_bandwidth > 0:
            model_arrival_time = (model_size * 8) / max_model_bandwidth

        # 计算数据集到达时间
        dataset_arrival_time = float("inf")
        if max_dataset_bandwidth > 0:
            dataset_arrival_time = (dataset_size * 8) / max_dataset_bandwidth

        # 返回最大到达时间
        return max(model_arrival_time, dataset_arrival_time)
