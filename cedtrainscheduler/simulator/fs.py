import json
from dataclasses import dataclass
from typing import Optional

from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo


@dataclass
class DatasetInfo:
    """数据集信息"""

    size_mb: int  # 数据集大小(MB)
    storage_nodes: list[str]  # 存储节点IP列表


@dataclass
class ModelInfo:
    """模型信息"""

    size_mb: int  # 模型大小(MB)
    storage_nodes: list[str]  # 存储节点IP列表


@dataclass
class TaskDataInfo:
    task_name: str  # 任务名称
    dataset: DatasetInfo  # 数据集信息
    model: ModelInfo  # 模型信息


class FileSystem:
    """文件系统模拟器"""

    def __init__(self, config_path: str):
        self.task_data_info: dict[str, TaskDataInfo] = {}  # 任务名称 -> 任务信息
        self.init_task_data(config_path)

    def init_task_data(self, config_path: str) -> None:
        """从配置文件加载训练任务信息"""
        with open(config_path) as f:
            config = json.load(f)

        for task in config["tasks"]:
            dataset_info = DatasetInfo(
                size_mb=task["dataset"]["size_mb"], storage_nodes=task["dataset"]["storage_nodes"]
            )

            model_info = ModelInfo(size_mb=task["model"]["size_mb"], storage_nodes=task["model"]["storage_nodes"])

            task_data_info = TaskDataInfo(task_name=task["task_name"], dataset=dataset_info, model=model_info)

            self.task_data_info[task["task_name"]] = task_data_info

    def get_task(self, task_name: str) -> Optional[TaskDataInfo]:
        """获取指定任务的数据信息"""
        return self.tasks.get(task_name)

    def get_dataset_locations(self, model_name: str) -> list[str]:
        """获取指定模型的数据集存储位置"""
        task = self.get_task(model_name)
        if task:
            return task.dataset.storage_nodes
        return []

    def get_model_locations(self, model_name: str) -> list[str]:
        """获取指定模型的模型文件存储位置"""
        task = self.get_task(model_name)
        if task:
            return task.model.storage_nodes
        return []

    def get_dataset_size(self, model_name: str) -> int:
        """获取指定模型的数据集大小(MB)"""
        task = self.get_task(model_name)
        if task:
            return task.dataset.size_mb
        return 0

    def get_model_size(self, model_name: str) -> int:
        """获取指定模型的模型文件大小(MB)"""
        task = self.get_task(model_name)
        if task:
            return task.model.size_mb
        return 0

    def get_all_models(self) -> list[str]:
        """获取所有模型名称列表"""
        return list(self.tasks.keys())

    def get_task_type(self, model_name: str) -> str:
        """获取指定模型的任务类型"""
        task = self.get_task(model_name)
        if task:
            return task.task_type
        return ""

    def get_data_arival_time(self, task: TaskWrapRuntimeInfo, node_id: str) -> float:
        return 0
