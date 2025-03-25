import logging
from typing import Optional

import requests

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.types.cluster import Cluster
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo


class BaseClient:
    """API客户端基类"""

    def __init__(self, manager_host: str, manager_port: int):
        """
        初始化客户端

        Args:
            manager_host: Manager主机地址
            manager_port: Manager端口
        """
        self.base_url = f"http://{manager_host}:{manager_port}"
        self.logger = logging.getLogger(__name__)

    async def _make_request(self, endpoint: str, data: dict) -> Optional[dict]:
        """
        发送HTTP请求到服务器

        Args:
            endpoint: API端点路径
            data: 请求数据

        Returns:
            Optional[dict]: 响应数据，失败时返回None
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()  # 如果HTTP请求返回了不成功的状态码，将抛出HTTPError异常
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"请求失败: {e}")
            return None


class MasterManagerClient(BaseClient):
    """Master客户端，用于注册"""

    async def register_master(
        self, cluster: Cluster, task_infos: dict[str, TaskWrapRuntimeInfo], master_info: ComponentInfo
    ) -> Optional[dict]:
        """
        注册Master到Manager

        Args:
            cluster: 集群信息
            task_infos: 集群上的任务信息
            master_info: Master信息
        Returns:
            Optional[dict]: 注册结果，失败时返回None
        """
        data = {
            "cluster": cluster.__dict__,
            "task_infos": {task_id: task.__dict__ for task_id, task in task_infos.items()},
            "master_info": master_info.__dict__,
        }

        self.logger.info(f"注册Master {cluster.cluster_id} 到Manager")
        return self._make_request("/api/master/register", data)


class TaskManagerClient(BaseClient):
    """Task客户端，用于提交任务"""

    async def submit_task(self, task: TaskWrapRuntimeInfo) -> Optional[dict]:
        """
        向Manager提交任务

        Args:
            task: 任务包装的运行时信息

        Returns:
            Optional[dict]: 任务提交结果，失败时返回None
        """
        data = {"task": task.__dict__}

        self.logger.info(f"提交任务 {task.task.task_id} 到Master")
        return self._make_request("/api/task/submit", data)
