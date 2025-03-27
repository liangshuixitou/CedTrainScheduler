from typing import Optional

import requests

from cedtrainscheduler.runtime.types.cluster import Node
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.runtime.utils.logger import setup_logger


class BaseClient:
    """API客户端基类"""

    def __init__(self, master_host: str, master_port: int):
        """
        初始化客户端

        Args:
            master_host: Master主机地址
            master_port: Master端口
        """
        self.base_url = f"http://{master_host}:{master_port}"
        self.logger = setup_logger(__name__)

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
            self.logger.error(f"Request to {url} failed: {e}")
            return None


class WorkerMasterClient(BaseClient):
    """Worker客户端，用于工作节点注册"""

    async def register_worker(self, node: Node, tasks: list[TaskInst]) -> Optional[dict]:
        """
        注册工作节点到Master

        Args:
            node: 节点信息
            tasks: 节点上的任务信息

        Returns:
            Optional[dict]: 注册结果，失败时返回None
        """
        data = {"node": node.__dict__, "tasks": [task.__dict__ for task in tasks]}

        self.logger.info(f"Register worker {node.node_id} to master")
        return self._make_request("/api/worker/register", data)


class ManagerMasterClient(BaseClient):
    """管理客户端，用于任务提交"""

    async def submit_task(self, task: TaskWrapRuntimeInfo) -> Optional[dict]:
        """
        向Master提交任务

        Args:
            task: 任务包装的运行时信息

        Returns:
            Optional[dict]: 任务提交结果，失败时返回None
        """
        data = {"task": task.__dict__}

        self.logger.info(f"Submit task {task.task_meta.task_id} to master")
        return self._make_request("/api/task/submit", data)
