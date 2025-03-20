import logging
from typing import Optional

import requests

from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo


class BaseClient:
    """API客户端基类"""

    def __init__(self, worker_host: str, worker_port: int):
        """
        初始化客户端

        Args:
            worker_host: Worker主机地址
            worker_port: Worker端口
        """
        self.base_url = f"http://{worker_host}:{worker_port}"
        self.logger = logging.getLogger(__name__)

    def _make_request(self, endpoint: str, data: dict) -> Optional[dict]:
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


class MasterWorkerClient(BaseClient):
    """Worker客户端，用于任务提交"""

    async def submit_task(self, task_inst: TaskInst, gpu_id: str) -> Optional[dict]:
        """
        向Worker提交任务

        Args:
            task: 任务包装的运行时信息

        Returns:
            Optional[dict]: 任务提交结果，失败时返回None
        """

        self.logger.info(f"提交任务 {task_inst.task_id} 到Worker")
        data = {"task_inst": task_inst.__dict__, "gpu_id": gpu_id}
        return self._make_request("/api/task/inst/submit", data)

    async def start_task_inst(
        self, task_inst: TaskInst, gpu_id: str, task_record: dict[str, TaskWrapRuntimeInfo]
    ) -> Optional[dict]:
        """
        启动任务实例

        Args:
            task_inst: 任务实例
            gpu_id: GPU ID
            task_record: 任务记录

        Returns:
            Optional[dict]: 任务启动结果，失败时返回None
        """
        data = {"task_inst": task_inst.__dict__, "gpu_id": gpu_id, "task_record": task_record.__dict__}
        return self._make_request("/api/task/inst/start", data)
