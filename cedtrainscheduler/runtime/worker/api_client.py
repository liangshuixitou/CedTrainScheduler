from typing import Optional

import requests

from cedtrainscheduler.runtime.types.model import TaskInstModel
from cedtrainscheduler.runtime.types.model import WorkerTaskInstStartModel
from cedtrainscheduler.runtime.types.model import WorkerTaskInstSubmitModel
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.utils.logger import setup_logger


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

        data = WorkerTaskInstSubmitModel(task_inst=TaskInstModel.from_task_inst(task_inst), gpu_id=gpu_id).model_dump()
        return await self._make_request("/api/task/inst/submit", data)

    async def start_task_inst(
        self,
        task_inst: TaskInst,
        gpu_id: str,
        task_name: str,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: int,
    ) -> Optional[dict]:
        """
        启动任务实例

        Args:
            task_inst: 任务实例
            gpu_id: GPU ID
            task_name: 任务名称
            world_size: 世界大小
            inst_rank: 任务实例排名
            master_addr: 主节点地址
            master_port: 主节点端口

        Returns:
            Optional[dict]: 任务启动结果，失败时返回None
        """
        data = WorkerTaskInstStartModel(
            task_inst=TaskInstModel.from_task_inst(task_inst),
            gpu_id=gpu_id,
            task_name=task_name,
            world_size=world_size,
            inst_rank=inst_rank,
            master_addr=master_addr,
            master_port=master_port,
        ).model_dump()
        return await self._make_request("/api/task/inst/start", data)
