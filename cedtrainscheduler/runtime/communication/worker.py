"""Worker 服务器模块"""

from typing import Any
from typing import Callable
from typing import dict

from cedtrainscheduler.runtime.communication.base import BaseClient
from cedtrainscheduler.runtime.communication.base import BaseServer


class WorkerServer(BaseServer):
    """Worker 服务器，接收来自 Master 的请求"""

    def __init__(self, worker_id: str):
        super().__init__(worker_id, "Worker")

    async def register_handler(self, handler: Callable):
        """注册请求处理函数"""
        self.handler = handler


class WorkerClient(BaseClient):
    """Worker 客户端，用于Master发送请求给Worker"""

    def __init__(self, master_id: str):
        super().__init__(master_id, "Master")

    async def assign_task(
        self,
        worker_host: str,
        worker_port: int,
        task_id: str,
        task_meta: dict[str, Any],
        instances: dict[int, dict[str, Any]],
        total_instances: int,
    ) -> dict[str, Any]:
        """分配任务到 Worker"""
        # 连接到 Worker
        if not await self.connect(worker_host, worker_port):
            return {"status": "error", "message": f"Failed to connect to Worker at {worker_host}:{worker_port}"}

        # 准备请求
        request = {
            "action": "assign_task",
            "task_id": task_id,
            "task_meta": task_meta,
            "instances": instances,
            "total_instances": total_instances,
        }

        # 发送请求
        response = await self.send_request(request)

        # 关闭连接
        await self.close()

        return response

    async def get_worker_status(self, worker_host: str, worker_port: int) -> dict[str, Any]:
        """获取 Worker 状态"""
        # 连接到 Worker
        if not await self.connect(worker_host, worker_port):
            return {"status": "error", "message": f"Failed to connect to Worker at {worker_host}:{worker_port}"}

        # 准备请求
        request = {"action": "get_worker_status"}

        # 发送请求
        response = await self.send_request(request)

        # 关闭连接
        await self.close()

        return response
