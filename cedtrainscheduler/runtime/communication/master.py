"""Master 服务器模块"""

import time
from typing import Any
from typing import Callable

from cedtrainscheduler.runtime.communication.base import BaseClient
from cedtrainscheduler.runtime.communication.base import BaseServer


class MasterServer(BaseServer):
    """Master 服务器, 接收来自 Scheduler 的请求"""

    def __init__(self, master_id: str):
        super().__init__(master_id, "Master")

    async def register_handler(self, handler: Callable):
        """注册请求处理函数"""
        self.handler = handler


class SchedulerMasterClient(BaseClient):
    """Master 客户端, 用于scheduler发送请求给Master"""

    def __init__(self, scheduler_id: str):
        super().__init__(scheduler_id, "Scheduler")

    async def submit_task(
        self, master_host: str, master_port: int, task_meta: dict[str, Any], schedule_infos: dict[int, dict[str, Any]]
    ) -> dict[str, Any]:
        """提交任务到 Master"""
        # 连接到 Master
        if not await self.connect(master_host, master_port):
            return {"status": "error", "message": "Failed to connect to Master"}

        # 准备请求
        request = {"action": "submit_task", "task_meta": task_meta, "schedule_infos": schedule_infos}

        # 发送请求
        response = await self.send_request(request)

        # 关闭连接
        await self.close()

        return response

    async def get_task_status(self, master_host: str, master_port: int, task_id: str) -> dict[str, Any]:
        """获取任务状态"""
        # 连接到 Master
        if not await self.connect(master_host, master_port):
            return {"status": "error", "message": "Failed to connect to Master"}

        # 准备请求
        request = {"action": "get_task_status", "task_id": task_id}

        # 发送请求
        response = await self.send_request(request)

        # 关闭连接
        await self.close()

        return response

    async def get_cluster_status(self, master_host: str, master_port: int) -> dict[str, Any]:
        """获取集群状态"""
        # 连接到 Master
        if not await self.connect(master_host, master_port):
            return {"status": "error", "message": "Failed to connect to Master"}

        # 准备请求
        request = {"action": "get_cluster_status"}

        # 发送请求
        response = await self.send_request(request)

        # 关闭连接
        await self.close()

        return response


class WorkerMasterClient(BaseClient):
    """Master 客户端, 用于worker发送请求给Master"""

    def __init__(self, worker_id: str):
        super().__init__(worker_id, "Worker")

    async def send_status_update(self, master_host: str, master_port: int, status: dict[str, Any]) -> dict[str, Any]:
        """发送状态更新到 Master"""
        # 连接到 Master
        if not await self.connect(master_host, master_port):
            return {"status": "error", "message": f"Failed to connect to Master at {master_host}:{master_port}"}

        # 准备请求
        request = {"action": "update_worker_status", "status": status}

        # 发送请求
        response = await self.send_request(request)

        # 关闭连接
        await self.close()

        return response

    async def send_task_status_update(
        self, master_host: str, master_port: int, task_id: str, inst_id: int, status: str, gpu_id: str
    ) -> dict[str, Any]:
        """发送任务状态更新到 Master"""
        # 连接到 Master
        if not await self.connect(master_host, master_port):
            return {"status": "error", "message": f"Failed to connect to Master at {master_host}:{master_port}"}

        # 准备请求
        request = {
            "action": "update_task_status",
            "task_id": task_id,
            "inst_id": inst_id,
            "status": status,
            "gpu_id": gpu_id,
            "timestamp": time.time(),
        }

        # 发送请求
        response = await self.send_request(request)

        # 关闭连接
        await self.close()

        return response
