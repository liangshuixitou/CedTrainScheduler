import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI
from uvicorn.config import Config

from cedtrainscheduler.runtime.manager.service import ManagerService
from cedtrainscheduler.runtime.manager.types import MasterRegisterModel
from cedtrainscheduler.runtime.manager.types import TaskSubmitModel
from cedtrainscheduler.runtime.utils.logger import setup_logger


class ManagerAPIServer:
    """FastAPI server for Manager component - 轻量化实现"""

    def __init__(self, manager_service: ManagerService):
        """
        初始化Manager API服务器

        Args:
            manager: Manager实例，用于处理请求
        """
        self.manager_service = manager_service
        self.app = FastAPI(title="Manager API", version="1.0.0")
        self.server: Optional[uvicorn.Server] = None
        self.server_task = None
        self.setup_routes()

        self.logger = setup_logger(__name__)

    def setup_routes(self):
        """配置API端点"""

        @self.app.post("/api/task/submit")
        async def handle_task_submit(request: TaskSubmitModel):
            """处理来自Worker的任务提交"""
            # 使用 Pydantic 模型的转换方法生成自定义类对象
            task_meta = request.task.to_task_meta()
            return await self.manager_service.handle_task_submit(task_meta)

        @self.app.post("/api/master/register")
        async def handle_master_register(request: MasterRegisterModel):
            """处理Master注册"""
            # 使用 Pydantic 模型的转换方法生成自定义类对象
            cluster = request.cluster.to_cluster()
            task_infos = {task_id: task.to_task_wrap_runtime_info() for task_id, task in request.task_infos.items()}
            master_info = request.master_info.to_component_info()
            return await self.manager_service.handle_master_register(cluster, task_infos, master_info)

    async def start(self, host="0.0.0.0", port=5000) -> asyncio.Task:
        """启动API服务器，返回服务器运行的Task"""
        config = Config(app=self.app, host=host, port=port, log_level="info")
        self.server = uvicorn.Server(config)

        # 创建服务器任务但不在内部管理
        self.server_task = asyncio.create_task(self.server.serve())
        self.logger.info(f"Manager API server started on {host}:{port}")
        return self.server_task

    async def stop(self):
        """停止API服务器"""
        if self.server:
            self.server.should_exit = True
            if self.server_task:
                await self.server_task
            self.logger.info("Manager API server stopping")
