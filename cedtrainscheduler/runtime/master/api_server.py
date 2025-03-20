import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI
from uvicorn.config import Config

from cedtrainscheduler.runtime.master.master import Master
from cedtrainscheduler.runtime.master.types import TaskSubmitModel
from cedtrainscheduler.runtime.master.types import WorkerRegisterModel


class MasterAPIServer:
    """FastAPI server for Master component - 轻量化实现"""

    def __init__(self, master: Master):
        """
        初始化Master API服务器

        Args:
            master: Master实例，用于处理请求
        """
        self.master = master
        self.app = FastAPI(title="Master API", version="1.0.0")
        self.logger = logging.getLogger(__name__)
        self.server: Optional[uvicorn.Server] = None
        self.setup_routes()

    def setup_routes(self):
        """配置API端点"""

        @self.app.post("/api/task/submit")
        async def handle_task_submit(request: TaskSubmitModel):
            """处理来自Worker的任务提交"""
            # 使用 Pydantic 模型的转换方法生成自定义类对象
            task_info = request.task.to_task_wrap_runtime_info()
            return await self.master.handle_task_submit(task_info)

        @self.app.post("/api/worker/register")
        async def handle_worker_register(request: WorkerRegisterModel):
            """处理工作节点注册"""
            # 使用 Pydantic 模型的转换方法生成自定义类对象
            node = request.node.to_node()
            tasks = [task.to_task_inst() for task in request.tasks]
            return await self.master.handle_worker_register(node, tasks)

    async def start(self, host="0.0.0.0", port=5000):
        """启动API服务器"""
        config = Config(app=self.app, host=host, port=port, log_level="info")
        self.server = uvicorn.Server(config)

        # 使用异步任务替代线程
        import asyncio

        self.server_task = asyncio.create_task(self.server.serve())

        self.logger.info(f"Master API server started on {host}:{port}")

    async def stop(self):
        """停止API服务器"""
        if self.server:
            self.server.should_exit = True
            if hasattr(self, "server_task"):
                await self.server_task
            self.logger.info("Master API server stopping")
