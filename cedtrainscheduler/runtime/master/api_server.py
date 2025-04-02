import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI
from uvicorn.config import Config

from cedtrainscheduler.runtime.master.service import MasterService
from cedtrainscheduler.runtime.types.model import MasterTaskSubmitModel
from cedtrainscheduler.runtime.types.model import MasterWorkerRegisterModel
from cedtrainscheduler.runtime.utils.logger import setup_logger


class MasterAPIServer:
    """FastAPI server for Master component - 轻量化实现"""

    def __init__(self, master_service: MasterService):
        """
        初始化Master API服务器

        Args:
            master_service: MasterService实例，用于处理请求
        """
        self.master_service = master_service
        self.app = FastAPI(title="Master API", version="1.0.0")
        self.logger = setup_logger(__name__)
        self.server: Optional[uvicorn.Server] = None
        self.server_task: Optional[asyncio.Task] = None
        self.setup_routes()

    def setup_routes(self):
        """配置API端点"""

        @self.app.post("/api/task/submit")
        async def handle_task_submit(request: MasterTaskSubmitModel):
            """处理来自Worker的任务提交"""
            # 使用 Pydantic 模型的转换方法生成自定义类对象
            task_info = request.task.to_task_wrap_runtime_info()
            sim_data_transfer_time = request.sim_data_transfer_time
            return await self.master_service.handle_task_submit(task_info, sim_data_transfer_time)

        @self.app.post("/api/worker/register")
        async def handle_worker_register(request: MasterWorkerRegisterModel):
            """处理工作节点注册"""
            # 使用 Pydantic 模型的转换方法生成自定义类对象
            node = request.node.to_node()
            tasks = [task.to_task_inst() for task in request.tasks]
            task_queue_map = {
                gpu_id: [task.to_task_inst() for task in queue_tasks]
                for gpu_id, queue_tasks in request.task_queue_map.items()
            }
            return await self.master_service.handle_worker_register(node, tasks, task_queue_map)

    async def start(self, host="0.0.0.0", port=5001) -> asyncio.Task:
        """启动API服务器"""
        config = Config(app=self.app, host=host, port=port, log_level="info")
        self.server = uvicorn.Server(config)

        self.server_task = asyncio.create_task(self.server.serve())

        self.logger.info(f"Master API server started on {host}:{port}")
        return self.server_task

    async def stop(self):
        """停止API服务器"""
        if self.server:
            self.server.should_exit = True
            if self.server_task:
                await self.server_task
            self.logger.info("Master API server stopping")
