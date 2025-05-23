import asyncio
from typing import Optional

import uvicorn
from fastapi import FastAPI
from uvicorn.config import Config

from cedtrainscheduler.runtime.types.model import WorkerTaskInstStartModel
from cedtrainscheduler.runtime.types.model import WorkerTaskInstSubmitModel
from cedtrainscheduler.runtime.utils.logger import setup_logger
from cedtrainscheduler.runtime.worker.service import WorkerService


class WorkerAPIServer:
    """FastAPI server for Worker component - 轻量化实现"""

    def __init__(self, worker_service: WorkerService):
        """
        初始化Worker API服务器

        Args:
            worker: Worker实例，用于处理请求
        """
        self.worker_service = worker_service
        self.app = FastAPI(title="Worker API", version="1.0.0")
        self.server: Optional[uvicorn.Server] = None
        self.server_task: Optional[asyncio.Task] = None
        self.setup_routes()

        self.logger = setup_logger(__name__)

    def setup_routes(self):
        """配置API端点"""

        @self.app.post("/api/task/inst/submit")
        async def handle_task_inst_submit(request: WorkerTaskInstSubmitModel):
            """处理来自Master的任务提交"""
            task_inst = request.task_inst.to_task_inst()
            gpu_id = request.gpu_id
            return await self.worker_service.handle_task_inst_submit(task_inst, gpu_id)

        @self.app.post("/api/task/inst/start")
        async def handle_task_inst_start(request: WorkerTaskInstStartModel):
            """处理来自Master的任务启动"""
            task_inst = request.task_inst.to_task_inst()
            gpu_id = request.gpu_id
            task_name = request.task_name
            world_size = request.world_size
            inst_rank = request.inst_rank
            master_addr = request.master_addr
            master_port = request.master_port
            plan_runtime = request.plan_runtime
            data_transfer_time = request.data_transfer_time
            return await self.worker_service.handle_task_inst_start(
                task_inst,
                gpu_id,
                task_name,
                world_size,
                inst_rank,
                master_addr,
                master_port,
                plan_runtime,
                data_transfer_time,
            )

        @self.app.post("/api/task/log/{task_id}/{inst_id}/{gpu_id}")
        async def handle_task_log(task_id: str, inst_id: int, gpu_id: str):
            """处理获取任务日志请求"""
            return await self.worker_service.handle_task_log(task_id, inst_id, gpu_id)

    async def start(self, host="0.0.0.0", port=5002) -> asyncio.Task:
        """启动API服务器"""
        config = Config(app=self.app, host=host, port=port, log_level="info")
        self.server = uvicorn.Server(config)

        self.server_task = asyncio.create_task(self.server.serve())
        self.logger.info(f"Worker API server started on {host}:{port}")
        return self.server_task

    async def stop(self):
        """停止API服务器"""
        if self.server:
            self.server.should_exit = True
            if self.server_task:
                await self.server_task
            self.logger.info("Worker API server stopping")
