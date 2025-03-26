import asyncio
import logging
import signal
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum


class ComponentType(str, Enum):
    MANAGER = "manager"
    MASTER = "master"
    WORKER = "worker"


@dataclass
class ComponentInfo:
    component_type: ComponentType
    component_id: str
    component_ip: str
    component_port: int


class BaseServer(ABC):
    def __init__(self, component_info: ComponentInfo):
        self.component_info = component_info
        self._running = False
        self._stop_event = asyncio.Event()
        self._server = None
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """设置信号处理器"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理系统信号"""
        logging.info(f"收到信号 {signum}，准备停止服务...")
        asyncio.create_task(self.stop())

    async def start(self):
        """启动服务器（非阻塞）"""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        # 启动必要的后台任务
        self._server = asyncio.create_task(self._serve())
        logging.info(f"服务器 {self.component_info.component_id} 已启动")

    async def run(self):
        """运行服务器（阻塞）"""
        await self.start()
        try:
            await self._stop_event.wait()
        finally:
            await self.stop()

    async def stop(self):
        """停止服务器"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        logging.info(f"服务器 {self.component_info.component_id} 已停止")

    @abstractmethod
    async def _serve(self):
        """运行后台任务的具体实现"""
        pass
