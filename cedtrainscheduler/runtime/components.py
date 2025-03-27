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
        self._tasks: list[asyncio.Task] = []
        self._logger = logging.getLogger(__name__)
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """设置信号处理器"""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理系统信号"""
        self._logger.info(f"Received signal {signum}, preparing to stop...")
        asyncio.create_task(self.stop())

    async def start(self):
        """启动服务器（非阻塞）"""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        # 启动所有服务
        await self._start()
        self._logger.info(f"Server {self.component_info.component_id} started")

    @abstractmethod
    async def _start(self):
        """启动所有需要的服务，由子类实现"""
        pass

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

        # 取消所有运行中的任务
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._tasks.clear()
        await self._stop()

        self._logger.info(f"Server {self.component_info.component_id} stopped")

    @abstractmethod
    async def _stop(self):
        """停止所有服务，由子类实现"""
        pass
