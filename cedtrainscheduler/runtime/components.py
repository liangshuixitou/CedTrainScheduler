import asyncio
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum


class ComponentType(str, Enum):
    MASTER = "master"
    SCHEDULER = "scheduler"
    WORKER = "worker"


@dataclass
class ComponentInfo:
    component_type: ComponentType
    component_id: str
    component_ip: str
    component_port: int


class BaseComponent(ABC):
    """Base class for all runtime components (Master, Worker, Executor)"""

    def __init__(self, component_id: str):
        self.component_id = component_id
        self.logger = logging.getLogger(f"{self.__class__.__name__}-{component_id}")
        self._running = False
        self._event_loop = None

    async def start(self):
        """Start the component's main loop"""
        self._running = True
        self._event_loop = asyncio.get_event_loop()
        await self._run()

    async def stop(self):
        """Stop the component's main loop"""
        self._running = False

    @abstractmethod
    async def _run(self):
        """Main loop implementation"""
        pass
