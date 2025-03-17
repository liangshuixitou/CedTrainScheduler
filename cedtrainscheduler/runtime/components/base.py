import asyncio
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any


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

    @abstractmethod
    async def handle_message(self, message: dict[str, Any]):
        """Handle incoming messages"""
        pass
