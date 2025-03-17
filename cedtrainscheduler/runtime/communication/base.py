"""基础通信组件"""

import asyncio
import json
import logging
from typing import Any
from typing import Callable

import zmq
import zmq.asyncio


class BaseClient:
    """基础客户端类"""

    def __init__(self, client_id: str, role: str):
        self.client_id = client_id
        self.role = role
        self.logger = logging.getLogger(f"{role}Client-{client_id}")
        self.context = zmq.asyncio.Context()
        self.socket = None
        self.connected = False

    async def connect(self, server_host: str, server_port: int) -> bool:
        """连接到服务器"""
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(f"tcp://{server_host}:{server_port}")
            self.connected = True
            self.logger.info(f"Connected to server at {server_host}:{server_port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to server: {e}")
            return False

    async def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """发送请求并等待响应"""
        if not self.connected or not self.socket:
            self.logger.error("Not connected to server")
            return {"status": "error", "message": "Not connected to server"}

        try:
            # 序列化请求
            data = json.dumps(request).encode("utf-8")

            # 发送请求
            await self.socket.send(data)

            # 接收响应
            response_data = await self.socket.recv()
            response = json.loads(response_data.decode("utf-8"))

            return response
        except Exception as e:
            self.logger.error(f"Error sending request: {e}")
            return {"status": "error", "message": str(e)}

    async def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            self.connected = False
            self.logger.info("Closed connection to server")


class BaseServer:
    """基础服务器类"""

    def __init__(self, server_id: str, role: str):
        self.server_id = server_id
        self.role = role
        self.logger = logging.getLogger(f"{role}Server-{server_id}")
        self.context = zmq.asyncio.Context()
        self.socket = None
        self.running = False
        self.handler = None
        self.task = None

    async def start(self, port: int, handler: Callable) -> bool:
        """启动服务器监听"""
        try:
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{port}")
            self.handler = handler
            self.running = True

            # 启动请求处理循环
            self.task = asyncio.create_task(self._handle_requests())

            self.logger.info(f"Server started on port {port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False

    async def _handle_requests(self):
        """处理请求循环"""
        while self.running:
            try:
                # 接收请求
                request_data = await self.socket.recv()

                # 解析请求
                request = json.loads(request_data.decode("utf-8"))

                # 处理请求
                try:
                    response = await self.handler(request)
                except Exception as e:
                    self.logger.error(f"Error handling request: {e}")
                    response = {"status": "error", "message": str(e)}

                # 发送响应
                response_data = json.dumps(response).encode("utf-8")
                await self.socket.send(response_data)

            except Exception as e:
                self.logger.error(f"Error in request handling loop: {e}")
                await asyncio.sleep(1)  # 防止紧密循环

    async def stop(self):
        """停止服务器"""
        self.running = False

        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        if self.socket:
            self.socket.close()

        self.logger.info("Server stopped")
