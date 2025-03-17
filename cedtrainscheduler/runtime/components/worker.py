import asyncio
import json
import os
import socket
import time
from typing import Any

from cedtrainscheduler.runtime.communication.master import WorkerMasterClient
from cedtrainscheduler.runtime.communication.worker import WorkerServer
from cedtrainscheduler.runtime.components.base import BaseComponent
from cedtrainscheduler.runtime.components.executor import Executor
from cedtrainscheduler.runtime.components.master import Master
from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.cluster import Node
from cedtrainscheduler.runtime.types.server_info import ComponentInfo
from cedtrainscheduler.runtime.types.task import TaskInstStatus
from cedtrainscheduler.runtime.types.task import TaskMeta
from cedtrainscheduler.runtime.types.task import TaskStatus


class Worker(BaseComponent):
    """Worker component for node-level task management"""

    def __init__(self, worker_info: ComponentInfo, master_info: ComponentInfo):
        super().__init__(worker_info.component_id)
        self.node: Node = Node()

        # Communication
        self.server = WorkerServer(worker_info.component_id)
        self.master_client = WorkerMasterClient(master_info.component_id)

        # Executors
        self.executors: dict[str, Executor] = {}  # executor_id -> Executor

        # State
        self.tasks: dict[str, TaskInstStatus] = {}  # task_id -> TaskInstStatus

    async def _run(self):
        """Main loop for the worker component"""
        # Set up communication channels
        await self._setup_communication()

        # Initialize executors
        await self._init_executors()

        # Send initial status
        await self._send_status_update()

        # Main loop
        while self._running:
            # Send periodic status update to master
            await self._send_status_update()

            # Check executor health
            await self._check_executor_health()

            await asyncio.sleep(10)  # Status update every 10 seconds

    async def _setup_communication(self):
        """Set up communication channels"""
        # Set up handlers for incoming messages
        task_handlers = {f"task_{self.node.node_id}": self._handle_task_assignment}

        executor_handlers = {"executor_status": self._handle_executor_status, "task_status": self._handle_task_status}

        await self.messenger.start_subscriber_handlers(task_handlers)
        await self.messenger.start_subscriber_handlers(executor_handlers)

        self.logger.info(f"Worker communication setup complete")

    async def _init_communication(self):
        """Initialize communication components"""
        # Create server to receive Master requests
        await self.server.start(self.master_status_port, self._handle_master_request)

        # Create client to communicate with Master
        await self.client.connect(self.master_host, self.master_status_port)

        self.logger.info("Worker communication initialized")

    async def _handle_master_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle requests from Master"""
        action = request.get("action")

        if action == "assign_task":
            return await self._handle_task_assignment(request)
        elif action == "get_worker_status":
            return await self._get_worker_status()
        elif action == "start_training":
            return await self._handle_start_training(request)
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    async def _init_executors(self):
        """Initialize each GPU's Executor"""
        self.executors = {}  # gpu_id -> Executor instance

        for gpu in self.node.gpus:
            executor_config = {
                "gpu_id": gpu.gpu_id,
                "gpu_type": gpu.gpu_type.value,
                "gpu_rank": gpu.gpu_rank,
                "node_id": self.node.node_id,
                "cluster_id": self.node.cluster_id,
                "work_dir": os.path.expanduser(f"~/.cedtrain/executor/{gpu.gpu_id}"),
            }

            # Create executor directory
            os.makedirs(executor_config["work_dir"], exist_ok=True)

            # Create Executor instance directly
            executor_id = f"{self.node.node_id}_{gpu.gpu_id}"
            executor = Executor(executor_id, gpu, executor_config)

            # Set bidirectional reference
            executor.set_worker(self)  # Allow Executor to callback Worker

            self.executors[gpu.gpu_id] = executor

            # Start executor
            asyncio.create_task(executor.start())

            self.logger.info(f"Created and started executor for GPU {gpu.gpu_id}")

    async def _send_status_update(self):
        """Send worker status update to master"""
        status_message = {
            "type": "worker_status",
            "node_id": self.node.node_id,
            "status": "active",
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "gpu_count": len(self.node.gpus),
            "timestamp": time.time(),
        }

        await self.messenger.publish("status_pub", "worker_status", status_message)

    async def _check_executor_health(self):
        """Check health of executors"""
        for gpu_id, executor in self.executors.items():
            if not getattr(executor, "_running", False):
                self.logger.warning(f"Executor for GPU {gpu_id} is not running, restarting...")

                # In a real implementation, you would restart the executor
                # For this example, we'll just start it if it's not running
                asyncio.create_task(executor.start())

    async def _handle_task_assignment(self, request):
        """Handle task assignment from Master"""
        task_id = request.get("task_id")
        task_meta = request.get("task_meta")
        instances = request.get("instances", [])
        total_instances = request.get("total_instances", 0)

        self.logger.info(f"Received task {task_id} with {len(instances)} instances")

        # Store task information
        if task_id not in self.tasks:
            self.tasks[task_id] = {
                "task_meta": task_meta,
                "instances": instances,
                "total_instances": total_instances,
                "received_time": time.time(),
                "status": {},
            }

        # Collect all GPU assignment information
        gpu_assignments = {}
        for instance in instances:
            inst_id = instance["inst_id"]
            gpu_id = instance["gpu_id"]
            gpu_assignments[inst_id] = gpu_id

        # Directly distribute tasks to each Executor
        for instance in instances:
            inst_id = instance["inst_id"]
            gpu_id = instance["gpu_id"]

            # Check if Executor exists
            if gpu_id not in self.executors:
                self.logger.error(f"Executor for GPU {gpu_id} not found")
                continue

            executor = self.executors[gpu_id]

            # Directly call Executor's method
            success = await executor.assign_task(
                task_id=task_id,
                inst_id=inst_id,
                task_meta=task_meta,
                total_instances=total_instances,
                gpu_assignments=gpu_assignments,
            )

            # Update local state
            self.tasks[task_id]["status"][inst_id] = "pending"

            if success:
                self.logger.info(f"Assigned instance {inst_id} of task {task_id} to executor on GPU {gpu_id}")
            else:
                self.logger.error(f"Failed to assign instance {inst_id} of task {task_id} to executor on GPU {gpu_id}")

        return {"status": "success", "message": f"Task {task_id} with {len(instances)} instances assigned"}

    async def notify_task_ready(self, task_id, inst_id, gpu_id):
        """Called by Executor to notify task instance is ready"""
        if task_id not in self.tasks:
            return False

        # Update local state
        self.tasks[task_id]["status"][inst_id] = "ready"

        # Check if all local instances are ready
        all_ready = True
        for status in self.tasks[task_id]["status"].values():
            if status != "ready":
                all_ready = False
                break

        if all_ready:
            self.logger.info(f"All local instances of task {task_id} are ready")

            # Notify all Executors to start task
            for instance in self.tasks[task_id]["instances"]:
                instance_gpu_id = instance["gpu_id"]
                if instance_gpu_id in self.executors:
                    await self.executors[instance_gpu_id].notify_task_ready(task_id)

        # Send status update to Master
        await self._send_task_status_update(task_id, inst_id, "ready", gpu_id)

        return True

    async def _send_task_status_update(self, task_id, inst_id, status, gpu_id):
        """Send task status update to Master"""
        response = await self.client.send_task_status_update(
            self.master_host, self.master_status_port, task_id, inst_id, status, gpu_id
        )
        return response and response.get("status") == "success"

    async def _handle_executor_status(self, message: dict[str, Any]):
        """Handle status updates from executors"""
        gpu_id = message.get("gpu_id")
        status = message.get("status")
        current_task = message.get("current_task")
        queue_info = message.get("queue_info", [])

        # Forward status to master
        master_message = {
            "type": "executor_status",
            "node_id": self.node.node_id,
            "gpu_id": gpu_id,
            "status": status,
            "current_task": current_task,
            "queue_info": queue_info,
            "timestamp": time.time(),
        }

        await self.messenger.publish("status_pub", "executor_status", master_message)

    async def _handle_task_status(self, message: dict[str, Any]):
        """Handle task status updates from executors"""
        task_id = message.get("task_id")
        inst_id = message.get("inst_id")
        status = message.get("status")
        data_status = message.get("data_status")
        gpu_id = message.get("gpu_id")

        # Update local state
        if task_id in self.tasks and inst_id in self.tasks[task_id]["status"]:
            self.tasks[task_id]["status"][inst_id] = status

        # Forward status to master
        master_message = {
            "type": "task_status",
            "task_id": task_id,
            "inst_id": inst_id,
            "status": status,
            "data_status": data_status,
            "node_id": self.node.node_id,
            "gpu_id": gpu_id,
            "timestamp": time.time(),
        }

        await self.messenger.publish("status_pub", "task_status", master_message)

        # Check if all local instances are in ready state
        if task_id in self.tasks:
            all_ready = True
            for status_val in self.tasks[task_id]["status"].values():
                if status_val != TaskInstStatus.Ready.value:
                    all_ready = False
                    break

            if all_ready:
                self.logger.info(f"All local instances of task {task_id} are ready")
                await self._notify_executors_task_ready(task_id)

    async def _notify_executors_task_ready(self, task_id: str):
        """Notify all involved executors that a task is ready to start"""
        if task_id not in self.tasks:
            return

        task_info = self.tasks[task_id]
        instances = task_info["instances"]

        # Notify each involved executor
        for instance in instances:
            gpu_id = instance["gpu_id"]

            # Send ready notification
            ready_message = {"task_id": task_id, "status": "ready_to_start", "timestamp": time.time()}

            topic = f"task_ready_{gpu_id}"
            await self.messenger.publish("executor_task_pub", topic, ready_message)

            self.logger.info(f"Notified executor {gpu_id} that task {task_id} is ready to start")

    async def handle_message(self, message: dict[str, Any]):
        """Handle incoming messages"""
        message_type = message.get("type")

        if message_type == "task_assignment":
            await self._handle_task_assignment(message)
        elif message_type == "executor_status":
            await self._handle_executor_status(message)
        elif message_type == "task_status":
            await self._handle_task_status(message)
        else:
            self.logger.warning(f"Unknown message type: {message_type}")

    async def _get_worker_status(self):
        """Get worker status"""
        status_message = {
            "type": "worker_status",
            "node_id": self.node.node_id,
            "status": "active",
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "gpu_count": len(self.node.gpus),
            "timestamp": time.time(),
        }
        return status_message

    async def _handle_start_training(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle start training request"""
        # Implementation of start training logic
        return {"status": "success", "message": "Training started"}
