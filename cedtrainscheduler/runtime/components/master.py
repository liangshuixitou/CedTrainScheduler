import asyncio
import time
from typing import Any

from cedtrainscheduler.runtime.communication.master import MasterServer
from cedtrainscheduler.runtime.communication.master import WorkerMasterClient
from cedtrainscheduler.runtime.components.base import BaseComponent
from cedtrainscheduler.runtime.types.cluster import Cluster
from cedtrainscheduler.runtime.types.task import TaskInstStatus
from cedtrainscheduler.runtime.types.task import TaskMeta
from cedtrainscheduler.runtime.types.task import TaskStatus
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo


class Master(BaseComponent):
    """Master component for cluster-level task management"""

    def __init__(self, master_id: str):
        super().__init__(master_id)

        # Communication
        self.server = MasterServer(master_id)
        self.clients: dict[str, WorkerMasterClient] = {}  # worker_id -> Worker Client

        # State
        self.cluster: Cluster = Cluster()
        self.tasks: dict[str, TaskWrapRuntimeInfo] = {}  # task_id -> TaskWrapRuntimeInfo

    async def _run(self):
        """Main loop for the master component"""
        # Set up communication channels
        await self._setup_communication()

        # Main loop for monitoring and management
        while self._running:
            await self._check_worker_health()
            await asyncio.sleep(5)  # Health check every 5 seconds

    async def _setup_communication(self):
        # Request handler for scheduler queries
        await self.server.start(self.request_port, self._handle_scheduler_request)

        self.logger.info("Master communication setup complete")

    async def _check_worker_health(self):
        """Check worker health based on heartbeats"""
        current_time = time.time()
        for node_id, worker_info in self.workers.items():
            # If no heartbeat for 30 seconds, mark as disconnected
            if current_time - worker_info["last_heartbeat"] > 30:
                if worker_info["status"] != "disconnected":
                    self.logger.warning(f"Worker {node_id} appears to be disconnected")
                    worker_info["status"] = "disconnected"

                    # Handle tasks that were assigned to this worker
                    await self._handle_worker_disconnect(node_id)

    async def _handle_worker_disconnect(self, node_id: str):
        """Handle a worker disconnection"""
        # Find tasks assigned to this worker and reschedule them
        affected_tasks = []
        for task_id, task_info in self.tasks.items():
            needs_rescheduling = False
            for inst_id, schedule_info in task_info.schedule_infos.items():
                for node_gpu in self.cluster.nodes:
                    if node_gpu.node_id == node_id and any(gpu.gpu_id == schedule_info.gpu_id for gpu in node_gpu.gpus):
                        needs_rescheduling = True
                        break

            if needs_rescheduling:
                affected_tasks.append(task_id)

        # Mark affected tasks for rescheduling
        for task_id in affected_tasks:
            self.logger.info(f"Task {task_id} needs rescheduling due to worker {node_id} disconnect")
            # In a real implementation, you would notify the scheduler to reschedule

    async def _handle_worker_status(self, message: dict[str, Any]):
        """Handle worker status updates"""
        node_id = message.get("node_id")
        status = message.get("status")

        if node_id in self.workers:
            self.workers[node_id]["status"] = status
            self.workers[node_id]["last_heartbeat"] = time.time()
            self.logger.debug(f"Updated worker {node_id} status to {status}")

    async def _handle_executor_status(self, message: dict[str, Any]):
        """Handle executor status updates"""
        node_id = message.get("node_id")
        gpu_id = message.get("gpu_id")
        status = message.get("status")
        current_task = message.get("current_task")
        task_queue = message.get("task_queue", [])

        if node_id in self.workers and gpu_id in self.workers[node_id]["executors"]:
            executor_info = self.workers[node_id]["executors"][gpu_id]
            executor_info["status"] = status
            executor_info["current_task"] = current_task
            executor_info["task_queue"] = task_queue

            # Update GPU status
            self.gpu_status[node_id][gpu_id] = task_queue

            self.logger.debug(f"Updated executor {node_id}:{gpu_id} status")

    async def _handle_task_status(self, message: dict[str, Any]):
        """Handle task status updates"""
        task_id = message.get("task_id")
        inst_id = message.get("inst_id")
        status = message.get("status")
        data_status = message.get("data_status", None)

        if task_id in self.tasks:
            task_info = self.tasks[task_id]

            # Update instance status
            if inst_id is not None:
                if status:
                    task_info.inst_status[inst_id] = TaskInstStatus(status)
                if data_status:
                    task_info.inst_data_status[inst_id] = data_status

            # Check if all instances are finished
            all_finished = all(s == TaskInstStatus.Finished for s in task_info.inst_status.values())
            if all_finished and task_info.task_meta.task_status != TaskStatus.Finished:
                task_info.task_meta.task_status = TaskStatus.Finished
                task_info.task_end_time = time.time()
                self.logger.info(f"Task {task_id} is now completed")

                # Notify scheduler about task completion
                await self._notify_task_completion(task_id)

    async def _notify_task_completion(self, task_id: str):
        """Notify scheduler about task completion"""
        # In a real implementation, this would notify the scheduler
        # This is a placeholder for the actual implementation
        self.logger.info(f"Notifying scheduler about completion of task {task_id}")

    async def _handle_scheduler_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle requests from the scheduler"""
        action = request.get("action")

        if action == "submit_task":
            return await self._handle_task_submission(request)
        elif action == "get_task_status":
            return await self._get_task_status(request)
        elif action == "get_cluster_status":
            return await self._get_cluster_status()
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    async def _handle_task_submission(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle task submission from scheduler"""
        task_meta = TaskMeta(**request.get("task_meta"))
        schedule_infos = {int(k): v for k, v in request.get("schedule_infos", {}).items()}

        # Create runtime info
        task_wrap_info = TaskWrapRuntimeInfo(
            task_meta=task_meta,
            schedule_infos=schedule_infos,
            inst_status={i: TaskInstStatus.Pending for i in range(task_meta.task_inst_num)},
            inst_data_status={i: TaskInstDataStatus.Pending for i in range(task_meta.task_inst_num)},
            task_submit_time=time.time(),
            task_start_time=0,
            task_end_time=0,
        )

        # Store task info
        self.tasks[task_meta.task_id] = task_wrap_info

        # Dispatch task to appropriate workers
        success = await self._dispatch_task(task_meta.task_id)

        if success:
            return {"status": "success", "task_id": task_meta.task_id}
        else:
            return {"status": "error", "error": "Failed to dispatch task"}

    async def _dispatch_task(self, task_id: str) -> bool:
        """Dispatch task to appropriate workers"""
        if task_id not in self.tasks:
            return False

        task_info = self.tasks[task_id]

        # Group instances by node for more efficient distribution
        node_instances = {}
        for inst_id, schedule_info in task_info.schedule_infos.items():
            gpu_id = schedule_info.gpu_id

            # Find which node contains this GPU
            target_node_id = None
            for node_id, node_status in self.gpu_status.items():
                if gpu_id in node_status:
                    target_node_id = node_id
                    break

            if target_node_id is None:
                self.logger.error(f"Cannot find node for GPU {gpu_id} for task {task_id}")
                return False

            if target_node_id not in node_instances:
                node_instances[target_node_id] = []

            node_instances[target_node_id].append({"inst_id": inst_id, "gpu_id": gpu_id})

        # Publish task info to each affected node
        for node_id, instances in node_instances.items():
            # Check if worker is available
            if node_id not in self.workers or self.workers[node_id]["status"] == "disconnected":
                self.logger.error(f"Worker {node_id} is not available for task {task_id}")
                return False

            # Prepare message
            message = {
                "task_id": task_id,
                "task_meta": task_info.task_meta.__dict__,
                "instances": instances,
                "total_instances": task_info.task_meta.task_inst_num,
                "dispatch_time": time.time(),
            }

            # Publish task to the node
            topic = f"task_{node_id}"
            await self.messenger.publish("task_pub", topic, message)
            self.logger.info(f"Dispatched task {task_id} to worker {node_id} with {len(instances)} instances")

            # Update GPU task queues
            for instance in instances:
                gpu_id = instance["gpu_id"]
                if gpu_id in self.gpu_status[node_id]:
                    self.gpu_status[node_id][gpu_id].append(task_id)

        # Update task status
        task_info.task_meta.task_status = TaskStatus.Pending

        return True

    async def _get_task_status(self, request: dict[str, Any]) -> dict[str, Any]:
        """Get status of a specific task"""
        task_id = request.get("task_id")

        if task_id not in self.tasks:
            return {"status": "error", "error": f"Task {task_id} not found"}

        task_info = self.tasks[task_id]
        return {
            "status": "success",
            "task_id": task_id,
            "task_status": task_info.task_meta.task_status,
            "inst_status": {str(k): v.value for k, v in task_info.inst_status.items()},
            "inst_data_status": {str(k): v.value for k, v in task_info.inst_data_status.items()},
            "submit_time": task_info.task_submit_time,
            "start_time": task_info.task_start_time,
            "end_time": task_info.task_end_time,
        }

    async def _get_cluster_status(self) -> dict[str, Any]:
        """Get status of the entire cluster"""
        workers_status = {}
        for node_id, worker_info in self.workers.items():
            workers_status[node_id] = {
                "status": worker_info["status"],
                "last_heartbeat": worker_info["last_heartbeat"],
                "executors": {},
            }

            for gpu_id, executor_info in worker_info["executors"].items():
                workers_status[node_id]["executors"][gpu_id] = {
                    "status": executor_info["status"],
                    "current_task": executor_info["current_task"],
                    "queue_size": len(executor_info["task_queue"]),
                }

        tasks_status = {}
        for task_id, task_info in self.tasks.items():
            tasks_status[task_id] = {
                "status": task_info.task_meta.task_status.value,
                "name": task_info.task_meta.task_name,
                "instances": task_info.task_meta.task_inst_num,
                "submit_time": task_info.task_submit_time,
            }

        return {
            "status": "success",
            "cluster_id": self.cluster.cluster_id,
            "cluster_name": self.cluster.cluster_name,
            "workers": workers_status,
            "tasks": tasks_status,
        }

    async def handle_message(self, message: dict[str, Any]):
        """Handle incoming messages"""
        message_type = message.get("type")

        if message_type == "worker_status":
            await self._handle_worker_status(message)
        elif message_type == "executor_status":
            await self._handle_executor_status(message)
        elif message_type == "task_status":
            await self._handle_task_status(message)
        else:
            self.logger.warning(f"Unknown message type: {message_type}")

    async def _coordinate_distributed_training(self, task_id: str):
        """协调分布式训练任务启动"""
        if task_id not in self.tasks:
            return False

        task_info = self.tasks[task_id]

        # 检查所有实例是否准备就绪
        all_ready = all(status == TaskInstStatus.Ready for status in task_info.inst_status.values())
        if not all_ready:
            return False

        # 确定主节点（通常是实例 0）
        master_inst = None
        master_node_id = None
        master_gpu_id = None

        for inst_id, schedule_info in task_info.schedule_infos.items():
            if inst_id == 0:  # 假设实例 0 是主节点
                master_gpu_id = schedule_info.gpu_id
                # 查找包含此 GPU 的节点
                for node_id, node_status in self.gpu_status.items():
                    if master_gpu_id in node_status:
                        master_node_id = node_id
                        break

        if not master_node_id or not master_gpu_id:
            self.logger.error(f"Cannot find master node for task {task_id}")
            return False

        # 获取主节点的 IP 地址
        master_ip = None
        if master_node_id in self.workers:
            master_ip = self.workers[master_node_id].get("ip_address")

        if not master_ip:
            self.logger.error(f"Cannot get IP address for master node {master_node_id}")
            return False

        # 更新主节点信息到任务中
        task_info.master_node = {"node_id": master_node_id, "gpu_id": master_gpu_id, "ip_address": master_ip}

        # 通知所有相关的 Worker 开始训练
        # 分组按节点发送
        node_instances = {}
        for inst_id, schedule_info in task_info.schedule_infos.items():
            gpu_id = schedule_info.gpu_id

            # 找到包含此 GPU 的节点
            target_node_id = None
            for node_id, node_status in self.gpu_status.items():
                if gpu_id in node_status:
                    target_node_id = node_id
                    break

            if target_node_id is None:
                continue

            if target_node_id not in node_instances:
                node_instances[target_node_id] = []

            node_instances[target_node_id].append(inst_id)

        # 向每个节点发送启动命令
        for node_id, instances in node_instances.items():
            message = {
                "action": "start_training",
                "task_id": task_id,
                "instances": instances,
                "master_node": task_info.master_node,
            }

            topic = f"control_{node_id}"
            await self.messenger.publish("task_pub", topic, message)

        return True

    async def _dispatch_task_to_worker(self, worker_id: str, task_id: str, task_info: dict, instances: list):
        """分发任务到 Worker"""
        # 获取 Worker 信息
        worker_info = self.workers.get(worker_id)
        if not worker_info:
            self.logger.error(f"Worker {worker_id} not found")
            return False

        # 创建客户端（如果不存在）
        if worker_id not in self.clients:
            self.clients[worker_id] = MasterClient(f"{self.component_id}_to_{worker_id}")

        # 获取 Worker 地址
        worker_host = worker_info.get("ip_address", "localhost")
        worker_port = worker_info.get("request_port", 6557)

        # 发送任务分配请求
        response = await self.clients[worker_id].assign_task(
            worker_host, worker_port, task_id, task_info, instances, task_info.task_meta.task_inst_num
        )

        if response and response.get("status") == "success":
            self.logger.info(f"Successfully assigned task {task_id} to worker {worker_id}")
            return True
        else:
            self.logger.error(f"Failed to assign task {task_id} to worker {worker_id}: {response}")
            return False

    async def _publish_cluster_update(self):
        """发布集群状态更新"""
        update = self._get_cluster_status()
        await self.notifier.publish_cluster_update(update)

    async def _publish_task_update(self, task_id: str):
        """发布任务状态更新"""
        if task_id not in self.tasks:
            return

        task_info = self.tasks[task_id]
        update = {
            "task_id": task_id,
            "status": task_info.task_meta.task_status.value,
            "instances": {i: status.value for i, status in task_info.inst_status.items()},
        }

        await self.notifier.publish_task_update(task_id, update)
