from typing import dict, list, Optional, Any, Union
import asyncio
import time
import json
import os
import subprocess
import signal
import logging
from collections import deque
import shlex
import socket

from cedtrainscheduler.runtime.components.base import BaseComponent
from cedtrainscheduler.runtime.communication.messenger import Messenger
from cedtrainscheduler.runtime.types.task import TaskInstStatus, TaskStatus, TaskInstDataStatus
from cedtrainscheduler.runtime.types.cluster import GPU


class Executor(BaseComponent):
    """Executor component for GPU-level task execution"""

    def __init__(self, executor_id: str, gpu: GPU, config: dict[str, Any]):
        super().__init__(executor_id)
        self.gpu = gpu
        self.config = config

        # Communication
        self.messenger = Messenger(executor_id, "executor")

        # Worker connection info
        self.worker_host = config.get("worker_host", "localhost")
        self.worker_task_port = config.get("worker_task_port", 6555)
        self.worker_status_port = config.get("worker_status_port", 6556)

        # State
        self.task_queue = deque()  # Queue of pending tasks
        self.current_task = None  # Currently running task
        self.task_process = None  # Process handle for current task
        self.task_instances = {}  # task_id -> {inst_id, task_meta, ...}

        # Environment
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu.gpu_rank)

        # Working directory
        self.work_dir = os.path.expanduser(f"~/.cedtrain/executor/{self.gpu.gpu_id}")
        os.makedirs(self.work_dir, exist_ok=True)

        # Worker 引用，初始为 None
        self.worker = None

    async def start(self):
        """启动 Executor"""
        self._running = True
        self._event_loop = asyncio.get_event_loop()

        # 不再需要设置网络通信
        # 只需初始化工作目录等
        os.makedirs(self.work_dir, exist_ok=True)

        # 启动主循环
        await self._run()

    async def _run(self):
        """Executor 主循环"""
        # 主循环检查任务队列和当前任务状态
        while self._running:
            # 检查是否需要启动新任务
            if not self.current_task and self.task_queue:
                await self._start_next_task()

            # 检查当前任务状态
            if self.current_task:
                await self._check_task_status()

            await asyncio.sleep(2)  # 每2秒检查一次

    async def assign_task(self, task_id, inst_id, task_meta, total_instances, gpu_assignments):
        """分配任务实例给 Executor（由 Worker 直接调用）"""
        self.logger.info(f"Received instance {inst_id} of task {task_id}")

        # 存储任务实例信息
        task_inst = {
            "task_id": task_id,
            "inst_id": inst_id,
            "task_meta": task_meta,
            "total_instances": total_instances,
            "gpu_assignments": gpu_assignments,
            "status": "pending",
            "data_status": "pending",
            "received_time": time.time(),
        }

        # 添加到实例字典
        if task_id not in self.task_instances:
            self.task_instances[task_id] = {}
        self.task_instances[task_id][inst_id] = task_inst

        # 添加到任务队列
        self.task_queue.append(task_inst)

        # 如果当前没有运行任务且这是队列首部的任务，尝试准备它
        if not self.current_task and len(self.task_queue) == 1:
            await self._prepare_task_data(task_id, inst_id)

        return True

    async def notify_task_ready(self, task_id):
        """通知任务可以启动（由 Worker 直接调用）"""
        if task_id not in self.task_instances:
            return False

        # 检查当前任务
        if self.current_task and self.current_task["task_id"] == task_id:
            await self._start_task(self.current_task)
            return True

        # 检查队列中的任务
        for task in self.task_queue:
            if task["task_id"] == task_id:
                task["can_start"] = True
                return True

        return False

    async def _prepare_task_data(self, task_id, inst_id):
        """准备任务数据"""
        if task_id not in self.task_instances or inst_id not in self.task_instances[task_id]:
            return False

        task_inst = self.task_instances[task_id][inst_id]

        # 更新状态
        task_inst["status"] = "preparing"
        task_inst["data_status"] = "preparing"

        # 在实际应用中，这里会准备训练数据、检查模型文件等
        # 这里仅做模拟
        await asyncio.sleep(2)  # 模拟数据准备

        # 更新状态
        task_inst["status"] = "ready"
        task_inst["data_status"] = "finished"

        # 通知 Worker
        worker = self._get_worker_reference()  # 获取 Worker 引用
        if worker:
            await worker.notify_task_ready(task_id, inst_id, self.gpu.gpu_id)

        return True

    def _get_worker_reference(self):
        """获取 Worker 引用"""
        return self.worker

    async def _start_next_task(self):
        """启动下一个任务"""
        if not self.task_queue:
            return False

        # 获取队列首部任务
        task = self.task_queue.popleft()
        self.current_task = task

        task_id = task["task_id"]
        inst_id = task["inst_id"]

        self.logger.info(f"Starting task {task_id} instance {inst_id}")

        # 检查是否可以启动
        if task.get("can_start", False):
            await self._start_task(task)
        else:
            self.logger.info(f"Task {task_id} instance {inst_id} is ready but waiting for other instances")

        return True

    async def _start_task(self, task):
        """实际启动任务"""
        task_id = task["task_id"]
        inst_id = task["inst_id"]
        task_meta = task["task_meta"]
        gpu_assignments = task["gpu_assignments"]

        # 记录启动时间
        task["start_time"] = time.time()

        # 构建命令和环境变量
        cmd, env = self._build_distributed_command(task_id, task_meta, inst_id, gpu_assignments)

        if not cmd:
            self.logger.error(f"Failed to build command for task {task_id}")
            await self._finish_current_task(error=True)
            return False

        # 记录目录和日志
        logs_dir = os.path.join(self.work_dir, f"task_{task_id}")
        os.makedirs(logs_dir, exist_ok=True)

        stdout_file = open(os.path.join(logs_dir, f"inst_{inst_id}_stdout.log"), "w")
        stderr_file = open(os.path.join(logs_dir, f"inst_{inst_id}_stderr.log"), "w")

        # 启动进程
        try:
            self.task_process = subprocess.Popen(
                shlex.split(cmd), stdout=stdout_file, stderr=stderr_file, env=env, cwd=self.work_dir
            )
            self.logger.info(f"Task {task_id} instance {inst_id} started with PID {self.task_process.pid}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start task {task_id}: {e}")
            await self._finish_current_task(error=True)
            return False

    def _build_distributed_command(
        self, task_id: str, task_meta: dict[str, Any], inst_id: int, gpu_assignments: dict[str, str]
    ) -> (str, dict[str, str]):
        """Build command for distributed training task"""
        try:
            # In a real implementation, this would look up the workload script and configuration
            # For this example, we'll use a placeholder command

            # Get parameters from task_meta
            task_name = task_meta.get("task_name", "")

            # This would be retrieved from a workload registry in a real implementation
            script_path = f"/path/to/workloads/{task_name}/train.py"

            # Configure DDP parameters
            world_size = len(gpu_assignments)
            rank = inst_id

            # In a real implementation, we would need to collect IP addresses for all nodes
            # For this example, we'll assume all GPUs are on the same node (localhost)
            master_addr = task_meta.get("master_addr", "localhost")

            # 如果这是第 0 个实例，使用自己的 IP 作为 master_addr
            if inst_id == 0:
                try:
                    hostname = socket.gethostname()
                    ip_addr = socket.gethostbyname(hostname)
                    # 将此 IP 地址通知给所有其他节点（通过某种方式）
                    # 可以通过 Worker->Master->其他 Worker 的路径
                except:
                    ip_addr = "localhost"
                    self.logger.warning("Could not determine IP address, using localhost")
                master_addr = ip_addr

            # 设置环境变量
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu.gpu_rank)
            env["WORLD_SIZE"] = str(world_size)
            env["RANK"] = str(rank)
            env["LOCAL_RANK"] = "0"  # Assuming 1 GPU per process
            env["MASTER_ADDR"] = master_addr
            env["MASTER_PORT"] = str(master_port)

            # Build command
            cmd = f"python {script_path} --task_id {task_id} --world_size {world_size} --rank {rank}"

            return cmd, env
        except Exception as e:
            self.logger.error(f"Error building distributed command: {e}")
            return None, None

    async def _check_task_status(self):
        """Check the status of the current task"""
        if not self.current_task or not self.task_process:
            return

        # Poll the process
        returncode = self.task_process.poll()

        # If returncode is not None, the process has finished
        if returncode is not None:
            if returncode == 0:
                self.logger.info(
                    f"Task {self.current_task['task_id']} instance {self.current_task['inst_id']} completed successfully"
                )
                await self._finish_current_task()
            else:
                self.logger.error(
                    f"Task {self.current_task['task_id']} instance {self.current_task['inst_id']} failed with code {returncode}"
                )
                await self._finish_current_task(error=True)

    async def _finish_current_task(self, error: bool = False):
        """Finish the current task and clean up"""
        if not self.current_task:
            return

        task_id = self.current_task["task_id"]
        inst_id = self.current_task["inst_id"]

        # Update status to Finished
        await self._update_task_status(
            task_id, inst_id, TaskInstStatus.Finished.value, TaskInstDataStatus.Finished.value
        )

        # Clean up process
        if self.task_process:
            # Make sure process is terminated
            try:
                if self.task_process.poll() is None:
                    self.task_process.terminate()
                    try:
                        self.task_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.task_process.kill()
            except Exception as e:
                self.logger.error(f"Error terminating process: {e}")

            self.task_process = None

        # Calculate runtime
        end_time = time.time()
        runtime = end_time - self.current_task["start_time"]

        self.logger.info(f"Task {task_id} instance {inst_id} finished in {runtime:.2f} seconds")

        # Clear current task
        self.current_task = None

        # Start next task if available
        if self.task_queue:
            await self._start_next_task()

    async def handle_message(self, message: dict[str, Any]):
        """Handle incoming messages"""
        message_type = message.get("type")

        if message_type == "task_assignment":
            await self._handle_task_assignment(message)
        elif message_type == "task_ready":
            await self._handle_task_ready(message)
        else:
            self.logger.warning(f"Unknown message type: {message_type}")

    def set_worker(self, worker):
        """设置 Worker 引用"""
        self.worker = worker
