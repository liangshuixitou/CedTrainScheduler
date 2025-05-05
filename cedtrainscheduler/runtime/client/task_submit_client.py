import asyncio
import time

import pandas as pd

from cedtrainscheduler.runtime.client.utils import print_task_list
from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType
from cedtrainscheduler.runtime.manager.api_client import TaskManagerClient
from cedtrainscheduler.runtime.types.cluster import GPUType
from cedtrainscheduler.runtime.types.task import TaskMeta
from cedtrainscheduler.runtime.types.task import TaskStatus
from cedtrainscheduler.runtime.utils.metric_util import print_task_metrics


class TaskSubmitClient:
    def __init__(self, manager_info: ComponentInfo):
        self.task_manager_client = TaskManagerClient(manager_info.component_ip, manager_info.component_port)

    async def submit_task(self, task_meta: TaskMeta):
        await self.task_manager_client.submit_task(task_meta)


async def benchmark(task_submit_client: TaskSubmitClient, csv_path: str):
    """Parse CSV file and submit tasks

    Args:
        task_submit_client: TaskSubmitClient instance
        csv_path: Path to the CSV file containing task information
    """
    df = pd.read_csv(csv_path)
    task_list = []

    for _, row in df.iterrows():
        task_meta = TaskMeta(
            task_id=row["job_name"],
            task_name=row["task_name"],
            task_inst_num=int(row["inst_num"]),
            task_plan_cpu=float(row["plan_cpu"]),
            task_plan_mem=float(row["plan_mem"]),
            task_plan_gpu=float(row["plan_gpu"]) / 100,
            task_start_time=time.time(),
            task_status=TaskStatus.Submitted,
            # 创建运行时间字典
            task_runtime={
                GPUType.T4: float(row["runtime_T4"]),
                GPUType.P100: float(row["runtime_P100"]),
                GPUType.V100: float(row["runtime_V100"]),
            },
        )
        task_list.append(task_meta)

    # 每个任务停止3s
    for task_meta in task_list:
        await asyncio.sleep(3)
        await task_submit_client.submit_task(task_meta)
        print(f"Submitted task_id: {task_meta.task_id}")


async def collect_metrics(task_submit_client: TaskSubmitClient):
    while True:
        metrics = await task_submit_client.task_manager_client.metrics()
        print_task_metrics(metrics)
        await asyncio.sleep(5)

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Task Manager Client")
    parser.add_argument("--id", default="manager", help="Manager component ID")
    parser.add_argument("--ip", default="127.0.0.1", help="Manager IP address")
    parser.add_argument("--port", type=int, default=5001, help="Manager port")
    parser.add_argument("--csv-path", type=str, default="task_bench.csv", help="CSV file path")
    parser.add_argument("command", choices=["submit", "list"], help="Command to execute")
    args = parser.parse_args()

    task_submit_client = TaskSubmitClient(
        ComponentInfo(
            component_id=args.id,
            component_ip=args.ip,
            component_port=args.port,
            component_type=ComponentType.MANAGER,
        ),
    )

    if args.command == "submit":
        await benchmark(task_submit_client, args.csv_path)
        await collect_metrics(task_submit_client)
    elif args.command == "list":
        task_list = await task_submit_client.task_manager_client.task_list()
        print_task_list(task_list)



if __name__ == "__main__":
    asyncio.run(main())
