import asyncio

import pandas as pd

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType
from cedtrainscheduler.runtime.manager.api_client import TaskManagerClient
from cedtrainscheduler.runtime.types.cluster import GPUType
from cedtrainscheduler.runtime.types.task import TaskMeta
from cedtrainscheduler.runtime.types.task import TaskStatus


class TaskSubmitClient:
    def __init__(self, manager_info: ComponentInfo):
        self.task_manager_client = TaskManagerClient(manager_info.component_ip, manager_info.component_port)

    async def submit_task(self, task_meta: TaskMeta):
        await self.task_manager_client.submit_task(task_meta)


async def test_submit_tasks_from_csv(task_submit_client: TaskSubmitClient, csv_path: str):
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
            task_start_time=0,
            task_status=TaskStatus.Submitted,
            # 创建运行时间字典
            task_runtime={
                GPUType.T4: float(row["runtime_T4"]),
                GPUType.P100: float(row["runtime_P100"]),
                GPUType.V100: float(row["runtime_V100"]),
            },
        )
        task_list.append(task_meta)

    # 每5个任务停止10s
    for i, task_meta in enumerate(task_list):
        if i % 5 == 0:
            await asyncio.sleep(10)
        await task_submit_client.submit_task(task_meta)


async def test_submit_one_task(task_submit_client: TaskSubmitClient):
    task_meta = TaskMeta(
        task_id="task-001",
        task_name="resnet50",
        task_inst_num=1,
        task_plan_cpu=0,
        task_plan_mem=0,
        task_plan_gpu=1,
        task_status=TaskStatus.Pending,
        task_start_time=0,
        task_runtime={GPUType.T4: 10000},
    )
    await task_submit_client.submit_task(task_meta)

    task_meta = TaskMeta(
        task_id="task-002",
        task_name="resnet50",
        task_inst_num=2,
        task_plan_cpu=0,
        task_plan_mem=0,
        task_plan_gpu=1,
        task_status=TaskStatus.Pending,
        task_start_time=0,
        task_runtime={GPUType.T4: 100},
    )
    await task_submit_client.submit_task(task_meta)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start the Master service")
    parser.add_argument("--id", default="manager", help="Manager component ID")
    parser.add_argument("--ip", default="127.0.0.1", help="Manager IP address")
    parser.add_argument("--port", type=int, default=5001, help="Manager port")
    parser.add_argument("--csv_path", type=str, default="task_bench.csv", help="CSV file path")
    args = parser.parse_args()

    task_submit_client = TaskSubmitClient(
        ComponentInfo(
            component_id=args.id,
            component_ip=args.ip,
            component_port=args.port,
            component_type=ComponentType.MANAGER,
        ),
    )

    # await test_submit_one_task(task_submit_client)
    await test_submit_tasks_from_csv(task_submit_client, args.csv_path)


if __name__ == "__main__":
    asyncio.run(main())
