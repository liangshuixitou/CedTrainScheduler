import asyncio

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType
from cedtrainscheduler.runtime.manager.api_client import TaskManagerClient
from cedtrainscheduler.runtime.types.cluster import GPUType
from cedtrainscheduler.runtime.types.task import TaskMeta
from cedtrainscheduler.runtime.types.task import TaskStatus


class TaskSubmitClient:
    def __init__(self, manager_info: ComponentInfo):
        self.task_manager_client = TaskManagerClient(manager_info.host, manager_info.port)

    async def submit_task(self, task_meta: TaskMeta):
        await self.task_manager_client.submit_task(task_meta)


async def test_submit_one_task(task_submit_client: TaskSubmitClient):
    task_meta = TaskMeta(
        task_id="task-001",
        task_name="resnet50",
        task_plan_cpu=0,
        task_plan_mem=0,
        task_plan_gpu=1,
        task_status=TaskStatus.Pending,
        task_start_time=0,
        task_runtime={GPUType.T4: 10000},
    )
    await task_submit_client.submit_task(task_meta)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start the Master service")
    parser.add_argument("--id", default="manager", help="Manager component ID")
    parser.add_argument("--ip", default="127.0.0.1", help="Manager IP address")
    parser.add_argument("--port", type=int, default=5001, help="Manager port")

    args = parser.parse_args()

    task_submit_client = TaskSubmitClient(
        ComponentInfo(
            component_id=args.id,
            component_ip=args.ip,
            component_port=args.port,
            component_type=ComponentType.MANAGER,
        ),
    )

    await test_submit_one_task(task_submit_client)


if __name__ == "__main__":
    asyncio.run(main())
