import asyncio

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType
from cedtrainscheduler.runtime.manager.manager import Manager
from cedtrainscheduler.runtime.types.args import ManagerArgs


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start the Manager service")
    parser.add_argument("--id", default="manager", help="Manager component ID")
    parser.add_argument("--ip", default="127.0.0.1", help="Manager IP address")
    parser.add_argument("--port", type=int, default=5001, help="Manager port")
    parser.add_argument("--scheduler-name", default="scheduler", help="Scheduler name")
    parser.add_argument("--cluster-name", default="cluster", help="Cluster name")

    args = parser.parse_args()

    manager = Manager(
        ManagerArgs(
            manager_info=ComponentInfo(
                component_id=args.id,
                component_ip=args.ip,
                component_port=args.port,
                component_type=ComponentType.MANAGER,
            ),
            scheduler_name=args.scheduler_name,
        )
    )

    await manager.run()


if __name__ == "__main__":
    asyncio.run(main())
