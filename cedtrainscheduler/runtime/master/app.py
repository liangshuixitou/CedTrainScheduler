import asyncio

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType
from cedtrainscheduler.runtime.master.master import Master
from cedtrainscheduler.runtime.types.args import MasterArgs


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start the Master service")
    parser.add_argument("--id", default="master", help="Master component ID")
    parser.add_argument("--ip", default="127.0.0.1", help="Master IP address")
    parser.add_argument("--port", type=int, default=5000, help="Master port")
    parser.add_argument("--manager-id", default="manager", help="Manager component ID")
    parser.add_argument("--manager-ip", default="127.0.0.1", help="Manager IP address")
    parser.add_argument("--manager-port", type=int, default=5001, help="Manager port")
    parser.add_argument("--cluster-name", default="cluster", help="Cluster name")
    parser.add_argument(
        "--cluster-type", default="cloud", choices=["cloud", "edge"], help="Cluster type (cloud or edge)"
    )

    args = parser.parse_args()

    master = Master(
        MasterArgs(
            master_info=ComponentInfo(
                component_id=args.id,
                component_ip=args.ip,
                component_port=args.port,
                component_type=ComponentType.MASTER,
            ),
            manager_info=ComponentInfo(
                component_id=args.manager_id,
                component_ip=args.manager_ip,
                component_port=args.manager_port,
                component_type=ComponentType.MANAGER,
            ),
            cluster_name=args.cluster_name,
            cluster_type=args.cluster_type,
        )
    )

    await master.run()


if __name__ == "__main__":
    asyncio.run(main())
