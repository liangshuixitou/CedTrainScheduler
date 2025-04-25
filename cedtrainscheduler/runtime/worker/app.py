import asyncio

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType
from cedtrainscheduler.runtime.types.args import WorkerArgs
from cedtrainscheduler.runtime.worker.worker import Worker


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Start the Worker service")
    parser.add_argument("--worker-id", default="worker", help="Worker component ID")
    parser.add_argument("--worker-ip", default="127.0.0.1", help="Worker IP address")
    parser.add_argument("--worker-port", type=int, default=5001, help="Worker port")
    parser.add_argument("--master-id", default="master", help="Master component ID")
    parser.add_argument("--master-ip", default="127.0.0.1", help="Master IP address")
    parser.add_argument("--master-port", type=int, default=5000, help="Master port")
    parser.add_argument("--gpu-type", default="NVIDIA", help="GPU type")
    parser.add_argument("--executor-python-path", default="", help="Executor python path")
    parser.add_argument("--gpu-ids", default="", help="GPU ids")
    parser.add_argument("--sim-gpu-num", default="", help="Simulator GPU number")

    args = parser.parse_args()

    gpu_ids = args.gpu_ids.split(",")
    sim_gpu_num = int(args.sim_gpu_num)

    worker = Worker(
        WorkerArgs(
            worker_info=ComponentInfo(
                component_id=args.worker_id,
                component_ip=args.worker_ip,
                component_port=args.worker_port,
                component_type=ComponentType.WORKER,
            ),
            master_info=ComponentInfo(
                component_id=args.master_id,
                component_ip=args.master_ip,
                component_port=args.master_port,
                component_type=ComponentType.MASTER,
            ),
            gpu_type=args.gpu_type,
            gpu_ids=gpu_ids,
            sim_gpu_num=sim_gpu_num,
        )
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
