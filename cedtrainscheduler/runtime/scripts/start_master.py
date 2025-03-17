#!/usr/bin/env python3
"""启动 Master 组件的脚本"""

import argparse
import asyncio
import json
import logging
import os
import sys

from cedtrainscheduler.runtime.components.master import Master
from cedtrainscheduler.runtime.configs.network_config import get_ip_address, create_cluster_config
from cedtrainscheduler.runtime.types.cluster import Cluster, ClusterType

async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Start a CED Train Scheduler Master")
    parser.add_argument("--cluster-id", required=True, help="Unique cluster ID")
    parser.add_argument("--cluster-name", required=True, help="Cluster name")
    parser.add_argument("--cluster-type", choices=["cloud", "edge", "terminal"], default="cloud", help="Cluster type")
    parser.add_argument("--config", help="Path to cluster configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    args = parser.parse_args()

    # 设置日志级别
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("master_starter")

    # 获取本机 IP 地址
    ip_address = get_ip_address()
    logger.info(f"Local IP address: {ip_address}")

    # 加载集群配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cluster_config = json.load(f)
    else:
        # 创建默认配置
        cluster_config = create_cluster_config(ip_address)

    # 创建集群对象 (在实际应用中，这应该从配置文件中加载)
    # 这里创建一个示例集群
    cluster = Cluster(
        cluster_id=args.cluster_id,
        cluster_name=args.cluster_name,
        cluster_type=ClusterType(args.cluster_type),
        nodes=[]  # 实际应该从配置中加载
    )

    # 创建 Master 组件
    master = Master(args.cluster_id, cluster, cluster_config["master"])

    # 启动 Master
    logger.info(f"Starting Master for cluster {args.cluster_name} ({args.cluster_id})")
    try:
        await master.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
        await master.stop()
    except Exception as e:
        logger.error(f"Error starting master: {e}")
        await master.stop()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
