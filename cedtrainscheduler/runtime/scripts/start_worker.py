#!/usr/bin/env python3
"""启动 Worker 组件的脚本"""

import argparse
import asyncio
import json
import logging
import os
import sys

from cedtrainscheduler.runtime.components.worker import Worker
from cedtrainscheduler.runtime.configs.network_config import get_ip_address, create_cluster_config
from cedtrainscheduler.runtime.types.cluster import Node
from cedtrainscheduler.runtime.utils.gpu_detection import detect_gpus

async def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Start a CED Train Scheduler Worker")
    parser.add_argument("--node-id", required=True, help="Unique node ID")
    parser.add_argument("--cluster-id", required=True, help="Cluster ID")
    parser.add_argument("--master-host", required=True, help="Master host address")
    parser.add_argument("--config", help="Path to node configuration file")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    args = parser.parse_args()

    # 设置日志级别
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("worker_starter")

    # 获取本机 IP 地址
    ip_address = get_ip_address()
    logger.info(f"Local IP address: {ip_address}")

    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            node_config = json.load(f)
    else:
        # 创建默认配置
        cluster_config = create_cluster_config(args.master_host)
        node_config = {
            "worker": cluster_config["worker"],
            "gpus": []  # 实际应自动检测
        }

    # 添加 Master 信息
    node_config["worker"]["master_host"] = args.master_host

    # 自动检测 GPU
    gpus = detect_gpus(args.node_id)
    logger.info(f"Detected {len(gpus)} GPUs")

    # 创建节点
    node = Node(
        node_id=args.node_id,
        cluster_id=args.cluster_id,
        gpus=gpus  # 使用检测到的 GPU
    )

    # 增加通信端口配置
    node_config["worker"]["request_port"] = node_config["worker"].get("request_port", 6557)
    node_config["worker"]["local_pub_port"] = node_config["worker"].get("local_pub_port", 6558)

    # 创建 Worker 组件
    worker = Worker(args.node_id, node, node_config["worker"])

    # 启动 Worker
    logger.info(f"Starting Worker {args.node_id} connecting to master at {args.master_host}")
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
        await worker.stop()
    except Exception as e:
        logger.error(f"Error starting worker: {e}")
        await worker.stop()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
