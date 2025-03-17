"""GPU 检测工具"""

import json
import logging
import subprocess
from typing import list
from typing import Optional

from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.cluster import GPUType

logger = logging.getLogger(__name__)


def detect_gpus(node_id: str) -> list[GPU]:
    """检测本地 GPU 设备"""
    gpus = []

    try:
        # 尝试使用 nvidia-smi 工具
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            check=True,
        )

        # 解析输出
        for i, line in enumerate(result.stdout.strip().split("\n")):
            try:
                index, name, uuid, memory = line.split(", ")

                # 尝试确定 GPU 类型
                gpu_type = determine_gpu_type(name)

                gpu = GPU(gpu_id=f"{node_id}_gpu_{index}", gpu_type=gpu_type, gpu_rank=int(index), node_id=node_id)

                gpus.append(gpu)
                logger.info(f"Detected GPU: {name} ({gpu_type.value}) - UUID: {uuid}")
            except Exception as e:
                logger.error(f"Error parsing GPU info for line {i}: {e}")
                continue

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running nvidia-smi: {e}")
    except FileNotFoundError:
        logger.error("nvidia-smi not found - NVIDIA drivers may not be installed")

    return gpus


def determine_gpu_type(name: str) -> GPUType:
    """根据 GPU 名称确定类型"""
    name = name.lower()

    if "v100" in name:
        return GPUType.V100
    elif "p100" in name:
        return GPUType.P100
    elif "t4" in name:
        return GPUType.T4
    else:
        # 默认为 T4
        return GPUType.T4
