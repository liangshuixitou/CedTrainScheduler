import torch


class GPUUtil:
    @staticmethod
    def get_gpu_count() -> int:
        """
        获取当前节点可用的GPU数量
        Returns:
            int: GPU数量
        """
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()

    @staticmethod
    def get_gpus(node_id: str) -> list[str]:
        """
        获取当前节点上所有GPU-ID
        Returns:
            List[str]: GPU-ID列表
        """
        gpu_count = GPUUtil.get_gpu_count()
        gpus = []
        for i in range(gpu_count):
            gpus.append(f"{node_id}-gpu-{i}")
        return gpus
