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
    def get_gpus_with_ids(node_id: str, gpu_ids: list[int]) -> dict[int, str]:
        """
        获取当前节点上指定GPU-ID
        Returns:
            dict[int, str]: GPU-ID列表
        """
        gpus = {}
        for gpu_id in gpu_ids:
            gpus[gpu_id] = f"{node_id}-gpu-{gpu_id}"
        return gpus

    @staticmethod
    def get_gpus_with_num(node_id: str, gpu_num: int) -> dict[int, str]:
        """
        获取当前节点上指定数量的GPU-ID
        Returns:
            dict[int, str]: GPU-ID列表
        """
        gpus = {}
        for i in range(gpu_num):
            gpus[i] = f"{node_id}-gpu-{i}"
        return gpus
