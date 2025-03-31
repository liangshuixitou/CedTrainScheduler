import os
from cedtrainscheduler.runtime.workload.workload import WORKLOAD_INFOS

LOG_BASE_DIR = "~/logs/train_logs"


class ScriptGenerator:
    @staticmethod
    def generate_script(
        gpu_rank: int,
        task_name: str,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: int,
        python_path: str = "python",
    ) -> str:
        workload_info = WORKLOAD_INFOS.get(task_name)
        if not workload_info:
            raise ValueError(f"Workload info not found for task name: {task_name}")

        script_file_path = workload_info.script_file_path

        # 定义日志文件路径
        base_dir = os.path.expanduser(LOG_BASE_DIR)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        log_dir = os.path.join(base_dir, task_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f"gpu{gpu_rank}_rank{inst_rank}.log")

        # 构建命令组件
        env_vars = [
            f"export MASTER_ADDR={master_addr}",
            f"export MASTER_PORT={master_port}",
            f"export WORLD_SIZE={world_size}",
        ]

        # 构建训练命令
        train_cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_rank}",
            python_path,
            script_file_path,
            f"--rank={inst_rank}",
            f"--model_file_path={workload_info.model_file_path}",
            f"--dataset_dir_path={workload_info.dataset_dir_path}",
            f">> {log_file} 2>&1",  # 将标准输出和错误输出重定向到日志文件
        ]

        # 组合所有命令
        cmd_parts = [
            "#!/bin/bash",
            "# 分布式训练环境配置",
            *env_vars,
            "",  # 空行分隔
            " ".join(train_cmd),
        ]

        return "\n".join(cmd_parts)
