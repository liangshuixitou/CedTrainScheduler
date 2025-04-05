import os

from cedtrainscheduler.runtime.workload.workload import WORKLOAD_INFOS

LOG_BASE_DIR = "~/logs/train_logs"


class ScriptGenerator:
    @staticmethod
    def generate_script(
        gpu_rank: int,
        task_id: str,
        task_name: str,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: int,
        plan_runtime: int,
        data_transfer_time: int,
        python_path: str = "python",
    ) -> str:
        workload_info = WORKLOAD_INFOS.get(task_name.lower())
        if not workload_info:
            raise ValueError(f"Workload info not found for task name: {task_name}")

        script_file_path = workload_info.script_file_path

        # 定义日志文件路径
        log_file = ScriptGenerator.build_log_file_path(task_id, gpu_rank, inst_rank)

        cmd = (
            f"export MASTER_ADDR={master_addr}; "
            f"export MASTER_PORT={master_port}; "
            f"export WORLD_SIZE={world_size}; "
            f"CUDA_VISIBLE_DEVICES={gpu_rank} "
            f"{python_path} {script_file_path} "
            f"--rank={inst_rank} "
            f"--model_file_path={workload_info.model_file_path} "
            f"--dataset_dir_path={workload_info.dataset_dir_path} "
            f"--runtime={plan_runtime} "
            f"--data_transfer_time={data_transfer_time} "
            f"2>&1 | tee -a {log_file}"
        )

        return cmd

    @staticmethod
    def build_log_file_path(task_id: str, gpu_rank: int, inst_rank: int) -> str:
        base_dir = os.path.expanduser(LOG_BASE_DIR)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        log_dir = os.path.join(base_dir, task_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, f"gpu{gpu_rank}_rank{inst_rank}.log")
        return log_file
