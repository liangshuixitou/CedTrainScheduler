from cedtrainscheduler.runtime.workload.workload import WORKLOAD_INFOS
from cedtrainscheduler.runtime.workload.workload import WorkloadType


class ScriptGenerator:
    @staticmethod
    def generate_script(
        gpu_rank: int,
        task_name: str,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: str,
        python_path: str = "python",
    ) -> str:
        workload_info = WORKLOAD_INFOS.get(task_name)
        if not workload_info:
            raise ValueError(f"Workload info not found for task name: {task_name}")

        script_file_path = workload_info.script_file_path

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

print(ScriptGenerator.generate_script(world_size=2, inst_rank=0, gpu_rank=0, task_name=WorkloadType.RESNET50,
                                      master_addr="127.0.0.1", master_port="29500"))
print(ScriptGenerator.generate_script(world_size=2, inst_rank=1, gpu_rank=1, task_name=WorkloadType.RESNET50,
                                      master_addr="127.0.0.1", master_port="29500"))
