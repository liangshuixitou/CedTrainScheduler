from cedtrainscheduler.runtime.workload.workload import WORKLOAD_INFOS


class ScriptGenerator:
    @staticmethod
    def generate_script(
        gpu_rank: int,
        task_name: str,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: str,
    ) -> str:
        workload_info = WORKLOAD_INFOS.get(task_name)
        if not workload_info:
            raise ValueError(f"Workload info not found for task name: {task_name}")

        script_file_path = workload_info.script_file_path

        # 构建分布式训练命令
        cmd = f"""#!/bin/bash
# 分布式训练环境配置
export MASTER_ADDR={master_addr}
export MASTER_PORT={master_port}
export WORLD_SIZE={world_size}

# 执行训练脚本，指定GPU和进程排名
CUDA_VISIBLE_DEVICES={gpu_rank} python {script_file_path} --rank={inst_rank}"""

        return cmd
