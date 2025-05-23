import os

from cedtrainscheduler.runtime.workload.workload import WORKLOAD_INFOS
from cedtrainscheduler.runtime.workload.workload import WorkloadType

LOG_BASE_DIR = "~/logs/train_logs"


class ScriptGenerator:
    @staticmethod
    def generate_python_script(
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
            workload_info = WORKLOAD_INFOS.get(WorkloadType.DEFAULT.value)

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
            f">> {log_file} 2>&1"
        )

        return cmd

    @staticmethod
    def generate_ix_docker_script(
        gpu_rank: int,
        task_id: str,
        task_name: str,
        world_size: int,
        inst_rank: int,
        master_addr: str,
        master_port: int,
        plan_runtime: int,
        data_transfer_time: int,
        python_path: str = "/usr/local/bin/python3.10",
    ) -> str:
        workload_info = WORKLOAD_INFOS.get(task_name.lower())
        if not workload_info:
            raise ValueError(f"Workload info not found for task name: {task_name}")

        log_file = ScriptGenerator.build_log_file_path(task_id, gpu_rank, inst_rank)
        log_dir = os.path.dirname(log_file)
        script_dir = os.path.dirname(workload_info.script_file_path)
        model_dir = os.path.dirname(workload_info.model_file_path)
        dataset_dir = os.path.dirname(workload_info.dataset_dir_path)

        # Build the Docker run command
        docker_cmd = (
            f'docker run --rm --shm-size="32g" '
            f"-v /dev:/dev -v /usr/src/:/usr/src -v /data1:/data1 "
            f"-v /lib/modules/:/lib/modules "
            f"-v {script_dir}:{script_dir} "
            f"-v {model_dir}:{model_dir} "
            f"-v {dataset_dir}:{dataset_dir} "
            f"-v {log_dir}:{log_dir} "
            f"--privileged=true --cap-add=ALL --pid=host --net=host "
            f"-e MASTER_ADDR={master_addr} "
            f"-e MASTER_PORT={master_port} "
            f"-e WORLD_SIZE={world_size} "
            f"-e CUDA_VISIBLE_DEVICES={gpu_rank} "
            f"--name=task_{task_id}_gpu{gpu_rank}_inst{inst_rank} "
            f"corex:3.2.1 "
            f"bash -c '"
            f"{python_path} {workload_info.script_file_path} "
            f"--rank={inst_rank} "
            f"--model_file_path={workload_info.model_file_path} "
            f"--dataset_dir_path={workload_info.dataset_dir_path} "
            f"--runtime={plan_runtime} "
            f"--data_transfer_time={data_transfer_time} "
            f">> {log_file} 2>&1'"
        )

        return docker_cmd

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

    @staticmethod
    def get_task_log_file(task_id: str, gpu_rank: int, inst_rank: int) -> str:
        base_dir = os.path.expanduser(LOG_BASE_DIR)
        log_dir = os.path.join(base_dir, task_id)
        log_file = os.path.join(log_dir, f"gpu{gpu_rank}_rank{inst_rank}.log")
        return log_file


# print(ScriptGenerator.generate_python_script(
#     gpu_rank=4,
#     task_id="test",
#     task_name="resnet50",
#     world_size=1,
#     inst_rank=0,
#     master_addr="127.0.0.1",
#     master_port=12345,
#     plan_runtime=100,
#     data_transfer_time=10,
#     python_path="python",
# ))


# print(ScriptGenerator.generate_ix_docker_script(
#     gpu_rank=4,
#     task_id="test",
#     task_name="resnet50",
#     world_size=1,
#     inst_rank=0,
#     master_addr="127.0.0.1",
#     master_port=12345,
#     plan_runtime=100,
#     data_transfer_time=10,
#     python_path="python",
# ))
