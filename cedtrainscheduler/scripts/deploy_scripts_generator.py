from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType

PROJECT_PATH = "/root/project/CedTrainScheduler"
CONDA_ENV_NAME = "cedtrainscheduler"
EXECUTOR_PYTHON_PATH = "/root/anaconda3/envs/BI100/bin/python"

class ComponentGenerator:
    @staticmethod
    def generate_manager_command(component_info: ComponentInfo, scheduler_name: str, fs_config_path: str) -> str:
        return (
            f"conda activate {CONDA_ENV_NAME} && "
            f"cd {PROJECT_PATH} && "
            f"python cedtrainscheduler/runtime/manager/app.py "
            f"--id {component_info.component_id} "
            f"--ip {component_info.component_ip} "
            f"--port {component_info.component_port} "
            f"--scheduler-name {scheduler_name} "
            f"--fs-config-path {fs_config_path} "
        )

    @staticmethod
    def generate_master_command(
        worker_component_info: ComponentInfo,
        manager_component_info: ComponentInfo,
        cluster_name: str,
        cluster_type: str,
    ) -> str:
        return (
            f"conda activate {CONDA_ENV_NAME} && "
            f"cd {PROJECT_PATH} && "
            f"python cedtrainscheduler/runtime/master/app.py "
            f"--id {worker_component_info.component_id} "
            f"--ip {worker_component_info.component_ip} "
            f"--port {worker_component_info.component_port} "
            f"--manager-id {manager_component_info.component_id} "
            f"--manager-ip {manager_component_info.component_ip} "
            f"--manager-port {manager_component_info.component_port} "
            f"--cluster-name {cluster_name} "
            f"--cluster-type {cluster_type} "
        )

    @staticmethod
    def generate_worker_command(
        worker_component_info: ComponentInfo,
        master_component_info: ComponentInfo,
        gpu_type: str,
        executor_python_path: str | None = None,
        gpu_ids: list[int] | None = None,
        sim_gpu_num: int | None = None,
    ) -> str:
        cmd = (
            f"conda activate {CONDA_ENV_NAME} && "
            f"cd {PROJECT_PATH} && "
            f"python cedtrainscheduler/runtime/worker/app.py "
            f"--worker-id {worker_component_info.component_id} "
            f"--worker-ip {worker_component_info.component_ip} "
            f"--worker-port {worker_component_info.component_port} "
            f"--master-id {master_component_info.component_id} "
            f"--master-ip {master_component_info.component_ip} "
            f"--master-port {master_component_info.component_port} "
            f"--gpu-type {gpu_type} "
        )

        if executor_python_path is not None:
            cmd += f"--executor-python-path {executor_python_path} "
        if gpu_ids is not None:
            cmd += f"--gpu-ids {gpu_ids} "
        if sim_gpu_num is not None:
            cmd += f"--sim-gpu-num {sim_gpu_num} "

        return cmd

    @staticmethod
    def generate_task_submit_client_command(
        task_submit_client_component_info: ComponentInfo,
        csv_path: str,
    ) -> str:
        return (
            f"conda activate {CONDA_ENV_NAME} && "
            f"cd {PROJECT_PATH} && "
            f"python cedtrainscheduler/runtime/client/task_submit_client.py "
            f"--id {task_submit_client_component_info.component_id} "
            f"--ip {task_submit_client_component_info.component_ip} "
            f"--port {task_submit_client_component_info.component_port} "
            f"--csv-path {csv_path} "
        )


class WorkerConfig:
    def __init__(
        self,
        component_info: ComponentInfo,
        gpu_type: str,
        executor_python_path: str | None = None,
        gpu_ids: list[int] | None = None,
        sim_gpu_num: int | None = None,
    ):
        self.component_info = component_info
        self.gpu_type = gpu_type
        self.executor_python_path = executor_python_path
        self.gpu_ids = gpu_ids
        self.sim_gpu_num = sim_gpu_num

    def generate_worker_command(self, master_component_info: ComponentInfo) -> str:
        return ComponentGenerator.generate_worker_command(
            self.component_info,
            master_component_info,
            self.gpu_type,
            self.executor_python_path,
            self.gpu_ids,
            self.sim_gpu_num,
        )

class MasterConfig:
    def __init__(
        self,
        component_info: ComponentInfo,
        cluster_name: str,
        cluster_type: str,
        worker_configs: dict[str, WorkerConfig],
    ):
        self.component_info = component_info
        self.cluster_name = cluster_name
        self.cluster_type = cluster_type
        self.worker_configs = worker_configs

    def generate_master_command(self, manager_component_info: ComponentInfo) -> str:
        return ComponentGenerator.generate_master_command(
            self.component_info, manager_component_info, self.cluster_name, self.cluster_type
        )


class ManagerConfig:
    def __init__(
        self,
        component_info: ComponentInfo,
        scheduler_name: str,
        fs_config_path: str,
        master_configs: dict[str, MasterConfig],
    ):
        self.component_info = component_info
        self.scheduler_name = scheduler_name
        self.fs_config_path = fs_config_path
        self.master_configs = master_configs

    def generate_manager_command(self) -> str:
        return ComponentGenerator.generate_manager_command(self.component_info, self.scheduler_name, self.fs_config_path)


class DeploymentConfig:
    def __init__(
        self,
        manager_config: ManagerConfig,
    ):
        self.manager_config = manager_config

    def print_deployment_command(self) -> None:
        manager_command = self.manager_config.generate_manager_command()
        print(f"\n*** manager {self.manager_config.component_info.component_id} command ***\n")
        print(manager_command)
        print(f"\n*** manager {self.manager_config.component_info.component_id} command ***\n")

        for master_id, master_config in self.manager_config.master_configs.items():
            master_command = master_config.generate_master_command(self.manager_config.component_info)
            print(f"\n*** master {master_id} command ***\n")
            print(master_command)
            print(f"\n*** master {master_id} command ***\n")

            for worker_id, worker_config in master_config.worker_configs.items():
                worker_command = worker_config.generate_worker_command(master_config.component_info)
                print(f"\n*** worker {worker_id} command ***\n")
                print(worker_command)
                print(f"\n*** worker {worker_id} command ***\n")


class TaskSubmitClientConfig:
    def __init__(self, component_info: ComponentInfo, csv_path: str):
        self.component_info = component_info
        self.csv_path = csv_path

    def generate_task_submit_client_command(self) -> str:
        return ComponentGenerator.generate_task_submit_client_command(self.component_info, self.csv_path)


# node1_ip = "36.103.199.97"
# node2_ip = "36.103.199.216"
# node3_ip = "36.103.199.200"

node1_ip = "10.31.12.19"
node2_ip = "10.31.12.20"
node3_ip = "10.31.12.33"

runtime_config = ManagerConfig(
    component_info=ComponentInfo(
        component_type=ComponentType.MANAGER, component_id="manager", component_ip=node1_ip, component_port=5000
    ),
    scheduler_name="fcfs",
    fs_config_path=f"{PROJECT_PATH}/cedtrainscheduler/runtime/manager/config/multi_node_fs_config.json",
    master_configs={
        "master-cloud": MasterConfig(
            component_info=ComponentInfo(
                component_type=ComponentType.MASTER,
                component_id="master-cloud",
                component_ip=node1_ip,
                component_port=5001,
            ),
            cluster_name="master-cloud",
            cluster_type="cloud",
            worker_configs={
                "cloud-worker-1": WorkerConfig(
                    component_info=ComponentInfo(
                        component_type=ComponentType.WORKER,
                        component_id="cloud-worker-1",
                        component_ip=node1_ip,
                        component_port=5002,
                    ),
                    gpu_type="V100",
                    executor_python_path=PYTHON_PATH,
                    gpu_ids="4,5,6,7",
                ),
            },
        ),
        "master-edge": MasterConfig(
            component_info=ComponentInfo(
                component_type=ComponentType.MASTER,
                component_id="master-edge",
                component_ip=node2_ip,
                component_port=5001,
            ),
            cluster_name="master-edge",
            cluster_type="edge",
            worker_configs={
                "edge-worker-1": WorkerConfig(
                    component_info=ComponentInfo(
                        component_type=ComponentType.WORKER,
                        component_id="edge-worker-1",
                        component_ip=node2_ip,
                        component_port=5002,
                    ),
                    gpu_type="P100",
                    executor_python_path=PYTHON_PATH,
                    gpu_ids="6,7",
                ),
            },
        ),
        "master-terminal": MasterConfig(
            component_info=ComponentInfo(
                component_type=ComponentType.MASTER,
                component_id="master-terminal",
                component_ip=node3_ip,
                component_port=5001,
            ),
            cluster_name="master-terminal",
            cluster_type="terminal",
            worker_configs={
                "terminal-worker-1": WorkerConfig(
                    component_info=ComponentInfo(
                        component_type=ComponentType.WORKER,
                        component_id="terminal-worker-1",
                        component_ip=node3_ip,
                        component_port=5002,
                    ),
                    executor_python_path=PYTHON_PATH,
                    gpu_ids="6,7",
                    gpu_type="T4",
                ),
            },
        ),
    },
)

task_submit_client_config = TaskSubmitClientConfig(
    component_info=ComponentInfo(
        component_type=ComponentType.MANAGER,
        component_id="manager",
        component_ip=node1_ip,
        component_port=5000,
    ),
    csv_path=f"{PROJECT_PATH}/cedtrainscheduler/cases/task/case_micro_40_tasks.csv",
)


def main():
    deployment_config = DeploymentConfig(runtime_config)
    deployment_config.print_deployment_command()

    # micro_deployment_config = DeploymentConfig(micro_runtime_config)
    # micro_deployment_config.print_deployment_command()

    task_submit_client_command = task_submit_client_config.generate_task_submit_client_command()
    print(task_submit_client_command)


if __name__ == "__main__":
    main()
