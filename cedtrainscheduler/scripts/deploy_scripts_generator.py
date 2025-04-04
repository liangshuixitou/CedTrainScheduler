from cedtrainscheduler.runtime.components import ComponentInfo

PROJECT_PATH = "/home/ubuntu/CedTrainScheduler"


class ComponentGenerator:
    @staticmethod
    def generate_manager_command(component_info: ComponentInfo, scheduler_name: str) -> str:
        return (
            f"cd {PROJECT_PATH} &&"
            f"python cedtrainscheduler/runtime/manager/app.py"
            f"--id {component_info.component_id}"
            f"--ip {component_info.component_ip}"
            f"--port {component_info.component_port}"
            f"--scheduler-name {scheduler_name}"
        )

    @staticmethod
    def generate_master_command(
        worker_component_info: ComponentInfo,
        manager_component_info: ComponentInfo,
        cluster_name: str,
        cluster_type: str,
    ) -> str:
        return (
            f"cd {PROJECT_PATH} &&"
            f"python cedtrainscheduler/runtime/master/app.py"
            f"--id {worker_component_info.component_id}"
            f"--ip {worker_component_info.component_ip}"
            f"--port {worker_component_info.component_port}"
            f"--manager-id {manager_component_info.component_id}"
            f"--manager-ip {manager_component_info.component_ip}"
            f"--manager-port {manager_component_info.component_port}"
            f"--cluster-name {cluster_name}"
            f"--cluster-type {cluster_type}"
        )

    @staticmethod
    def generate_worker_command(
        worker_component_info: ComponentInfo,
        master_component_info: ComponentInfo,
        gpu_type: str,
    ) -> str:
        return (
            f"cd {PROJECT_PATH} &&"
            f"python cedtrainscheduler/runtime/worker/app.py"
            f"--worker-id {worker_component_info.component_id}"
            f"--worker-ip {worker_component_info.component_ip}"
            f"--worker-port {worker_component_info.component_port}"
            f"--master-id {master_component_info.component_id}"
            f"--master-ip {master_component_info.component_ip}"
            f"--master-port {master_component_info.component_port}"
            f"--gpu-type {gpu_type}"
        )


class WorkerConfig:
    def __init__(self, component_info: ComponentInfo, gpu_type: str):
        self.component_info = component_info
        self.gpu_type = gpu_type

    def generate_worker_command(self, master_component_info: ComponentInfo) -> str:
        return ComponentGenerator.generate_worker_command(self.component_info, master_component_info, self.gpu_type)


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
    def __init__(self, component_info: ComponentInfo, scheduler_name: str, master_configs: dict[str, MasterConfig]):
        self.component_info = component_info
        self.scheduler_name = scheduler_name
        self.master_configs = master_configs

    def generate_manager_command(self) -> str:
        return ComponentGenerator.generate_manager_command(self.component_info, self.scheduler_name)


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


node1_ip = "192.168.1.101"
node2_ip = "192.168.1.102"
node3_ip = "192.168.1.103"

runtime_config = ManagerConfig(
    component_info=ComponentInfo(component_id="manager", component_ip=node1_ip, component_port=5000),
    scheduler_name="sjf",
    master_configs={
        "master-cloud": MasterConfig(
            component_info=ComponentInfo(component_id="master-cloud", component_ip=node1_ip, component_port=5001),
            cluster_name="master-cloud",
            cluster_type="cloud",
            worker_configs={
                "cloud-worker-1": WorkerConfig(
                    component_info=ComponentInfo(
                        component_id="cloud-worker-1", component_ip=node2_ip, component_port=5002
                    ),
                    gpu_type="V100",
                ),
            },
        ),
        "master-edge": MasterConfig(
            component_info=ComponentInfo(component_id="master-edge", component_ip=node2_ip, component_port=5001),
            cluster_name="master-edge",
            cluster_type="edge",
            worker_configs={
                "edge-worker-1": WorkerConfig(
                    component_info=ComponentInfo(
                        component_id="edge-worker-1", component_ip=node2_ip, component_port=5002
                    ),
                    gpu_type="P100",
                ),
            },
        ),
        "master-terminal": MasterConfig(
            component_info=ComponentInfo(component_id="master-terminal", component_ip=node3_ip, component_port=5001),
            cluster_name="master-terminal",
            cluster_type="terminal",
            worker_configs={
                "terminal-worker-1": WorkerConfig(
                    component_info=ComponentInfo(
                        component_id="terminal-worker-1", component_ip=node3_ip, component_port=5002
                    ),
                    gpu_type="T4",
                ),
            },
        ),
    },
)

node1_ip = "192.168.1.101"

micro_runtime_config = ManagerConfig(
    component_info=ComponentInfo(component_id="manager", component_ip=node1_ip, component_port=5000),
    scheduler_name="sjf",
    master_configs={
        "master-cloud": MasterConfig(
            component_info=ComponentInfo(component_id="master-cloud", component_ip=node1_ip, component_port=5001),
            cluster_name="master-cloud",
            cluster_type="cloud",
            worker_configs={
                "cloud-worker-1": WorkerConfig(
                    component_info=ComponentInfo(
                        component_id="cloud-worker-1", component_ip=node2_ip, component_port=5002
                    ),
                    gpu_type="V100",
                ),
            },
        ),
    },
)


def main():
    # deployment_config = DeploymentConfig(runtime_config)
    # deployment_config.print_deployment_command()

    micro_deployment_config = DeploymentConfig(micro_runtime_config)
    micro_deployment_config.print_deployment_command()


if __name__ == "__main__":
    main()
