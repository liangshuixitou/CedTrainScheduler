PROJECT_PATH = "/home/ubuntu/CedTrainScheduler"


class DeploymentCommandGenerator:
    def __init__(self, ssh_config: dict[str, dict]):
        self.ssh_config = ssh_config

    def generate_manager_command(self, manager_ip: str, port: int = 5000, scheduler_name: str = "sjf") -> str:
        cmd = (
            f"cd {PROJECT_PATH} && "
            f"python cedtrainscheduler/runtime/manager/app.py "
            f"--id manager --ip {manager_ip} --port {port} "
            f"--scheduler-name {scheduler_name}"
        )
        return cmd

    def generate_master_command(
        self, master_ip: str, port: int, manager_ip: str, manager_port: int, cluster_name: str, cluster_type: str
    ) -> str:
        cmd = (
            f"cd {PROJECT_PATH} && "
            f"python cedtrainscheduler/runtime/master/app.py "
            f"--id master --ip {master_ip} --port {port} "
            f"--manager-id manager --manager-ip {manager_ip} "
            f"--manager-port {manager_port} "
            f"--cluster-name local-cluster --cluster-type cloud"
        )
        return cmd

    def generate_worker_command(
        self, worker_id: str, worker_ip: str, worker_port: int, master_ip: str, master_port: int
    ) -> str:
        cmd = (
            f"cd {PROJECT_PATH} && "
            f"python cedtrainscheduler/runtime/worker/app.py "
            f"--worker-id {worker_id} --worker-ip {worker_ip} "
            f"--worker-port {worker_port} --master-id master "
            f"--master-ip {master_ip} --master-port {master_port} "
            f"--gpu-type T4"
        )
        return cmd


def main():
    # 配置示例
    ssh_config = {
        "manager": {"hostname": "192.168.1.100", "username": "ubuntu"},
        "master": {"hostname": "192.168.1.101", "username": "ubuntu"},
        "worker1": {"hostname": "192.168.1.102", "username": "ubuntu"},
    }

    generator = DeploymentCommandGenerator(ssh_config)

    print(generator.generate_manager_command(manager_ip="36.103.199.112", port=5000, scheduler_name="sjf"))

    print(
        generator.generate_master_command(
            master_ip="36.103.199.112",
            port=5001,
            manager_ip="36.103.199.112",
            manager_port=5000,
            cluster_name="local-cluster",
            cluster_type="cloud",
        )
    )

    print(
        generator.generate_worker_command(
            worker_id="worker1",
            worker_ip="36.103.199.112",
            worker_port=5002,
            master_ip="36.103.199.112",
            master_port=5001,
        )
    )


if __name__ == "__main__":
    main()
