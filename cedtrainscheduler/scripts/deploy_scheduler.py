#!/usr/bin/env python3
import paramiko
import argparse
import os
from typing import Dict, List
import time


class ClusterDeployer:
    def __init__(self, ssh_config: Dict[str, Dict]):
        """
        ssh_config format:
        {
            "host1": {
                "hostname": "192.168.1.100",
                "username": "user1",
                "password": "pass1",  # 或使用 key_filename
                "key_filename": "~/.ssh/id_rsa"  # 可选
            }
        }
        """
        self.ssh_config = ssh_config
        self.ssh_clients = {}

    def connect(self, host: str) -> paramiko.SSHClient:
        if host in self.ssh_clients:
            return self.ssh_clients[host]

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        config = self.ssh_config[host]
        connect_args = {
            "hostname": config["hostname"],
            "username": config["username"],
        }

        if "password" in config:
            connect_args["password"] = config["password"]
        if "key_filename" in config:
            connect_args["key_filename"] = os.path.expanduser(config["key_filename"])

        client.connect(**connect_args)
        self.ssh_clients[host] = client
        return client

    def execute_command(self, host: str, command: str, background: bool = False) -> None:
        client = self.connect(host)
        if background:
            command = f"nohup {command} > /tmp/{host}_output.log 2>&1 &"
        stdin, stdout, stderr = client.exec_command(command)
        print(f"Executing on {host}:")
        print(f"Command: {command}")
        print("Output:", stdout.read().decode())
        print("Error:", stderr.read().decode())

    def deploy_manager(self, host: str, manager_ip: str, port: int = 5000):
        cmd = (
            f"cd /path/to/project && "  # 请替换为实际项目路径
            f"python cedtrainscheduler/runtime/manager/app.py "
            f"--id manager --ip {manager_ip} --port {port} "
            f"--scheduler-name sjf"
        )
        self.execute_command(host, cmd, background=True)

    def deploy_master(self, host: str, master_ip: str, port: int, manager_ip: str, manager_port: int):
        cmd = (
            f"cd /path/to/project && "  # 请替换为实际项目路径
            f"python cedtrainscheduler/runtime/master/app.py "
            f"--id master --ip {master_ip} --port {port} "
            f"--manager-id manager --manager-ip {manager_ip} "
            f"--manager-port {manager_port} "
            f"--cluster-name local-cluster --cluster-type cloud"
        )
        self.execute_command(host, cmd, background=True)

    def deploy_worker(
        self, host: str, worker_id: str, worker_ip: str, worker_port: int, master_ip: str, master_port: int
    ):
        cmd = (
            f"cd /path/to/project && "  # 请替换为实际项目路径
            f"python cedtrainscheduler/runtime/worker/app.py "
            f"--worker-id {worker_id} --worker-ip {worker_ip} "
            f"--worker-port {worker_port} --master-id master "
            f"--master-ip {master_ip} --master-port {master_port} "
            f"--gpu-type T4"
        )
        self.execute_command(host, cmd, background=True)

    def close_all(self):
        for client in self.ssh_clients.values():
            client.close()


def main():
    # 配置示例
    ssh_config = {
        "manager": {"hostname": "192.168.1.100", "username": "user1", "key_filename": "~/.ssh/id_rsa"},
        "master": {"hostname": "192.168.1.101", "username": "user1", "key_filename": "~/.ssh/id_rsa"},
        "worker1": {"hostname": "192.168.1.102", "username": "user1", "key_filename": "~/.ssh/id_rsa"},
    }

    deployer = ClusterDeployer(ssh_config)

    try:
        # 部署 Manager
        deployer.deploy_manager("manager", "192.168.1.100", 5000)
        time.sleep(5)  # 等待 Manager 启动

        # 部署 Master
        deployer.deploy_master("master", "192.168.1.101", 5001, "192.168.1.100", 5000)
        time.sleep(5)  # 等待 Master 启动

        # 部署 Worker
        deployer.deploy_worker("worker1", "worker1", "192.168.1.102", 5002, "192.168.1.101", 5001)

    finally:
        deployer.close_all()


if __name__ == "__main__":
    main()
