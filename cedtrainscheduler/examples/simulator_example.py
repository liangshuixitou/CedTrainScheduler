import os

from cedtrainscheduler.simulator.config import SimulatorConfig
from cedtrainscheduler.simulator.simulator import Simulator


def main():
    # 配置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))

    config = SimulatorConfig(
        # 集群配置文件路径
        cluster_config_path=os.path.join(base_dir, "cases/cluster/cluster_config.json"),
        # 文件系统配置路径
        fs_config_path=os.path.join(base_dir, "cases/cluster/fs_config.json"),
        # 调度器名称 - 可以是 "fcfs" 或 "ced"
        scheduler_name="fcfs",
        # 任务配置文件路径
        task_config_path=os.path.join(base_dir, "cases/tasks/task_config.json"),
        # 输出结果路径
        output_path=os.path.join(base_dir, "output"),
    )

    # 创建模拟器实例
    simulator = Simulator(config)

    # 运行模拟
    simulator.simulation()


if __name__ == "__main__":
    main()
