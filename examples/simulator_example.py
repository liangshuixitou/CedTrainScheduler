import os

from cedtrainscheduler.simulator.config import SimulatorConfig
from cedtrainscheduler.simulator.simulator import Simulator


def run_simulation(scheduler_name, fs_config_path):
    """运行模拟并返回结果"""
    current_dir = os.getcwd()  # 使用 os.getcwd() 获取当前工作目录
    base_dir = os.path.dirname(current_dir)

    config = SimulatorConfig(
        cluster_config_path=os.path.join(base_dir, "cedtrainscheduler/cases/cluster/cluster_config.json"),
        fs_config_path=os.path.join(base_dir, "cedtrainscheduler/cases/cluster/" + fs_config_path),
        scheduler_name=scheduler_name,
        task_config_path=os.path.join(base_dir, "cedtrainscheduler/cases/task/case_5000_tasks.csv"),
        output_path=os.path.join(base_dir, "examples/output"),
    )

    simulator = Simulator(config)
    results = simulator.simulation()
    return results


# 假设 results_dict_light 和 results_dict_heavy 是两个模拟器的结果字典
# 运行不同调度器的模拟
schedulers = ["k8s-data", "k8s", "sjf-data", "sjf", "ced"]
results_dict_light = {}
results_dict_heavy = {}

for scheduler in schedulers:
    # 使用轻量版配置运行模拟
    results_light = run_simulation(scheduler, fs_config_path="fs_config_light.json")
    results_dict_light[scheduler] = results_light
    print(results_light)

    # 使用重版配置运行模拟
    results_heavy = run_simulation(scheduler, fs_config_path="fs_config_heavy.json")
    results_dict_heavy[scheduler] = results_heavy
    print(results_heavy)
