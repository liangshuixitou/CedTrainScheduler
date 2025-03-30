import os

from cedtrainscheduler.simulator.config import SimulatorConfig
from cedtrainscheduler.simulator.simulator import Simulator
from cedtrainscheduler.simulator.types import Metrics


def run_simulation(scheduler_name, fs_config_path, jobs_count, task_smaple_type):
    """运行模拟并返回结果"""
    base_dir = os.getcwd()

    config = SimulatorConfig(
        cluster_config_path=os.path.join(base_dir, "cedtrainscheduler/cases/cluster/cluster_config.json"),
        fs_config_path=os.path.join(base_dir, "cedtrainscheduler/cases/cluster/"+fs_config_path),
        scheduler_name=scheduler_name,
        task_config_path=os.path.join(base_dir, f"cedtrainscheduler/cases/task/case_{task_smaple_type}_{jobs_count}_tasks.csv"),
        output_path=os.path.join(base_dir, "examples/outputs"),
    )

    simulator = Simulator(config)
    results = simulator.simulation()
    return results


# 运行不同调度器的模拟
schedulers = ["k8s-data", "k8s", "sjf-data", "sjf", "ced"]
# schedulers = ["greedy", "ced"]
jobs_count_list = [1000, 2000]
results_dict: dict[int, dict[str, Metrics]] = {}

for jobs_count in jobs_count_list:
    results_dict[jobs_count] = {}
    for scheduler in schedulers:
        results = run_simulation(scheduler, fs_config_path='fs_config.json', jobs_count=jobs_count, task_smaple_type="random")
        results_dict[jobs_count][scheduler] = results
        print(results)
