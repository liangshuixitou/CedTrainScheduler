from dataclasses import dataclass


@dataclass
class SimulatorConfig:
    cluster_config_path: str
    fs_config_path: str
    scheduler_name: str
    task_config_path: str
    output_path: str
