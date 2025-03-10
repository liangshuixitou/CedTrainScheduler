from enum import auto
from enum import Enum

from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record


class SchedulerType(Enum):
    FCFS_DATA = auto()
    FCFS = auto()
    SJF_DATA = auto()
    SJF = auto()
    CED = auto()


class SchedulerFactory:
    @staticmethod
    def create_scheduler(
        scheduler_name: str,
        config_path: str,
        cluster_manager: ClusterManager,
        task_record: Record,
        file_system: FileSystem,
    ) -> SchedulerBase:
        # 延迟导入以避免循环依赖
        from cedtrainscheduler.scheduler.ced_scheduler import CEDScheduler
        from cedtrainscheduler.scheduler.fcfs_data_scheduler import FCFSDataScheduler
        from cedtrainscheduler.scheduler.fcfs_scheduler import FCFSScheduler
        from cedtrainscheduler.scheduler.sjf_data_scheduler import SJFDataScheduler
        from cedtrainscheduler.scheduler.sjf_scheduler import SJFScheduler

        scheduler_map = {
            "k8s-data": FCFSDataScheduler,
            "k8s": FCFSScheduler,
            "sjf-data": SJFDataScheduler,
            "sjf": SJFScheduler,
            "ced": CEDScheduler,
        }

        if scheduler_name.lower() not in scheduler_map:
            print(scheduler_map.keys())
            print(scheduler_name)
            raise ValueError(f"Unknown scheduler type: {scheduler_name}")

        return scheduler_map[scheduler_name.lower()](config_path, cluster_manager, task_record, file_system)
