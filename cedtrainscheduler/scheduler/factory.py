from enum import auto
from enum import Enum

from cedtrainscheduler.scheduler.scheduler import SchedulerBase


class SchedulerType(Enum):
    FCFS_DATA = auto()
    FCFS = auto()
    SJF_DATA = auto()
    SJF = auto()
    DTSM = auto()
    TIRESIAS = auto()
    CED = auto()


class SchedulerFactory:
    @staticmethod
    def create_scheduler(
        scheduler_name: str,
    ) -> SchedulerBase:
        # 延迟导入以避免循环依赖
        from cedtrainscheduler.scheduler.ced_scheduler import CedScheduler
        from cedtrainscheduler.scheduler.dtsm_scheduler import DTSMScheduler
        from cedtrainscheduler.scheduler.fcfs_data_scheduler import FCFSDataScheduler
        from cedtrainscheduler.scheduler.fcfs_scheduler import FCFSScheduler
        from cedtrainscheduler.scheduler.sjf_data_scheduler import SJFDataScheduler
        from cedtrainscheduler.scheduler.sjf_scheduler import SJFScheduler
        from cedtrainscheduler.scheduler.tiresias_scheduler import TiresiasScheduler

        scheduler_map = {
            "fcfs-data": FCFSDataScheduler,
            "fcfs": FCFSScheduler,
            "sjf-data": SJFDataScheduler,
            "sjf": SJFScheduler,
            "dtsm": DTSMScheduler,
            "sc-rm": CedScheduler,
            "tiresias": TiresiasScheduler,
        }

        if scheduler_name.lower() not in scheduler_map:
            print(scheduler_map.keys())
            print(scheduler_name)
            raise ValueError(f"Unknown scheduler type: {scheduler_name}")

        return scheduler_map[scheduler_name.lower()]()
