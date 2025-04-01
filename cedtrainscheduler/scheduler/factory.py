from enum import auto
from enum import Enum

from cedtrainscheduler.scheduler.scheduler import SchedulerBase


class SchedulerType(Enum):
    FCFS_DATA = auto()
    FCFS = auto()
    SJF_DATA = auto()
    SJF = auto()
    DTSM = auto()
    SCRM = auto()
    FIRST_FIT = auto()
    CHRONUS = auto()
    ALLOX = auto()
    GREEDY = auto()
    IOTCP = auto()


class SchedulerFactory:
    @staticmethod
    def create_scheduler(
        scheduler_name: str,
    ) -> SchedulerBase:
        # 延迟导入以避免循环依赖
        from cedtrainscheduler.scheduler.allox_scheduler import AlloxScheduler
        from cedtrainscheduler.scheduler.chronus_scheduler import ChronusScheduler
        from cedtrainscheduler.scheduler.dtsm_scheduler import DTSMScheduler
        from cedtrainscheduler.scheduler.fcfs_data_scheduler import FCFSDataScheduler
        from cedtrainscheduler.scheduler.fcfs_scheduler import FCFSScheduler
        from cedtrainscheduler.scheduler.first_fit_scheduler import FirstFitScheduler
        from cedtrainscheduler.scheduler.greedy_scheduler import GreedyScheduler
        from cedtrainscheduler.scheduler.iotcp_scheduler import IOTCPScheduler
        from cedtrainscheduler.scheduler.scrm_scheduler import SCRMScheduler
        from cedtrainscheduler.scheduler.sjf_data_scheduler import SJFDataScheduler
        from cedtrainscheduler.scheduler.sjf_scheduler import SJFScheduler

        scheduler_map = {
            "fcfs_data": FCFSDataScheduler,
            "fcfs": FCFSScheduler,
            "sjf_data": SJFDataScheduler,
            "sjf": SJFScheduler,
            "dtsm": DTSMScheduler,
            "sc_rm": SCRMScheduler,
            "first_fit": FirstFitScheduler,
            "chronus": ChronusScheduler,
            "allox": AlloxScheduler,
            "greedy": GreedyScheduler,
            "io_tcp": IOTCPScheduler,
        }

        if scheduler_name.lower() not in scheduler_map:
            print(scheduler_map.keys())
            print(scheduler_name)
            raise ValueError(f"Unknown scheduler type: {scheduler_name}")

        return scheduler_map[scheduler_name.lower()]()
