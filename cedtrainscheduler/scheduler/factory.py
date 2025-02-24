from enum import auto
from enum import Enum


class SchedulerType(Enum):
    FCFS = auto()
    CED = auto()


class SchedulerFactory:
    @staticmethod
    def create_scheduler(scheduler_name: str):
        # 延迟导入以避免循环依赖
        from cedtrainscheduler.scheduler.ced_scheduler import CEDScheduler
        from cedtrainscheduler.scheduler.fcfs_scheduler import FCFScheduler

        scheduler_map = {"fcfs": FCFScheduler, "ced": CEDScheduler}

        if scheduler_name.lower() not in scheduler_map:
            raise ValueError(f"Unknown scheduler type: {scheduler_name}")

        return scheduler_map[scheduler_name.lower()]()
