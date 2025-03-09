import json
import os
from enum import Enum

from cedtrainscheduler.scheduler.types.task import TaskInstDataStatus
from cedtrainscheduler.scheduler.types.task import TaskInstStatus
from cedtrainscheduler.scheduler.types.task import TaskStatus
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo


class Record:
    def __init__(self):
        self.task_record: dict[str, TaskWrapRuntimeInfo] = {}

    def get_task_record(self, task_id: str) -> TaskWrapRuntimeInfo:
        return self.task_record[task_id]

    def log_task_submit(self, task: TaskWrapRuntimeInfo, current_time: float):
        task.task_meta.task_status = TaskStatus.Pending
        task.inst_status = {inst_id: TaskInstStatus.Pending for inst_id in range(task.task_meta.task_inst_num)}
        task.inst_data_status = {inst_id: TaskInstDataStatus.Pending for inst_id in range(task.task_meta.task_inst_num)}
        self.task_record[task.task_meta.task_id] = task

    def log_task_inst_ready(self, task: TaskWrapRuntimeInfo, inst_id: int):
        self.task_record[task.task_meta.task_id].inst_status[inst_id] = TaskInstStatus.Ready

    def log_task_inst_data_arrival(self, task: TaskWrapRuntimeInfo, inst_id: int):
        self.task_record[task.task_meta.task_id].inst_data_status[inst_id] = TaskInstDataStatus.Finished

    def check_task_inst_ready(self, task: TaskWrapRuntimeInfo) -> bool:
        for inst_id in range(task.task_meta.task_inst_num):
            if self.task_record[task.task_meta.task_id].inst_status[inst_id] != TaskInstStatus.Ready:
                return False
        return True

    def check_task_inst_data_arrival(self, task: TaskWrapRuntimeInfo) -> bool:
        for inst_id in range(task.task_meta.task_inst_num):
            if self.task_record[task.task_meta.task_id].inst_data_status[inst_id] != TaskInstDataStatus.Finished:
                return False
        return True

    def log_task_start(self, current_time: float, task: TaskWrapRuntimeInfo):
        task.task_meta.task_status = TaskStatus.Running
        task.task_start_time = current_time
        for inst_id in range(task.task_meta.task_inst_num):
            task.inst_status[inst_id] = TaskInstStatus.Running
        self.task_record[task.task_meta.task_id] = task

    def log_task_finish(self, task: TaskWrapRuntimeInfo, current_time: float):
        task.task_meta.task_status = TaskStatus.Finished
        task.task_end_time = current_time
        for inst_id in range(task.task_meta.task_inst_num):
            task.inst_status[inst_id] = TaskInstStatus.Finished
        self.task_record[task.task_meta.task_id] = task

    def save_task_result(self, output_path: str, scheduler_name: str):
        def serialize(obj):
            if isinstance(obj, TaskWrapRuntimeInfo):
                return {
                    "task_meta": obj.task_meta.__dict__,
                    "schedule_infos": {k: v.__dict__ for k, v in obj.schedule_infos.items()},
                    "inst_status": {k: v.value for k, v in obj.inst_status.items()},
                    "inst_data_status": {k: v.value for k, v in obj.inst_data_status.items()},
                    "task_submit_time": obj.task_submit_time,
                    "task_start_time": obj.task_start_time,
                    "task_end_time": obj.task_end_time,
                }
            elif isinstance(obj, Enum):
                return obj.value
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            else:
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        # Get total number of tasks
        total_tasks = len(self.task_record)

        # Create filename with scheduler name and task count info
        filename = f"simulation_{scheduler_name}_tasks_{total_tasks}.json"

        # Join directory path with filename
        full_path = os.path.join(output_path, filename)

        with open(full_path, "w") as f:
            json.dump(self.task_record, f, default=serialize, indent=4)
