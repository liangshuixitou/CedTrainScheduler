from cedtrainscheduler.scheduler.types.task import TaskInstDataStatus
from cedtrainscheduler.scheduler.types.task import TaskInstStatus
from cedtrainscheduler.scheduler.types.task import TaskStatus
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo


class Record:
    def __init__(self):
        self.task_record = dict[str, TaskWrapRuntimeInfo]

    def get_task_record(self, task_id: str) -> TaskWrapRuntimeInfo:
        return self.task_record[task_id]

    def log_task_submit(self, task: TaskWrapRuntimeInfo, current_time: float):
        task.task_status = TaskStatus.Pending
        task.inst_status = {
            inst_id: TaskInstStatus.Pending for inst_id in task.schedule_infos.keys()
        }
        task.inst_data_status = {
            inst_id: TaskInstDataStatus.Pending for inst_id in task.schedule_infos.keys()
        }
        self.task_record[task.task_meta.task_id] = task

    def log_task_inst_ready(self, task: TaskWrapRuntimeInfo, inst_id: int):
        self.task_record[task.task_meta.task_id].inst_status[inst_id] = TaskInstStatus.Ready

    def log_task_inst_data_arival(self, task: TaskWrapRuntimeInfo, inst_id: int):
        self.task_record[task.task_meta.task_id].inst_data_status[inst_id] = TaskInstDataStatus.Finished

    def check_task_inst_ready(self, task: TaskWrapRuntimeInfo) -> bool:
        for inst_id in task.task_meta.schedule_infos.keys():
            if self.task_record[task.task_meta.task_id].inst_status[inst_id] != TaskInstStatus.Ready:
                return False
        return True

    def check_task_inst_data_arival(self, task: TaskWrapRuntimeInfo) -> bool:
        for inst_id in task.task_meta.schedule_infos.keys():
            if self.task_record[task.task_meta.task_id].inst_data_status[inst_id] != TaskInstDataStatus.Finished:
                return False
        return True

    def log_task_finish(self, task: TaskWrapRuntimeInfo, current_time: float):
        task.task_status = TaskStatus.Finished
        task.task_end_time = current_time
        for inst_id in task.task_meta.schedule_infos.keys():
            task.inst_status[inst_id] = TaskInstStatus.Finished
        self.task_record[task.task_meta.task_id] = task
