from threading import Lock

from cedtrainscheduler.runtime.types.task import TaskInst


class Executor:
    def __init__(self):
        self.task_record: list[TaskInst] = []
        self.task_record_lock = Lock()

    def append_task(self, task: TaskInst):
        with self.task_record_lock:
            self.task_record.append(task)

    def get_task_record(self) -> list[TaskInst]:
        with self.task_record_lock:
            return self.task_record
