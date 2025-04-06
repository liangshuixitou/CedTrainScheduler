from queue import Queue

from cedtrainscheduler.scheduler.types.cluster import GPUType
from cedtrainscheduler.scheduler.types.task import TaskInst
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo


class GPUExecutor:
    def __init__(self, gpu_id: str, gpu_type: GPUType):
        self.gpu_type = gpu_type
        self.gpu_id = gpu_id
        self.pending_queue: Queue[TaskInst] = Queue()
        self.running_task: TaskInst = None

    def empty(self) -> bool:
        return self.pending_queue.empty() and self.running_task is None

    def put(self, task: TaskInst):
        self.pending_queue.put(task)

    def run_next_task_inst(self):
        if self.pending_queue.empty():
            self.running_task = None
        else:
            task = self.pending_queue.get()
            self.running_task = task

    def get_next_task_inst(self) -> TaskInst:
        self.running_task = None
        if self.pending_queue.empty():
            return None
        task = self.pending_queue.queue[0]
        return task

    def queue_time(self, current_time: float, task_record: dict[str, TaskWrapRuntimeInfo]) -> float:
        if self.running_task:
            execute_time = task_record[self.running_task.task_id].task_meta.task_runtime[self.gpu_type] - (
                current_time - task_record[self.running_task.task_id].task_start_time
            )
            # if task_record[self.running_task.task_id].task_start_time == 0:
            #     print(f"task {self.running_task.task_id} start time is 0")
        else:
            execute_time = 0
        pending_time = 0
        for task in list(self.pending_queue.queue):
            pending_time += task_record[task.task_id].task_meta.task_runtime[self.gpu_type]
        return execute_time + pending_time
