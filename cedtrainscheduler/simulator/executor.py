from queue import Queue

from cedtrainscheduler.scheduler.types.task import TaskInst


class GPUExecutor:
    def __init__(self, gpu_id: str):
        self.gpu_id = gpu_id
        self.task_map: dict[str, TaskInst] = {}
        self.pending_queue: Queue[TaskInst] = Queue()
        self.running_task: TaskInst = None

    def empty(self) -> bool:
        return self.pending_queue.empty() and self.running_task is None

    def put(self, task: TaskInst):
        self.task_map[task.task_id] = task
        self.pending_queue.put(task)

    def run_next_task(self) -> TaskInst:
        if self.pending_queue.empty():
            return None
        task = self.pending_queue.get()
        self.running_task = task
        return task
