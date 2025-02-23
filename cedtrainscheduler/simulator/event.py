import heapq

from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo


class EventType:
    TaskSubmit = "TaskSubmit"
    DataArrival = "DataArrival"
    TaskFinish = "TaskFinish"


class EventBase:
    def __init__(self, time: float):
        self.time = time

    def __str__(self):
        return f"{self.event_type} at {self.time}"


class EventTaskSubmit(EventBase):
    def __init__(self, time: float, task: TaskWrapRuntimeInfo):
        super().__init__(time)
        self.task = task


class EventDataArrival(EventBase):
    def __init__(self, time: float, task: TaskWrapRuntimeInfo, inst_id: int):
        super().__init__(time)
        self.task = task
        self.inst_id = inst_id


class EventTaskFinish(EventBase):
    def __init__(self, time: float, task: TaskWrapRuntimeInfo):
        super().__init__(time)
        self.task = task


class EventLoopManager:
    def __init__(self):
        self.event_queue = []

    def add_event(self, event: EventBase):
        # 使用事件的时间作为优先级
        heapq.heappush(self.event_queue, (event.time, event))

    def get_next_event(self) -> EventBase:
        # 获取并移除最早的事件
        if self.event_queue:
            return heapq.heappop(self.event_queue)
        return None

    def has_events(self) -> bool:
        # 检查队列中是否还有事件
        return len(self.event_queue) > 0
