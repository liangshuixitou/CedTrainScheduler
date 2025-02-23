from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.task import TaskInst
from cedtrainscheduler.scheduler.types.task import TaskInstStatus
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.config import SimulatorConfig
from cedtrainscheduler.simulator.event import EventDataArrival
from cedtrainscheduler.simulator.event import EventLoopManager
from cedtrainscheduler.simulator.event import EventTaskFinish
from cedtrainscheduler.simulator.event import EventTaskSubmit
from cedtrainscheduler.simulator.event import EventType
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record


class Simulator:
    def __init__(self, config: SimulatorConfig):
        self.cluster_manager = ClusterManager(config.cluster_config_path)
        self.file_system = FileSystem(config.fs_config_path)
        self.scheduler = SchedulerBase(config.scheduler_name)
        self.task_record = Record()
        self.event_loop_manager = EventLoopManager()
        self.current_time = 0

    def handle_task_submit(self, event: EventTaskSubmit):
        self.task_record.log_task_submit(event.task, self.current_time)
        task = event.task
        task_id = task.task_meta.task_id
        task_meta = task.task_meta
        for inst_id, inst_schedule_info in task_meta.schedule_infos.items():
            if self.cluster_manager.gpu_task_queue[inst_schedule_info.schedule_gpu_id].empty():
                self.handle_task_ready(task, inst_id)
            self.cluster_manager.gpu_task_queue[inst_schedule_info.schedule_gpu_id].put(
                TaskInst(task_id, inst_id, TaskInstStatus.Pending)
            )

    def handle_task_ready(self, task: TaskWrapRuntimeInfo, inst_id: int):
        self.task_record.log_task_inst_ready(task, inst_id)
        if self.task_record.check_task_inst_ready(task):
            self.event_loop_manager.add_event(
                EventDataArrival(
                    self.current_time
                    + self.file_system.get_data_arival_time(
                        task,
                        self.cluster_manager.gpu_node_map[
                            task.task_meta.schedule_infos[inst_id].schedule_gpu_id
                        ].node_id,
                    ),
                    task,
                    inst_id,
                )
            )

    def handle_data_arival(self, event: EventDataArrival):
        self.task_record.log_task_inst_data_arival(event.task, event.inst_id)
        if self.task_record.check_task_inst_data_arival(event.task):
            self.event_loop_manager.add_event(
                EventTaskFinish(
                    self.current_time + event.task.task_meta.task_runtime[event.task.task_meta.task_gpu_type],
                    event.task,
                )
            )

    def handle_task_finish(self, event: EventTaskFinish):
        self.task_record.log_task_finish(event.task)
        for inst_id in event.task.task_meta.schedule_infos.keys():
            next_task_inst = self.cluster_manager.gpu_task_queue[
                event.task.task_meta.schedule_infos[inst_id].schedule_gpu_id
            ].run_next_task()
            if next_task_inst:
                self.handle_task_ready(
                    self.task_record.get_task_record(next_task_inst.task_id), next_task_inst.inst_id
                )

    def simulation(self):
        while True:
            tasks = self.scheduler.schedule(
                self.cluster_manager.clusters.copy(),
                self.cluster_manager.gpu_task_queue.copy(),
                self.task_record.task_record.copy(),
            )
            for task in tasks:
                self.event_loop_manager.add_event(EventTaskSubmit(self.current_time, task))

            while self.event_loop_manager.has_events():
                event = self.event_loop_manager.get_next_event()
                if event is None:
                    break
                if event.event_type == EventType.TaskSubmit:
                    self.handle_task_submit(event)
                    break
                if event.event_type == EventType.DataArrival:
                    self.handle_data_arrival(event)
                    break
                if event.event_type == EventType.TaskFinish:
                    self.handle_task_finish(event)
                    break
