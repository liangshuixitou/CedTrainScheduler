from cedtrainscheduler.scheduler.factory import SchedulerFactory
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
from cedtrainscheduler.simulator.types import Metrics


class Simulator:
    def __init__(self, config: SimulatorConfig):
        self.cluster_manager = ClusterManager(config.cluster_config_path)
        self.file_system = FileSystem(config.fs_config_path)
        self.scheduler: SchedulerBase = SchedulerFactory.create_scheduler(config.scheduler_name)
        self.scheduler.load_config(config.task_config_path)
        self.task_record = Record()
        self.event_loop_manager = EventLoopManager()
        self.output_path = config.output_path
        self.current_time = 0

    def handle_task_submit(self, event: EventTaskSubmit):
        self.task_record.log_task_submit(event.task, self.current_time)
        task = event.task
        task_id = task.task_meta.task_id
        for inst_id, inst_schedule_info in task.schedule_infos.items():
            if self.cluster_manager.gpu_task_queue[inst_schedule_info.gpu_id].empty():
                self.cluster_manager.gpu_task_queue[inst_schedule_info.gpu_id].put(
                    TaskInst(task_id, inst_id, TaskInstStatus.Pending)
                )
                self.handle_task_inst_ready(task, inst_id)
            else:
                self.cluster_manager.gpu_task_queue[inst_schedule_info.gpu_id].put(
                    TaskInst(task_id, inst_id, TaskInstStatus.Pending)
                )

    def handle_task_inst_ready(self, task: TaskWrapRuntimeInfo, inst_id: int):
        self.task_record.log_task_inst_ready(task, inst_id)
        if self.task_record.check_task_inst_ready(task):
            for inst_id in range(task.task_meta.task_inst_num):
                # 添加所有节点的数据到达事件
                self.event_loop_manager.add_event(
                    EventDataArrival(
                        self.current_time
                        + self.file_system.get_data_arival_time(
                            task,
                            self.cluster_manager.gpu_node_map[task.schedule_infos[inst_id].gpu_id].node_id,
                            self.cluster_manager,
                            self.scheduler.scheduler_name,
                        ),
                        task,
                        inst_id,
                    )
                )

    def handle_data_inst_arival(self, event: EventDataArrival):
        self.task_record.log_task_inst_data_arival(event.task, event.inst_id)
        if self.task_record.check_task_inst_data_arival(event.task):
            self.task_record.log_task_start(self.current_time, event.task)
            for inst_id in range(event.task.task_meta.task_inst_num):
                self.cluster_manager.gpu_task_queue[event.task.schedule_infos[inst_id].gpu_id].run_next_task_inst()
            self.event_loop_manager.add_event(
                EventTaskFinish(
                    self.current_time
                    + event.task.task_meta.task_runtime[
                        self.cluster_manager.gpu_task_queue[event.task.schedule_infos[event.inst_id].gpu_id].gpu_type
                    ],
                    event.task,
                )
            )

    def handle_task_finish(self, event: EventTaskFinish):
        self.task_record.log_task_finish(event.task, self.current_time)
        for inst_id in range(event.task.task_meta.task_inst_num):
            next_task_inst = self.cluster_manager.gpu_task_queue[
                event.task.schedule_infos[inst_id].gpu_id
            ].get_next_task_inst()
            if next_task_inst:
                self.handle_task_inst_ready(
                    self.task_record.get_task_record(next_task_inst.task_id), next_task_inst.inst_id
                )

    def simulation(self) -> Metrics:
        while True:
            # 初始调度
            task, is_finished = self.scheduler.schedule(
                self.current_time,
                self.cluster_manager.clusters.copy(),
                self.cluster_manager.gpu_task_queue.copy(),
                self.file_system.task_data_info.copy(),
                self.task_record.task_record.copy(),
            )
            if task:
                self.event_loop_manager.add_event(EventTaskSubmit(self.current_time, task))

            # 事件处理循环
            while self.event_loop_manager.has_events():
                event = self.event_loop_manager.get_next_event()
                self.current_time = event.time
                if event is None:
                    break
                if event.event_type == EventType.TaskSubmit:
                    self.handle_task_submit(event)
                elif event.event_type == EventType.DataArrival:
                    self.handle_data_inst_arival(event)
                elif event.event_type == EventType.TaskFinish:
                    self.handle_task_finish(event)

                # 在每次事件处理后立即尝试调度
                task, is_finished = self.scheduler.schedule(
                    self.current_time,
                    self.cluster_manager.clusters.copy(),
                    self.cluster_manager.gpu_task_queue.copy(),
                    self.file_system.task_data_info.copy(),
                    self.task_record.task_record.copy(),
                )
                if task:
                    self.event_loop_manager.add_event(EventTaskSubmit(self.current_time, task))

            # 结束条件：没有事件且调度器返回已完成状态
            if not self.event_loop_manager.has_events() and is_finished:
                self.task_record.save_task_result(self.output_path)
                metrics = self.count_metrics(self.task_record.task_record)
                return metrics

    def count_metrics(self, task_record: dict[str, TaskWrapRuntimeInfo]) -> Metrics:
        total_runtime = self.current_time

        avg_queue_time = 0
        for task in task_record.values():
            avg_queue_time += task.task_start_time - task.task_submit_time
        avg_queue_time /= len(task_record)

        avg_running_time = 0
        for task in task_record.values():
            avg_running_time += task.task_end_time - task.task_submit_time
        avg_running_time /= len(task_record)

        avg_execution_time = 0
        for task in task_record.values():
            avg_execution_time += task.task_end_time - task.task_start_time
        avg_execution_time /= len(task_record)

        metrics: Metrics = Metrics(
            scheduler_name=self.scheduler.scheduler_name,
            task_count=len(task_record),
            total_runtime=total_runtime,
            avg_queue_time=avg_queue_time,
            avg_running_time=avg_running_time,
            avg_execution_time=avg_execution_time,
        )
        return metrics
