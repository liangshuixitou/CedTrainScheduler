import pandas as pd

from cedtrainscheduler.scheduler.factory import SchedulerFactory
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.cluster import ClusterType
from cedtrainscheduler.scheduler.types.cluster import GPUType
from cedtrainscheduler.scheduler.types.scheduler_context import SchedulerContext
from cedtrainscheduler.scheduler.types.task import TaskInst
from cedtrainscheduler.scheduler.types.task import TaskInstStatus
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskStatus
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.config import SimulatorConfig
from cedtrainscheduler.simulator.event import EventDataArrival
from cedtrainscheduler.simulator.event import EventLoopManager
from cedtrainscheduler.simulator.event import EventTaskFinish
from cedtrainscheduler.simulator.event import EventTaskParse
from cedtrainscheduler.simulator.event import EventTaskSubmit
from cedtrainscheduler.simulator.event import EventType
from cedtrainscheduler.simulator.fs import FileSystem
from cedtrainscheduler.simulator.manager import ClusterManager
from cedtrainscheduler.simulator.record import Record
from cedtrainscheduler.simulator.types import Metrics


class Simulator:
    def __init__(self, config: SimulatorConfig):
        self.cluster_manager = ClusterManager(config.cluster_config_path)
        self.file_system = FileSystem(config.fs_config_path, self.cluster_manager)
        self.task_record = Record()
        self.event_loop_manager = EventLoopManager()

        self.scheduler: SchedulerBase = SchedulerFactory.create_scheduler(
            config.scheduler_name,
        )
        self.current_time = 0
        self.load_task_config(config.task_config_path)

    def load_task_config(self, task_config_path: str):
        df = pd.read_csv(task_config_path)
        task_list = []
        for _, row in df.iterrows():
            task_meta = TaskMeta(
                task_id=row["job_name"],
                task_name=row["task_name"],
                task_inst_num=int(row["inst_num"]),
                task_plan_cpu=float(row["plan_cpu"]),
                task_plan_mem=float(row["plan_mem"]),
                task_plan_gpu=float(row["plan_gpu"]) / 100,
                # task_start_time=float(row["start_time"]),
                task_start_time=float(0),
                task_status=TaskStatus.Submitted,
                # 创建运行时间字典
                task_runtime={
                    GPUType.T4: float(row["runtime_T4"]),
                    GPUType.P100: float(row["runtime_P100"]),
                    GPUType.V100: float(row["runtime_V100"]),
                },
            )
            task_list.append(task_meta)

        for task in task_list:
            self.event_loop_manager.add_event(EventTaskParse(task.task_start_time, task))

    def handle_task_parse(self, event: EventTaskParse):
        self.scheduler.submit_task(
            SchedulerContext(
                current_time=self.current_time,
                cluster_manager=self.cluster_manager,
                task_record=self.task_record.task_record,
                file_system=self.file_system,
                task_queue=self.scheduler.task_queue,
            ),
            event.task,
        )

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
                task_data_info = self.file_system.get_task_data_info(task.task_meta.task_name)
                self.event_loop_manager.add_event(
                    EventDataArrival(
                        self.current_time
                        + self.file_system.get_data_arrival_time(
                            task_data_info,
                            self.cluster_manager.node_cluster_map[
                                self.cluster_manager.gpu_node_map[task.schedule_infos[inst_id].gpu_id].node_id
                            ].cluster_id,
                            self.cluster_manager,
                            self.scheduler.scheduler_name,
                        ),
                        task,
                        inst_id,
                    )
                )

    def handle_data_inst_arival(self, event: EventDataArrival):
        self.task_record.log_task_inst_data_arrival(event.task, event.inst_id)
        if self.task_record.check_task_inst_data_arrival(event.task):
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
            # 事件处理循环
            is_finished = True
            while self.event_loop_manager.has_events():
                event = self.event_loop_manager.get_next_event()
                self.current_time = event.time
                if event is None:
                    break
                if event.event_type == EventType.TaskParse:
                    self.handle_task_parse(event)
                    # 如果还有事件，则不进行调度
                    if self.event_loop_manager.has_events() and not self.scheduler.is_queue_full():
                        continue
                elif event.event_type == EventType.TaskSubmit:
                    self.handle_task_submit(event)
                elif event.event_type == EventType.DataArrival:
                    self.handle_data_inst_arival(event)
                elif event.event_type == EventType.TaskFinish:
                    self.handle_task_finish(event)

                # 在每次事件处理后立即尝试调度
                task, is_finished = self.scheduler.schedule(
                    SchedulerContext(
                        current_time=self.current_time,
                        cluster_manager=self.cluster_manager,
                        task_record=self.task_record.task_record,
                        file_system=self.file_system,
                        task_queue=self.scheduler.task_queue,
                    ),
                )
                if task:
                    self.event_loop_manager.add_event(EventTaskSubmit(self.current_time, task))

            # 结束条件：没有事件且调度器返回已完成状态
            if not self.event_loop_manager.has_events() and is_finished:
                metrics = self.count_metrics(self.task_record.task_record)
                return metrics

    def count_metrics(self, task_record: dict[str, TaskWrapRuntimeInfo]) -> Metrics:
        start_time = min(task.task_start_time for task in task_record.values())
        end_time = max(task.task_end_time for task in task_record.values())
        total_runtime = end_time - start_time

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

        # 统计任务被调度至云、边、端的任务数量
        cloud_count = 0
        edge_count = 0
        terminal_count = 0
        for task in task_record.values():
            inst_id = list(task.schedule_infos.keys())[0]
            gpu_id = task.schedule_infos[inst_id].gpu_id
            node = self.cluster_manager.gpu_node_map[gpu_id]
            cluster_type = self.cluster_manager.node_cluster_map[node.node_id].cluster_type
            if cluster_type == ClusterType.CLOUD:
                cloud_count += 1
            elif cluster_type == ClusterType.EDGE:
                edge_count += 1
            else:
                terminal_count += 1

        metrics: Metrics = Metrics(
            scheduler_name=self.scheduler.scheduler_name,
            task_count=len(task_record),
            total_runtime=total_runtime,
            avg_queue_time=avg_queue_time,
            avg_running_time=avg_running_time,
            avg_execution_time=avg_execution_time,
            cloud_count=cloud_count,
            edge_count=edge_count,
            terminal_count=terminal_count,
        )
        return metrics
