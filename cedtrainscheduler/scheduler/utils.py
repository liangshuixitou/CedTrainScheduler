from cedtrainscheduler.runtime.master.master import TaskWrapRuntimeInfo
from cedtrainscheduler.scheduler.types.task import ScheduleInfo
from cedtrainscheduler.scheduler.types.task import TaskMeta


def build_task_wrap_runtime_info(task_meta: TaskMeta, schedule_infos: dict[str, ScheduleInfo]):
    return TaskWrapRuntimeInfo(
        task_meta=task_meta,
        schedule_infos=schedule_infos,
        inst_status={},
        inst_data_status={},
        task_submit_time=task_meta.task_start_time,
        task_start_time=0,
        task_end_time=0,
    )
