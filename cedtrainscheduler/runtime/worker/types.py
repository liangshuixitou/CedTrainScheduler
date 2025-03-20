from pydantic import BaseModel

from cedtrainscheduler.runtime.types.task import (
    ScheduleInfo,
    TaskInst,
    TaskInstDataStatus,
    TaskInstStatus,
    TaskMeta,
    TaskWrapRuntimeInfo,
)
from cedtrainscheduler.runtime.types.task import TaskStatus


# TaskInst 相关模型
class TaskInstModel(BaseModel):
    task_id: str
    inst_id: int
    inst_status: TaskStatus

    def to_task_inst(self) -> TaskInst:
        return TaskInst(task_id=self.task_id, inst_id=self.inst_id, inst_status=self.inst_status)

    @classmethod
    def from_task_inst(cls, task_inst: TaskInst) -> "TaskInstModel":
        return cls(task_id=task_inst.task_id, inst_id=task_inst.inst_id, inst_status=task_inst.inst_status)


# TaskMeta 相关模型
class TaskMetaModel(BaseModel):
    task_id: str
    task_name: str
    task_inst_num: int
    task_plan_cpu: float
    task_plan_mem: float
    task_plan_gpu: int
    task_status: TaskStatus
    task_start_time: float
    task_runtime: dict[str, float]

    def to_task_meta(self) -> TaskMeta:
        return TaskMeta(
            task_id=self.task_id,
            task_name=self.task_name,
            task_inst_num=self.task_inst_num,
            task_plan_cpu=self.task_plan_cpu,
            task_plan_mem=self.task_plan_mem,
            task_plan_gpu=self.task_plan_gpu,
            task_status=self.task_status,
            task_start_time=self.task_start_time,
            task_runtime=self.task_runtime,
        )

    @classmethod
    def from_task_meta(cls, task_meta: TaskMeta) -> "TaskMetaModel":
        return cls(
            task_id=task_meta.task_id,
            task_name=task_meta.task_name,
            task_inst_num=task_meta.task_inst_num,
            task_plan_cpu=task_meta.task_plan_cpu,
            task_plan_mem=task_meta.task_plan_mem,
            task_plan_gpu=task_meta.task_plan_gpu,
            task_status=task_meta.task_status,
            task_start_time=task_meta.task_start_time,
            task_runtime=task_meta.task_runtime,
        )


# ScheduleInfo 相关模型
class ScheduleInfoModel(BaseModel):
    inst_id: int
    gpu_id: str

    def to_schedule_info(self) -> ScheduleInfo:
        return ScheduleInfo(inst_id=self.inst_id, gpu_id=self.gpu_id)

    @classmethod
    def from_schedule_info(cls, schedule_info: ScheduleInfo) -> "ScheduleInfoModel":
        return cls(inst_id=schedule_info.inst_id, gpu_id=schedule_info.gpu_id)


# TaskWrapRuntimeInfo 相关模型
class TaskWrapRuntimeInfoModel(BaseModel):
    task_meta: TaskMetaModel
    schedule_infos: dict[int, ScheduleInfoModel]
    inst_status: dict[int, TaskInstStatus]
    inst_data_status: dict[int, TaskInstDataStatus]
    task_submit_time: float
    task_start_time: float
    task_end_time: float

    def to_task_wrap_runtime_info(self) -> TaskWrapRuntimeInfo:
        return TaskWrapRuntimeInfo(
            task_meta=self.task_meta.to_task_meta(),
            schedule_infos={inst_id: info.to_schedule_info() for inst_id, info in self.schedule_infos.items()},
            inst_status=self.inst_status,
            inst_data_status=self.inst_data_status,
            task_submit_time=self.task_submit_time,
            task_start_time=self.task_start_time,
            task_end_time=self.task_end_time,
        )

    @classmethod
    def from_task_wrap_runtime_info(cls, info: TaskWrapRuntimeInfo) -> "TaskWrapRuntimeInfoModel":
        return cls(
            task_meta=TaskMetaModel.from_task_meta(info.task_meta),
            schedule_infos={
                inst_id: ScheduleInfoModel.from_schedule_info(sched_info)
                for inst_id, sched_info in info.schedule_infos.items()
            },
            inst_status=info.inst_status,
            inst_data_status=info.inst_data_status,
            task_submit_time=info.task_submit_time,
            task_start_time=info.task_start_time,
            task_end_time=info.task_end_time,
        )


# API 请求模型
class TaskInstSubmitModel(BaseModel):
    task_inst: TaskInstModel
    gpu_id: str


class TaskInstStartModel(BaseModel):
    task_inst: TaskInstModel
    gpu_id: str
    task_record: dict[str, TaskWrapRuntimeInfoModel]
