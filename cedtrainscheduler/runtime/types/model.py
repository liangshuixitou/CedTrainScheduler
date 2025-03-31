from pydantic import BaseModel

from cedtrainscheduler.runtime.components import ComponentInfo
from cedtrainscheduler.runtime.components import ComponentType
from cedtrainscheduler.runtime.types.cluster import Cluster
from cedtrainscheduler.runtime.types.cluster import ClusterType
from cedtrainscheduler.runtime.types.cluster import GPU
from cedtrainscheduler.runtime.types.cluster import GPUType
from cedtrainscheduler.runtime.types.cluster import Node
from cedtrainscheduler.runtime.types.task import ScheduleInfo
from cedtrainscheduler.runtime.types.task import TaskInst
from cedtrainscheduler.runtime.types.task import TaskInstDataStatus
from cedtrainscheduler.runtime.types.task import TaskInstStatus
from cedtrainscheduler.runtime.types.task import TaskMeta
from cedtrainscheduler.runtime.types.task import TaskStatus
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo


# GPU 相关模型
class GPUModel(BaseModel):
    gpu_id: str
    gpu_type: GPUType
    gpu_rank: int
    node_id: str

    def to_gpu(self) -> GPU:
        return GPU(gpu_id=self.gpu_id, gpu_type=self.gpu_type, gpu_rank=self.gpu_rank, node_id=self.node_id)

    @classmethod
    def from_gpu(cls, gpu: GPU) -> "GPUModel":
        return cls(gpu_id=gpu.gpu_id, gpu_type=gpu.gpu_type, gpu_rank=gpu.gpu_rank, node_id=gpu.node_id)


# Node 相关模型
class NodeModel(BaseModel):
    node_id: str
    ip: str
    port: int
    cluster_id: str
    gpus: dict[str, GPUModel]

    def to_node(self) -> Node:
        return Node(
            node_id=self.node_id,
            ip=self.ip,
            port=self.port,
            cluster_id=self.cluster_id,
            gpus={gpu_id: gpu.to_gpu() for gpu_id, gpu in self.gpus.items()},
        )

    @classmethod
    def from_node(cls, node: Node) -> "NodeModel":
        return cls(
            node_id=node.node_id,
            ip=node.ip,
            port=node.port,
            cluster_id=node.cluster_id,
            gpus={gpu_id: GPUModel.from_gpu(gpu) for gpu_id, gpu in node.gpus.items()},
        )


class ClusterModel(BaseModel):
    cluster_id: str
    cluster_name: str
    cluster_type: ClusterType
    nodes: dict[str, NodeModel]

    def to_cluster(self) -> Cluster:
        return Cluster(
            cluster_id=self.cluster_id,
            cluster_name=self.cluster_name,
            cluster_type=self.cluster_type,
            nodes={node_id: node.to_node() for node_id, node in self.nodes.items()},
        )

    @classmethod
    def from_cluster(cls, cluster: Cluster) -> "ClusterModel":
        return cls(
            cluster_id=cluster.cluster_id,
            cluster_name=cluster.cluster_name,
            cluster_type=cluster.cluster_type,
            nodes={node_id: NodeModel.from_node(node) for node_id, node in cluster.nodes.items()},
        )


# TaskInst 相关模型
class TaskInstModel(BaseModel):
    task_id: str
    inst_id: int
    inst_status: TaskInstStatus

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


class ComponentInfoModel(BaseModel):
    component_type: ComponentType
    component_id: str
    component_ip: str
    component_port: int

    def to_component_info(self) -> ComponentInfo:
        return ComponentInfo(
            component_type=self.component_type,
            component_id=self.component_id,
            component_ip=self.component_ip,
            component_port=self.component_port,
        )

    @classmethod
    def from_component_info(cls, info: ComponentInfo) -> "ComponentInfoModel":
        return cls(
            component_type=info.component_type,
            component_id=info.component_id,
            component_ip=info.component_ip,
            component_port=info.component_port,
        )


# manager model
class ManagerTaskSubmitModel(BaseModel):
    task: TaskMetaModel


class ManagerMasterRegisterModel(BaseModel):
    cluster: ClusterModel
    task_infos: dict[str, TaskWrapRuntimeInfoModel]
    master_info: ComponentInfoModel
    task_queue_map: dict[str, list[TaskInstModel]]


# master model
class MasterTaskSubmitModel(BaseModel):
    task: TaskWrapRuntimeInfoModel


class MasterWorkerRegisterModel(BaseModel):
    node: NodeModel
    tasks: list[TaskInstModel]
    task_queue_map: dict[str, list[TaskInstModel]]


# worker model
class WorkerTaskInstSubmitModel(BaseModel):
    task_inst: TaskInstModel
    gpu_id: str


class WorkerTaskInstStartModel(BaseModel):
    task_inst: TaskInstModel
    gpu_id: str
    task_name: str
    world_size: int
    inst_rank: int
    master_addr: str
    master_port: int
