from cedtrainscheduler.runtime.manager.constant import INTER_DOMAIN_BANDWIDTH
from cedtrainscheduler.runtime.manager.constant import INTRA_DOMAIN_BANDWIDTH
from cedtrainscheduler.runtime.types.cluster import Cluster as RuntimeCluster
from cedtrainscheduler.runtime.types.cluster import GPU as RuntimeGPU
from cedtrainscheduler.runtime.types.cluster import Node as RuntimeNode
from cedtrainscheduler.runtime.types.task import ScheduleInfo as RuntimeScheduleInfo
from cedtrainscheduler.runtime.types.task import TaskInst as RuntimeTaskInst
from cedtrainscheduler.runtime.types.task import TaskInstStatus as RuntimeTaskInstStatus
from cedtrainscheduler.runtime.types.task import TaskMeta as RuntimeTaskMeta
from cedtrainscheduler.runtime.types.task import TaskStatus as RuntimeTaskStatus
from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo as RuntimeTaskWrapRuntimeInfo
from cedtrainscheduler.scheduler.types.cluster import Cluster as SchedulerCluster
from cedtrainscheduler.scheduler.types.cluster import GPU as SchedulerGPU
from cedtrainscheduler.scheduler.types.cluster import Node as SchedulerNode
from cedtrainscheduler.scheduler.types.task import ScheduleInfo as SchedulerScheduleInfo
from cedtrainscheduler.scheduler.types.task import TaskInst as SchedulerTaskInst
from cedtrainscheduler.scheduler.types.task import TaskInstStatus as SchedulerTaskInstStatus
from cedtrainscheduler.scheduler.types.task import TaskMeta as SchedulerTaskMeta
from cedtrainscheduler.scheduler.types.task import TaskStatus as SchedulerTaskStatus
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo as SchedulerTaskWrapRuntimeInfo


class TypeConverter:
    @staticmethod
    def convert_runtime_gpu_to_scheduler_gpu(runtime_gpu: RuntimeGPU) -> SchedulerGPU:
        return SchedulerGPU(
            gpu_id=runtime_gpu.gpu_id,
            gpu_type=runtime_gpu.gpu_type,
        )

    @staticmethod
    def convert_runtime_node_to_scheduler_node(runtime_node: RuntimeNode) -> SchedulerNode:
        gpus = []
        for _, gpu in runtime_node.gpus.items():
            gpus.append(TypeConverter.convert_runtime_gpu_to_scheduler_gpu(gpu))
        return SchedulerNode(
            node_id=runtime_node.node_id,
            cpu_cores=0,
            memory=0,
            ip_address=runtime_node.ip + ":" + str(runtime_node.port),
            gpus=gpus,
        )

    @staticmethod
    def convert_runtime_cluster_to_scheduler_cluster(runtime_cluster: RuntimeCluster) -> SchedulerCluster:
        nodes = []
        for _, runtime_node in runtime_cluster.nodes.items():
            nodes.append(TypeConverter.convert_runtime_node_to_scheduler_node(runtime_node))
        return SchedulerCluster(
            cluster_id=runtime_cluster.cluster_id,
            cluster_name=runtime_cluster.cluster_name,
            cluster_type=runtime_cluster.cluster_type,
            nodes=nodes,
            inter_domain_bandwidth=INTER_DOMAIN_BANDWIDTH,
            intra_domain_bandwidth=INTRA_DOMAIN_BANDWIDTH,
        )

    @staticmethod
    def convert_runtime_task_inst_status_to_scheduler_task_inst_status(
        runtime_task_inst_status: RuntimeTaskInstStatus,
    ) -> SchedulerTaskInstStatus:
        return runtime_task_inst_status.value

    @staticmethod
    def convert_runtime_task_status_to_scheduler_task_status(
        runtime_task_status: RuntimeTaskStatus,
    ) -> SchedulerTaskStatus:
        return runtime_task_status.value

    @staticmethod
    def convert_runtime_task_inst_to_scheduler_task_inst(runtime_task_inst: RuntimeTaskInst) -> SchedulerTaskInst:
        return SchedulerTaskInst(
            task_id=runtime_task_inst.task_id,
            inst_id=runtime_task_inst.inst_id,
            inst_status=TypeConverter.convert_runtime_task_inst_status_to_scheduler_task_inst_status(
                runtime_task_inst.inst_status
            ),
        )

    @staticmethod
    def convert_runtime_task_meta_to_scheduler_task_meta(runtime_task_meta: RuntimeTaskMeta) -> SchedulerTaskMeta:
        return SchedulerTaskMeta(
            task_id=runtime_task_meta.task_id,
            task_name=runtime_task_meta.task_name,
            task_plan_cpu=runtime_task_meta.task_plan_cpu,
            task_plan_mem=runtime_task_meta.task_plan_mem,
            task_plan_gpu=runtime_task_meta.task_plan_gpu,
            task_inst_num=runtime_task_meta.task_inst_num,
            task_status=TypeConverter.convert_runtime_task_status_to_scheduler_task_status(
                runtime_task_meta.task_status
            ),
            task_start_time=runtime_task_meta.task_start_time,
            task_runtime=runtime_task_meta.task_runtime,
        )

    @staticmethod
    def convert_runtime_schedule_info_to_scheduler_schedule_info(
        runtime_schedule_info: RuntimeScheduleInfo,
    ) -> SchedulerScheduleInfo:
        return SchedulerScheduleInfo(
            inst_id=runtime_schedule_info.inst_id,
            gpu_id=runtime_schedule_info.gpu_id,
        )

    @staticmethod
    def convert_runtime_task_wrap_runtime_info_to_scheduler_task_wrap_runtime_info(
        runtime_task_wrap_runtime_info: RuntimeTaskWrapRuntimeInfo,
    ) -> SchedulerTaskWrapRuntimeInfo:
        return SchedulerTaskWrapRuntimeInfo(
            task_meta=TypeConverter.convert_runtime_task_meta_to_scheduler_task_meta(
                runtime_task_wrap_runtime_info.task_meta
            ),
            schedule_infos=TypeConverter.convert_runtime_schedule_info_to_scheduler_schedule_info(
                runtime_task_wrap_runtime_info.schedule_infos
            ),
            inst_status=TypeConverter.convert_runtime_task_inst_status_to_scheduler_task_inst_status(
                runtime_task_wrap_runtime_info.inst_status
            ),
            inst_data_status=TypeConverter.convert_runtime_task_inst_data_status_to_scheduler_task_inst_data_status(
                runtime_task_wrap_runtime_info.inst_data_status
            ),
            task_submit_time=runtime_task_wrap_runtime_info.task_submit_time,
            task_start_time=runtime_task_wrap_runtime_info.task_start_time,
            task_end_time=runtime_task_wrap_runtime_info.task_end_time,
        )
