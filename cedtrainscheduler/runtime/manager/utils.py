from cedtrainscheduler.runtime.types.cluster import Cluster as RuntimeCluster
from cedtrainscheduler.runtime.types.task import TaskInst as RuntimeTaskInst
from cedtrainscheduler.runtime.types.task import TaskInstStatus as RuntimeTaskInstStatus
from cedtrainscheduler.runtime.types.task import TaskMeta as RuntimeTaskMeta
from cedtrainscheduler.scheduler.types.cluster import Cluster as SchedulerCluster
from cedtrainscheduler.scheduler.types.cluster import GPU as SchedulerGPU
from cedtrainscheduler.scheduler.types.cluster import Node as SchedulerNode
from cedtrainscheduler.scheduler.types.task import TaskInst as SchedulerTaskInst
from cedtrainscheduler.scheduler.types.task import TaskInstStatus as SchedulerTaskInstStatus
from cedtrainscheduler.scheduler.types.task import TaskMeta as SchedulerTaskMeta


class TypeConverter:
    @staticmethod
    def convert_runtime_cluster_to_scheduler_cluster(cls, runtime_cluster: RuntimeCluster) -> SchedulerCluster:
        nodes = []
        for _, node in runtime_cluster.nodes.items():
            gpus = []
            for _, gpu in node.gpus.items():
                gpus.append(
                    SchedulerGPU(
                        gpu_id=gpu.gpu_id,
                        gpu_type=gpu.gpu_type,
                    )
                )
            nodes.append(
                SchedulerNode(
                    node_id=node.node_id,
                    cpu_cores=0,
                    memory=0,
                    ip_address=node.ip + ":" + str(node.port),
                )
            )
        return SchedulerCluster(
            cluster_id=runtime_cluster.cluster_id,
            cluster_name=runtime_cluster.cluster_name,
            cluster_type=runtime_cluster.cluster_type,
            nodes=runtime_cluster.nodes,
        )

    @staticmethod
    def convert_runtime_task_inst_status_to_scheduler_task_inst_status(
        runtime_task_inst_status: RuntimeTaskInstStatus,
    ) -> SchedulerTaskInstStatus:
        return runtime_task_inst_status.value

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
            task_inst_num=runtime_task_meta.task_inst_num,
            task_plan_cpu=runtime_task_meta.task_plan_cpu,
        )
