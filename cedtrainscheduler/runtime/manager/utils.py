from cedtrainscheduler.runtime.types.cluster import Cluster as RuntimeCluster
from cedtrainscheduler.scheduler.types.cluster import Cluster as SchedulerCluster
from cedtrainscheduler.scheduler.types.cluster import GPU as SchedulerGPU
from cedtrainscheduler.scheduler.types.cluster import Node as SchedulerNode


class TypeConverter:
    @classmethod
    def convert_runtime_cluster_to_scheduler_cluster(cls, runtime_cluster: RuntimeCluster) -> SchedulerCluster:
        nodes = []
        for _, node in runtime_cluster.nodes.items():
            gpus = []
            for _, gpu in node.gpus.items():
                gpus.append(SchedulerGPU(
                    gpu_id=gpu.gpu_id,
                    gpu_type=gpu.gpu_type,
                ))
            nodes.append(SchedulerNode(
                node_id=node.node_id,
                cpu_cores=0,
                memory=0,
                ip_address=node.ip + ":" + str(node.port),

            ))
        return SchedulerCluster(
            cluster_id=runtime_cluster.cluster_id,
            cluster_name=runtime_cluster.cluster_name,
            cluster_type=runtime_cluster.cluster_type,
            nodes=runtime_cluster.nodes,
        )
