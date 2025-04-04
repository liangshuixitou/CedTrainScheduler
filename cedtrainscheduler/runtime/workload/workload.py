import logging
import os
from enum import Enum

MODEL_DIR = "~/data/models"
DATASET_DIR = "~/data/datasets"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


class WorkloadType(str, Enum):
    RESNET50 = "resnet50"
    RESNET18 = "resnet18"
    MOBILENETV3 = "mobilenetv3"
    MOBILENETV2 = "mobilenetv2"
    EFFICIENTNET = "efficientnet"
    VGG11 = "vgg11"
    DCGAN = "dcgan"
    POINTNET = "pointnet"
    BERT = "bert"
    LSTM = "lstm"
    TRANSFORMER = "transformer"


class WorkloadInfo:
    def __init__(
        self,
        workload_type: WorkloadType,
        model_file_name: str,
        dataset_dir_name: str,
        script_file_path: str,
    ):
        self.workload_type = workload_type
        self.model_file_path = os.path.expanduser(f"{MODEL_DIR}/{model_file_name}")
        self.dataset_dir_path = os.path.expanduser(f"{DATASET_DIR}/{dataset_dir_name}")
        self.script_file_path = os.path.expanduser(f"{SCRIPT_DIR}/{script_file_path}")


BASE_WORKLOAD_INFO = WorkloadInfo(
    workload_type=WorkloadType.RESNET50,
    model_file_name=f"{WorkloadType.RESNET50.value}.pth",
    dataset_dir_name=f"{WorkloadType.RESNET50.value}",
    script_file_path=f"{WorkloadType.RESNET50.value}/resnet_ddp.py",
)

WORKLOAD_INFOS: dict[WorkloadType, WorkloadInfo] = {
    WorkloadType.RESNET50: BASE_WORKLOAD_INFO,
    WorkloadType.RESNET18: BASE_WORKLOAD_INFO,
    WorkloadType.MOBILENETV3: BASE_WORKLOAD_INFO,
    WorkloadType.MOBILENETV2: BASE_WORKLOAD_INFO,
    WorkloadType.EFFICIENTNET: BASE_WORKLOAD_INFO,
    WorkloadType.VGG11: BASE_WORKLOAD_INFO,
    WorkloadType.DCGAN: BASE_WORKLOAD_INFO,
    WorkloadType.POINTNET: BASE_WORKLOAD_INFO,
    WorkloadType.BERT: BASE_WORKLOAD_INFO,
    WorkloadType.LSTM: BASE_WORKLOAD_INFO,
    WorkloadType.TRANSFORMER: BASE_WORKLOAD_INFO,
}
