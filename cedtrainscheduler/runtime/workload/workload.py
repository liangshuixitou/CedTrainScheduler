import logging
import os
from enum import Enum

MODEL_DIR = "~/data/models"
DATASET_DIR = "~/data/datasets"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)

class WorkloadType(str, Enum):
    RESNET50 = "resnet50"

class WorkloadInfo:
    def __init__(self,
                 workload_type: WorkloadType,
                 model_file_name: str,
                 dataset_dir_name: str,
                 script_file_path: str,
                 ):
        self.workload_type = workload_type
        self.model_file_path = os.path.expanduser(f"{MODEL_DIR}/{model_file_name}")
        self.dataset_dir_path = os.path.expanduser(f"{DATASET_DIR}/{dataset_dir_name}")
        self.script_file_path = os.path.expanduser(f"{SCRIPT_DIR}/{script_file_path}")


WORKLOAD_INFOS: dict[WorkloadType, WorkloadInfo] = {
    WorkloadType.RESNET50: WorkloadInfo(
        workload_type=WorkloadType.RESNET50,
        model_file_name=f"{WorkloadType.RESNET50.value}.pth",
        dataset_dir_name=f"{WorkloadType.RESNET50.value}",
        script_file_path=f"{WorkloadType.RESNET50.value}/resnet_ddp.py"
    )
}
