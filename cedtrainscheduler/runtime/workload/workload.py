import logging
from enum import Enum

MODEL_DIR = "~/data/models"
DATASET_DIR = "~/data/datasets"
SCRIPT_DIR = "~/data/scripts"

logger = logging.getLogger(__name__)

class WorkloadType(str, Enum):
    RESNET50 = "resnet50"

class WorkloadInfo:
    def __init__(self,
                 workload_type: WorkloadType,
                 model_url: str,
                 model_file_name: str,
                 dataset_url: str,
                 dataset_dir_name: str,
                 script_file_path: str,
                 ):
        self.workload_type = workload_type
        self.model_url = model_url
        self.model_file_name = model_file_name
        self.dataset_url = dataset_url
        self.dataset_dir_name = dataset_dir_name
        self.script_file_path = script_file_path


WORKLOAD_INFOS: dict[WorkloadType, WorkloadInfo] = {
    WorkloadType.RESNET50: WorkloadInfo(
        workload_type=WorkloadType.RESNET50,
        model_url="https://download.pytorch.org/models/resnet50-19c8e357.pth",
        model_file_name=f"{WorkloadType.RESNET50.value}.pth",
        dataset_url="https://www.kaggle.com/api/v1/datasets/download/hylanj/mini-imagenetformat-csv",
        dataset_dir_name=f"{WorkloadType.RESNET50.value}",
        script_file_path=f"{SCRIPT_DIR}/{WorkloadType.RESNET50.value}/resnet_ddp.py"
    )
}
