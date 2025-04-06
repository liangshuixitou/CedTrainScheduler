import os

INTER_DOMAIN_BANDWIDTH = 200
INTRA_DOMAIN_BANDWIDTH = 1200

# 使用绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_DIR = os.path.join(BASE_DIR, "config")
FS_CONFIG_PATH = os.path.join(CONFIG_DIR, "multi_node_fs_config.json")


RESULT_DIR = os.path.join(BASE_DIR, "metrics")
TASK_RECORD_SAVE_PATH = os.path.join(RESULT_DIR, "task_record.json")
