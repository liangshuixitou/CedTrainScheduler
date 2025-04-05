import os

INTER_DOMAIN_BANDWIDTH = 1000
INTRA_DOMAIN_BANDWIDTH = 200

# 使用绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_DIR = os.path.join(BASE_DIR, "config")
FS_CONFIG_PATH = os.path.join(CONFIG_DIR, "single_node_fs_config.json")


RESULT_DIR = os.path.join(BASE_DIR, "result")
TASK_RECORD_SAVE_PATH = os.path.join(RESULT_DIR, "task_record.json")
