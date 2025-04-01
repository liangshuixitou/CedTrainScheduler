import os

INTER_DOMAIN_BANDWIDTH = 1000
INTRA_DOMAIN_BANDWIDTH = 200

# 使用绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FS_CONFIG_PATH = os.path.join(BASE_DIR, "fs_config.json")
TASK_RECORD_SAVE_PATH = os.path.join(BASE_DIR, "task_record.json")
