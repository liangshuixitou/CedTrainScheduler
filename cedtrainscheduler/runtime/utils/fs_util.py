import os


def get_file_size(file_path: str) -> int:
    """获取文件大小"""
    return os.path.getsize(file_path)


def get_file_size_mb(file_path: str) -> int:
    return get_file_size(file_path) / 1024 / 1024


def get_dir_size_mb(dir_path: str) -> int:
    total_size = 0
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            total_size += get_file_size_mb(item_path)
        elif os.path.isdir(item_path):
            total_size += get_dir_size_mb(item_path)
    return total_size
