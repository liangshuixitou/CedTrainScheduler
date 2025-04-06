import sys


def get_python_executable_path() -> str:
    """
    获取当前Python执行器的路径

    Returns:
        str: Python执行器的完整路径
    """
    return sys.executable


PYTHON_PATH = get_python_executable_path()
