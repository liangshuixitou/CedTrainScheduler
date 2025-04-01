import numpy as np


def generate_poisson_timestamps(n_tasks, time_range=(0, 10000)):
    """
    生成泊松分布的时间戳
    params:
        n_tasks: 任务数量
        time_range: 时间范围元组 (start, end)
    """
    start_time, end_time = time_range
    total_time = end_time - start_time

    # 根据任务数量和时间区间计算合适的 lambda
    # 期望平均间隔 = 总时间 / 任务数量
    expected_interval = total_time / n_tasks

    # 生成时间间隔（指数分布）
    intervals = np.random.exponential(expected_interval, n_tasks)
    # 累加得到时间戳
    timestamps = np.cumsum(intervals)
    # 将时间戳缩放到指定范围
    timestamps = timestamps * total_time / timestamps[-1]

    return timestamps.tolist()
