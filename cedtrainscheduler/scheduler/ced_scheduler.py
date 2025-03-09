import itertools
import math

from cedtrainscheduler.scheduler.factory import SchedulerType
from cedtrainscheduler.scheduler.scheduler import SchedulerBase
from cedtrainscheduler.scheduler.types.cluster import Cluster
from cedtrainscheduler.scheduler.types.task import ScheduleInfo
from cedtrainscheduler.scheduler.types.task import TaskMeta
from cedtrainscheduler.scheduler.types.task import TaskWrapRuntimeInfo
from cedtrainscheduler.simulator.executor import GPUExecutor
from cedtrainscheduler.simulator.fs import TaskDataInfo


class CEDScheduler(SchedulerBase):
    def __init__(self):
        super().__init__(SchedulerType.CED)
        self.task_queue: list[TaskMeta] = []
        self.waiting_time_threshold = 300  # 设置等待时间阈值（秒）
        self.large_task_threshold = 3600  # 定义大任务的阈值（秒）
        self.small_task_threshold = 600  # 定义小任务的阈值（秒）
        self.last_queue_analysis_time = 0  # 上次分析队列的时间
        self.queue_analysis_interval = 60  # 队列分析间隔（秒）
        self.large_task_weight = 0.5  # 大任务权重初始值
        self.small_task_weight = 0.5  # 小任务权重初始值
        self.predicted_end_times = {}  # 记录每个GPU的预计结束时间

    def analyze_queue_composition(self, current_time):
        """分析队列中大小任务的比例，动态调整权重"""
        if current_time - self.last_queue_analysis_time < self.queue_analysis_interval:
            return

        self.last_queue_analysis_time = current_time

        if not self.task_queue:
            return

        # 统计大小任务数量
        large_tasks = 0
        small_tasks = 0
        medium_tasks = 0

        for task in self.task_queue:
            task_size = max(task.task_runtime.values())
            if task_size > self.large_task_threshold:
                large_tasks += 1
            elif task_size < self.small_task_threshold:
                small_tasks += 1
            else:
                medium_tasks += 1

        total_tasks = len(self.task_queue)

        # 根据队列组成动态调整权重
        if large_tasks > 0.6 * total_tasks:
            # 队列中大任务占比高，增加小任务权重以减少整体makespan
            self.large_task_weight = 0.3
            self.small_task_weight = 0.7
        elif small_tasks > 0.6 * total_tasks:
            # 队列中小任务占比高，增加大任务权重以避免大任务饥饿
            self.large_task_weight = 0.7
            self.small_task_weight = 0.3
        else:
            # 队列均衡，使用平衡权重
            self.large_task_weight = 0.5
            self.small_task_weight = 0.5

    def sort_task_queue(self, current_time):
        """使用多目标优化排序队列"""
        # 更新任务队列中任务的等待时间
        for task in self.task_queue:
            if not hasattr(task, "submit_time"):
                task.submit_time = current_time

            # 计算任务大小和等待时间
            wait_time = current_time - task.submit_time
            task_size = max(task.task_runtime.values())

            # 计算任务优先级得分
            # 小任务得分：倾向于选择运行时间短的任务
            sjf_score = 1.0 / (task_size + 1)

            # 等待时间得分：等待时间越长得分越高
            wait_score = wait_time / 3600.0  # 归一化，假设最长等待时间为1小时
            wait_score = min(wait_score, 1.0)  # 限制最大值为1

            # 根据任务大小使用不同权重
            if task_size > self.large_task_threshold:
                # 大任务
                task.priority = (sjf_score * 0.3 + wait_score * 0.7) * self.large_task_weight
                # 大任务等待时间超过阈值时额外提升优先级
                if wait_time > self.waiting_time_threshold:
                    task.priority *= 1.0 + wait_time / self.waiting_time_threshold * 0.5
            elif task_size < self.small_task_threshold:
                # 小任务
                task.priority = (sjf_score * 0.7 + wait_score * 0.3) * self.small_task_weight
            else:
                # 中等任务
                task.priority = sjf_score * 0.5 + wait_score * 0.5

        # 按优先级排序（优先级高的先执行）
        self.task_queue.sort(key=lambda x: x.priority if hasattr(x, "priority") else 0, reverse=True)

    def update_predicted_end_times(self, current_time, gpu_task_queue, task_record):
        """更新每个GPU的预计结束时间"""
        self.predicted_end_times = {}
        for gpu_id, executor in gpu_task_queue.items():
            queue_time = executor.queue_time(current_time, task_record)
            self.predicted_end_times[gpu_id] = current_time + queue_time

    def schedule(
        self,
        current_time: float,
        clusters: dict[str, Cluster],
        gpu_task_queue: dict[str, GPUExecutor],
        task_data_info: dict[str, TaskDataInfo],
        task_record: dict[str, TaskWrapRuntimeInfo],
    ) -> tuple[TaskWrapRuntimeInfo, bool]:
        if not self.task_queue or len(self.task_queue) == 0:
            return None, True

        # 分析队列组成并动态调整权重
        self.analyze_queue_composition(current_time)

        # 更新GPU预计结束时间
        self.update_predicted_end_times(current_time, gpu_task_queue, task_record)

        # 排序任务队列
        self.sort_task_queue(current_time)

        # 尝试为前N个任务找到最佳调度方案
        best_task_idx = None
        best_schedule = None
        best_score = float("-inf")

        # 考虑队列前几个任务，而不仅仅是第一个
        candidates = min(5, len(self.task_queue))

        for i in range(candidates):
            task = self.task_queue[i]
            schedule_result, score = self._try_schedule_task(task, current_time, clusters, gpu_task_queue, task_record)

            if schedule_result and score > best_score:
                best_task_idx = i
                best_schedule = schedule_result
                best_score = score

        if best_task_idx is not None:
            # 移除被调度的任务
            task = self.task_queue.pop(best_task_idx)

            # 更新预计结束时间
            for inst_id, schedule_info in best_schedule.schedule_infos.items():
                gpu_id = schedule_info.gpu_id
                gpu_type = gpu_task_queue[gpu_id].gpu_type
                self.predicted_end_times[gpu_id] = (
                    current_time
                    + gpu_task_queue[gpu_id].queue_time(current_time, task_record)
                    + task.task_runtime[gpu_type]
                )

            return best_schedule, len(self.task_queue) == 0

        # 如果没有找到合适的调度方案，尝试调度第一个任务
        task = self.task_queue[0]

        # 寻找集群内的所有GPU
        cluster_gpus = []
        for cluster in clusters.values():
            nodes = cluster.nodes
            for node in nodes:
                for gpu in node.gpus:
                    cluster_gpus.append(gpu.gpu_id)

        # 按照队列总体运行时间排序，选择运行时间最短的节点
        cluster_gpus.sort(key=lambda gpu_id: gpu_task_queue[gpu_id].queue_time(current_time, task_record))

        # 将GPU按类型分组
        gpu_groups = {}
        for gpu_id in cluster_gpus:
            gpu_type = gpu_task_queue[gpu_id].gpu_type
            if gpu_type not in gpu_groups:
                gpu_groups[gpu_type] = []
            gpu_groups[gpu_type].append(gpu_id)

        # 选择执行时间最短的GPU组
        min_execution_time = float("inf")
        selected_group = None

        for gpu_type, group in gpu_groups.items():
            # 选择所需数量的GPU
            selected_gpus = group[: task.task_inst_num]
            if len(selected_gpus) < task.task_inst_num:
                continue  # 如果当前组的GPU数量不足，跳过

            # 计算当前组的任务执行时间
            max_wait_time = max(
                gpu_task_queue[gpu_id].queue_time(current_time, task_record) for gpu_id in selected_gpus
            )
            execution_time = max_wait_time + task.task_runtime[gpu_type]

            # 更新最短执行时间的组
            if execution_time < min_execution_time:
                min_execution_time = execution_time
                selected_group = selected_gpus

        if not selected_group:
            # 如果没有找到合适的GPU组，返回None
            self.task_queue.pop(0)  # 移除无法调度的任务
            return None, len(self.task_queue) == 0

        # 给inst分配等待时间最短的节点
        schedule_infos = {}
        for inst_id, gpu_id in enumerate(selected_group):
            schedule_infos[inst_id] = ScheduleInfo(
                inst_id=inst_id,
                gpu_id=gpu_id,
            )

        # 创建任务运行时信息
        runtime_info = TaskWrapRuntimeInfo(
            task_meta=task,
            schedule_infos=schedule_infos,
            inst_status={},
            inst_data_status={},
            task_submit_time=current_time,
            task_start_time=-math.inf,
            task_end_time=-math.inf,
        )

        # 更新预计结束时间
        for inst_id, schedule_info in runtime_info.schedule_infos.items():
            gpu_id = schedule_info.gpu_id
            gpu_type = gpu_task_queue[gpu_id].gpu_type
            self.predicted_end_times[gpu_id] = (
                current_time
                + gpu_task_queue[gpu_id].queue_time(current_time, task_record)
                + task.task_runtime[gpu_type]
            )

        self.task_queue.pop(0)

        return runtime_info, len(self.task_queue) == 0

    def _try_schedule_task(self, task, current_time, clusters, gpu_task_queue, task_record):
        """尝试为任务找到最佳调度方案并返回评分"""
        # 寻找集群内的所有GPU
        cluster_gpus = []
        for cluster in clusters.values():
            nodes = cluster.nodes
            for node in nodes:
                for gpu in node.gpus:
                    cluster_gpus.append(gpu.gpu_id)

        # 将GPU按类型分组
        gpu_groups = {}
        for gpu_id in cluster_gpus:
            gpu_type = gpu_task_queue[gpu_id].gpu_type
            if gpu_type not in gpu_groups:
                gpu_groups[gpu_type] = []
            gpu_groups[gpu_type].append(gpu_id)

        # 选择执行时间最短的GPU组
        min_execution_time = float("inf")
        selected_group = None
        best_resource_score = 0
        best_end_time_alignment = 0
        task_size = max(task.task_runtime.values())

        for gpu_type, group in gpu_groups.items():
            # 计算任务在此GPU类型上的运行时间
            task_runtime = task.task_runtime[gpu_type]

            # 按照队列总体运行时间排序
            group.sort(key=lambda gpu_id: gpu_task_queue[gpu_id].queue_time(current_time, task_record))

            # 选择所需数量的GPU
            if len(group) < task.task_inst_num:
                continue  # 如果当前组的GPU数量不足，跳过

            # 尝试不同的GPU组合，寻找结束时间最一致的组合
            best_combo = None
            best_combo_score = float("-inf")
            best_combo_execution_time = float("inf")
            best_combo_end_time_variance = float("inf")

            # 如果任务需要的GPU数量较少，尝试更多组合以找到最佳匹配
            if task.task_inst_num <= 4 and len(group) >= task.task_inst_num * 2:
                # 尝试多种组合
                possible_combos = list(itertools.combinations(group, task.task_inst_num))
                # 限制组合数量以避免计算过多
                max_combos = min(20, len(possible_combos))
                combos_to_try = possible_combos[:max_combos]
            else:
                # 只尝试按等待时间排序后的前几个GPU
                combos_to_try = [group[: task.task_inst_num]]

            for combo in combos_to_try:
                # 计算当前组合的任务执行时间
                wait_times = [gpu_task_queue[gpu_id].queue_time(current_time, task_record) for gpu_id in combo]
                max_wait_time = max(wait_times)
                execution_time = max_wait_time + task_runtime

                # 计算任务结束时间
                end_times = [current_time + wait_time + task_runtime for wait_time in wait_times]

                # 计算结束时间与其他GPU预计结束时间的一致性
                end_time_alignment_score = 0
                for gpu_id in combo:
                    predicted_end_time = (
                        current_time + gpu_task_queue[gpu_id].queue_time(current_time, task_record) + task_runtime
                    )

                    # 寻找与此GPU结束时间最接近的其他GPU
                    time_diffs = []
                    for other_gpu_id, other_end_time in self.predicted_end_times.items():
                        if other_gpu_id not in combo:  # 只考虑不在当前组合中的GPU
                            time_diff = abs(predicted_end_time - other_end_time)
                            time_diffs.append(time_diff)

                    # 如果找到了接近的结束时间，增加一致性得分
                    if time_diffs:
                        min_time_diff = min(time_diffs)
                        # 时间差越小，得分越高
                        if min_time_diff < 300:  # 5分钟内的差异视为高度一致
                            end_time_alignment_score += 1.0
                        elif min_time_diff < 600:  # 10分钟内的差异
                            end_time_alignment_score += 0.7
                        elif min_time_diff < 1800:  # 30分钟内的差异
                            end_time_alignment_score += 0.3

                # 计算结束时间的方差（越小越好）
                avg_end_time = sum(end_times) / len(end_times)
                end_time_variance = sum((t - avg_end_time) ** 2 for t in end_times) / len(end_times)

                # 计算资源利用率得分
                avg_wait_time = sum(wait_times) / len(wait_times)
                wait_time_variance = sum((w - avg_wait_time) ** 2 for w in wait_times) / len(wait_times)
                resource_score = 1.0 / (1.0 + wait_time_variance) if wait_time_variance > 0 else 1.0

                # 综合评分：执行时间、资源利用率和结束时间一致性
                combo_score = (
                    (1.0 / execution_time) * 0.4  # 执行时间越短越好
                    + resource_score * 0.3  # 资源利用率越高越好
                    + (end_time_alignment_score / task.task_inst_num) * 0.3  # 结束时间一致性
                )

                # 更新最佳组合
                if combo_score > best_combo_score or (
                    combo_score == best_combo_score and execution_time < best_combo_execution_time
                ):
                    best_combo = combo
                    best_combo_score = combo_score
                    best_combo_execution_time = execution_time
                    best_combo_end_time_variance = end_time_variance

            # 如果找到了合适的组合
            if best_combo:
                # 更新全局最佳选择
                if best_combo_execution_time < min_execution_time or (
                    best_combo_execution_time == min_execution_time and best_combo_score > best_resource_score
                ):
                    min_execution_time = best_combo_execution_time
                    selected_group = best_combo
                    best_resource_score = best_combo_score
                    best_end_time_alignment = end_time_alignment_score / task.task_inst_num

        if not selected_group:
            return None, float("-inf")

        # 给inst分配等待时间最短的节点
        schedule_infos = {}
        for inst_id, gpu_id in enumerate(selected_group):
            schedule_infos[inst_id] = ScheduleInfo(
                inst_id=inst_id,
                gpu_id=gpu_id,
            )

        # 创建任务运行时信息
        runtime_info = TaskWrapRuntimeInfo(
            task_meta=task,
            schedule_infos=schedule_infos,
            inst_status={},
            inst_data_status={},
            task_submit_time=current_time,
            task_start_time=-math.inf,
            task_end_time=-math.inf,
        )

        # 计算调度得分
        wait_time = current_time - task.submit_time if hasattr(task, "submit_time") else 0

        # 任务优先级得分
        if task_size > self.large_task_threshold:
            # 大任务
            priority_factor = self.large_task_weight
            # 大任务等待时间超过阈值时额外提升优先级
            if wait_time > self.waiting_time_threshold:
                priority_factor *= 1.0 + wait_time / self.waiting_time_threshold * 0.5
        elif task_size < self.small_task_threshold:
            # 小任务
            priority_factor = self.small_task_weight
        else:
            # 中等任务
            priority_factor = 0.5

        # 执行时间得分：执行时间越短越好
        execution_score = 1.0 / (min_execution_time + 1)

        # 等待时间得分：等待时间越长得分越高
        wait_score = wait_time / 3600.0  # 归一化
        wait_score = min(wait_score, 1.0)  # 限制最大值

        # 综合得分
        final_score = (
            execution_score * 0.3 + wait_score * 0.2 + best_resource_score * 0.2 + best_end_time_alignment * 0.3
        ) * priority_factor

        return runtime_info, final_score
