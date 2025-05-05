from tabulate import tabulate

from cedtrainscheduler.runtime.types.task import TaskWrapRuntimeInfo


def print_task_list(task_list: list[TaskWrapRuntimeInfo]):
    """
    Print the task list in a tabular format.

    Args:
        task_list: list of TaskWrapRuntimeInfo
    """
    headers = ["Task ID", "Task Name", "Instances", "Status", "Submit Time", "Start Time", "End Time"]
    table_data = []

    for task in task_list:
        row = [
            task.task_id,
            task.task_name,
            task.task_inst_num,
            task.task_status,
            task.task_submit_time,
            task.task_start_time,
            task.task_end_time
        ]
        table_data.append(row)

    print("\n=== Task List ===")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()
