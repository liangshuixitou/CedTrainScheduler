from datetime import datetime

from tabulate import tabulate


def format_time(ts):
    if ts is None or ts == 0:
        return "-"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def print_task_list(task_list: list):
    """
    Print the task list in a tabular format.

    Args:
        task_list: list of TaskWrapRuntimeInfo
    """
    headers = ["Task ID", "Task Name", "Instances", "Plan GPU", "Status", "Submit Time", "Start Time", "End Time"]
    table_data = []

    for task in task_list:
        row = [
            task["task_meta"]["task_id"],
            task["task_meta"]["task_name"],
            task["task_meta"]["task_inst_num"],
            task["task_meta"]["task_plan_gpu"],
            task["task_meta"]["task_status"],
            format_time(task["task_submit_time"]),
            format_time(task["task_start_time"]),
            format_time(task["task_end_time"])
        ]
        table_data.append(row)

    print("\n=== Task List ===")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

    # Print all schedule information in a single table
    schedule_headers = ["Task ID", "Task Name", "Instance ID", "GPU ID"]
    schedule_data = []

    for task in task_list:
        task_id = task["task_meta"]["task_id"]
        task_name = task["task_meta"]["task_name"]
        schedule_infos = task["schedule_infos"]

        if schedule_infos:
            for inst_id, schedule_info in schedule_infos.items():
                schedule_data.append([
                    task_id,
                    task_name,
                    inst_id,
                    schedule_info["gpu_id"]
                ])

    if schedule_data:
        print("\n=== Schedule Information ===")
        print(tabulate(schedule_data, headers=schedule_headers, tablefmt="grid"))
        print()
