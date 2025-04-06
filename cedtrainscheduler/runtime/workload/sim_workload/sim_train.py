import argparse
import os
import random
import time
from datetime import timedelta

from tqdm import tqdm


def format_time(seconds):
    """将秒数转换为人类可读的格式"""
    return str(timedelta(seconds=int(seconds)))


def log_print(message, flush=True):
    """统一的日志打印函数"""
    print(f"[GPU {args.device_id}] [Rank {args.rank}] {message}", flush=flush)


def load_data():
    """Simulate data loading with progress bar"""
    log_print(f"Starting data loading (estimated time: {args.data_transfer_time} seconds)")
    with tqdm(total=args.data_transfer_time, desc="Loading data", unit="s") as pbar:
        for _ in range(args.data_transfer_time):
            time.sleep(1)
            pbar.update(1)
    log_print("Data loading completed")


# ==================== 训练主函数 ====================
def train():
    train_start_time = time.time()

    log_print(f"Starting training on device: {args.device_id}")
    log_print(f"World size: {args.world_size}, Rank: {args.rank}")
    epoch = 0
    while True:
        epoch_start_time = time.time()
        log_print(f"Starting epoch {epoch+1}")

        data_len = 1200
        for i in range(data_len):
            # 在每个epoch的第一个批次同步时间
            if i == 0:
                current_runtime = time.time() - train_start_time

                if current_runtime >= args.runtime:
                    log_print("Model converged, training completed")
                    break

            time.sleep(0.2)

            # 每个进程都打印日志
            if i % 10 == 0:
                log_print(f"Epoch [{epoch+1}], Step [{i}/{data_len}], Loss: {random.random():.4f}")

        if current_runtime >= args.runtime:
            break

        epoch_time = time.time() - epoch_start_time
        log_print(f"Epoch {epoch+1} time: {format_time(epoch_time)}")
        epoch += 1

    total_time = time.time() - train_start_time
    log_print(f"Total training time: {format_time(total_time)}")


# ==================== 参数解析 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--model_file_path", type=str, required=True)
    parser.add_argument("--dataset_dir_path", type=str, required=True)
    parser.add_argument("--runtime", type=int, default=3600, help="Training duration in seconds")
    parser.add_argument("--data_transfer_time", type=int, default=10, help="Data transfer time in seconds")
    # 可以添加其他参数
    args = parser.parse_args()

    # 从环境变量中获取分布式训练参数
    # 这些值在脚本中已经通过export设置
    args.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    args.master_port = os.environ.get("MASTER_PORT", "29500")
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    log_print(f"Arguments: {args}")

    load_data()

    train()
