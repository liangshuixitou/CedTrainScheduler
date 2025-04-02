import argparse
import os
import time
from datetime import timedelta

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms
from tqdm import tqdm


# ==================== 自定义数据集类 ====================
class ImageNetCSV(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # 创建类别到索引的映射
        unique_labels = sorted(self.data.iloc[:, 1].unique())
        self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # 将字符串标签转换为整数索引
        label_idx = self.class_to_idx[label]

        # 支持两种路径格式：直接在images目录下或子目录结构
        img_path = os.path.join(self.root_dir, "images", img_name)
        if not os.path.exists(img_path):
            subdir = img_name.split("_")[0]  # 根据实际目录结构调整
            img_path = os.path.join(self.root_dir, "images", subdir, img_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label_idx


# ==================== 分布式训练配置 ====================
def setup():
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        world_size=args.world_size,
        rank=args.rank,
    )


def cleanup():
    dist.destroy_process_group()


def format_time(seconds):
    """将秒数转换为人类可读的格式"""
    return str(timedelta(seconds=int(seconds)))


def log_print(message, flush=True):
    """统一的日志打印函数"""
    print(f"[Rank {args.rank}] {message}", flush=flush)


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
    # 初始化分布式环境
    setup()
    device = torch.device("cuda:0")  # 绑定到第一个GPU
    # 记录训练开始时间
    train_start_time = time.time()

    log_print(f"Starting training on device: {device}")
    log_print(f"World size: {args.world_size}, Rank: {args.rank}")

    # 加载ResNet50模型
    model = models.resnet50(pretrained=False)
    model_path = os.path.expanduser(args.model_file_path)

    # 容错加载预训练权重
    try:
        model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)
        log_print("Strict mode model loading succeeded")
    except RuntimeError:
        model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)
        log_print("Warning: Using non-strict mode to load model")

    model = model.to(device)
    ddp_model = DDP(model, device_ids=[device])

    # ImageNet标准化参数
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 数据预处理
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # 加载训练集
    train_dataset = ImageNetCSV(
        root_dir=args.dataset_dir_path,
        csv_file=os.path.expanduser(f"{args.dataset_dir_path}/train.csv"),
        transform=train_transform,
    )

    log_print(f"Dataset size: {len(train_dataset)}")

    # 分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank
    )

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,  # 根据显存调整
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
    )

    log_print(f"DataLoader created with {len(train_loader)} batches")

    # 定义优化器和损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)

    try:
        # 训练循环
        epoch = 0
        while True:
            epoch_start_time = time.time()
            train_sampler.set_epoch(epoch)
            log_print(f"Starting epoch {epoch+1}")

            for i, (images, labels) in enumerate(train_loader):
                # 检查是否达到指定的运行时间
                current_runtime = time.time() - train_start_time
                if current_runtime >= args.runtime:
                    log_print("Model converged, training completed")
                    break

                images = images.to(device)
                labels = labels.to(device)

                outputs = ddp_model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 每个进程都打印日志
                if i % 10 == 0:
                    log_print(f"Epoch [{epoch+1}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

            # 检查是否达到指定的运行时间
            if time.time() - train_start_time >= args.runtime:
                break

            epoch_time = time.time() - epoch_start_time
            log_print(f"Epoch {epoch+1} time: {format_time(epoch_time)}")
    finally:
        # 确保在任何情况下都会清理分布式环境
        total_time = time.time() - train_start_time
        log_print(f"Total training time: {format_time(total_time)}")
        cleanup()


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

    log_print(f"Arguments: {args}")

    load_data()

    train()
