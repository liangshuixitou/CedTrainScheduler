import argparse
import os

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


# ==================== 自定义数据集类 ====================
class ImageNetCSV(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # 支持两种路径格式：直接在images目录下或子目录结构
        img_path = os.path.join(self.root_dir, "images", img_name)
        if not os.path.exists(img_path):
            subdir = img_name.split('_')[0]  # 根据实际目录结构调整
            img_path = os.path.join(self.root_dir, "images", subdir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# ==================== 分布式训练配置 ====================
def setup():
    dist.init_process_group(
        backend='nccl',
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        world_size=args.world_size,
        rank=args.rank
    )

def cleanup():
    dist.destroy_process_group()

# ==================== 训练主函数 ====================
def train():
    # 初始化分布式环境
    setup()
    device = torch.device(f'cuda:{args.rank % 2}')  # 绑定到对应的GPU

    # 加载ResNet50模型
    model = models.resnet50(pretrained=False)
    model_path = os.path.expanduser("~/data/models/resnet50.pth")

    # 容错加载预训练权重
    try:
        model.load_state_dict(torch.load(model_path), strict=True)
        if args.rank == 0:
            print("严格模式加载模型成功")
    except RuntimeError:
        model.load_state_dict(torch.load(model_path), strict=False)
        if args.rank == 0:
            print("警告：使用非严格模式加载模型")

    model = model.to(device)
    ddp_model = DDP(model, device_ids=[device])

    # ImageNet标准化参数
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # 加载训练集
    train_dataset = ImageNetCSV(
        root_dir="~/data/datasets/restnet",
        csv_file=os.path.expanduser("~/data/datasets/restnet/train.csv"),
        transform=train_transform
    )

    # 分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank
    )

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,  # 根据显存调整
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler
    )

    # 定义优化器和损失函数
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001, momentum=0.9)

    # 训练循环
    for epoch in range(5):
        train_sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = ddp_model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 主进程打印日志
            if args.rank == 0 and i % 10 == 0:
                print(f'Epoch [{epoch+1}/5], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

    cleanup()

# ==================== 参数解析 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_addr", default="127.0.0.1", type=str)
    parser.add_argument("--master_port", default="29500", type=str)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    args = parser.parse_args()

    train()
