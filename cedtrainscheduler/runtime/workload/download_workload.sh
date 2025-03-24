 #!/bin/bash

set -e  # 遇到错误立即退出

# 设置日志颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# 定义常量
MODEL_DIR=~/data/models
DATASET_DIR=~/data/datasets

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# 创建目录（如果不存在）
mkdir_if_not_exists() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        log_info "Created directory: $1"
    fi
}

# 下载模型
download_model() {
    local model_type=$1
    local output_dir=$2
    local model_url=""
    local model_filename=""

    # 确定模型 URL 和文件名
    if [ "$model_type" == "resnet50" ]; then
        model_url="https://download.pytorch.org/models/resnet50-19c8e357.pth"
        model_filename="resnet50.pth"
    else
        log_error "Unsupported model type: $model_type"
        exit 1
    fi

    # 下载模型
    local output_path="$output_dir/$model_filename"
    log_info "Downloading model from $model_url"
    curl -L -o "$output_path" "$model_url"

    if [ $? -eq 0 ]; then
        log_info "Successfully downloaded $model_filename to $output_path"
    else
        log_error "Failed to download model"
        exit 1
    fi
}

# 下载数据集
download_dataset() {
    local dataset_type=$1
    local output_dir=$2
    local dataset_url=""

    # 确定数据集 URL
    if [ "$dataset_type" == "resnet50" ]; then
        dataset_url="https://www.kaggle.com/api/v1/datasets/download/hylanj/mini-imagenetformat-csv"
    else
        log_error "Unsupported dataset type: $dataset_type"
        exit 1
    fi

    # 创建数据集目录
    local dataset_dir="$output_dir/$dataset_type"
    mkdir -p "$dataset_dir"

    # 临时zip文件路径
    local zip_path="$output_dir/${dataset_type}.zip"

    # 下载数据集zip文件
    log_info "Downloading dataset from $dataset_url"
    if curl -L -o "$zip_path" "$dataset_url"; then
        log_info "Downloaded dataset to $zip_path"

        # 解压文件
        log_info "Extracting dataset to $dataset_dir"
        if unzip -q "$zip_path" -d "$dataset_dir"; then
            log_info "Successfully extracted dataset to $dataset_dir"

            # 删除zip文件
            rm "$zip_path"
            log_info "Removed temporary zip file: $zip_path"
        else
            log_error "Failed to extract dataset"
            log_warn "You may need to extract it manually with:"
            log_warn "unzip $zip_path -d $dataset_dir"
        fi
    else
        log_error "Failed to download dataset"
        log_warn "You can manually download the dataset using:"
        log_warn "curl -L -o $zip_path $dataset_url"
        log_warn "And then extract it to $dataset_dir"
    fi
}

# 主函数
main() {
    # 创建必要的目录
    mkdir_if_not_exists "$MODEL_DIR"
    mkdir_if_not_exists "$DATASET_DIR"

    # 下载模型和数据集
    download_model "resnet50" "$MODEL_DIR"
    download_dataset "resnet50" "$DATASET_DIR"

    log_info "All downloads completed successfully!"
}

# 执行主函数
main