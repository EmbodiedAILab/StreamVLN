#!/bin/bash
# ObjectNav LeRobot 依赖一键安装脚本
# 用法: bash scripts/install_objnav_lerobot_deps.sh

set -e

echo "=========================================="
echo "ObjectNav LeRobot 依赖安装"
echo "=========================================="

# 检查Python版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "检测到Python版本: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "❌ 错误: 需要Python 3.9或更高版本"
    echo "当前版本: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python版本符合要求"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到NVIDIA GPU"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "⚠ 警告: 未检测到NVIDIA GPU"
fi

echo ""
echo "=========================================="
echo "步骤 1/5: 安装PyTorch"
echo "=========================================="

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=========================================="
echo "步骤 2/5: 安装核心深度学习库"
echo "=========================================="

pip install transformers==4.37.0 \
            accelerate==0.25.0 \
            deepspeed==0.12.6 \
            peft==0.7.1

echo ""
echo "=========================================="
echo "步骤 3/5: 安装数据处理库"
echo "=========================================="

pip install pandas==2.0.3 \
            fastparquet==2024.2.0 \
            pyarrow==14.0.1 \
            numpy==1.24.4

echo ""
echo "=========================================="
echo "步骤 4/5: 安装视频处理库 (关键)"
echo "=========================================="

pip install av==15.1.0 \
            opencv-python==4.8.1.78 \
            Pillow==10.2.0

echo ""
echo "=========================================="
echo "步骤 5/5: 安装训练辅助工具"
echo "=========================================="

pip install tensorboard==2.15.1 \
            tqdm==4.66.1 \
            wandb==0.16.2

echo ""
echo "=========================================="
echo "验证安装"
echo "=========================================="

python << 'EOF'
import sys

def check_module(name, attr=None):
    try:
        mod = __import__(name)
        version = getattr(mod, attr or '__version__', 'unknown')
        print(f"✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {name}: {e}")
        return False

print("\n核心依赖:")
all_ok = True
all_ok &= check_module("torch")
all_ok &= check_module("transformers")
all_ok &= check_module("pandas")

print("\n数据处理:")
all_ok &= check_module("fastparquet")
all_ok &= check_module("pyarrow")

print("\n视频处理:")
all_ok &= check_module("av")  # PyAV
all_ok &= check_module("cv2")  # OpenCV
all_ok &= check_module("PIL")  # Pillow

print("\n训练相关:")
all_ok &= check_module("accelerate")
all_ok &= check_module("deepspeed")
all_ok &= check_module("peft")

print("\nCUDA:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA可用: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ CUDA不可用")
except Exception as e:
    print(f"✗ CUDA检查失败: {e}")
    all_ok = False

print("\n" + "="*50)
if all_ok:
    print("✓ 所有依赖安装成功！")
    print("\n下一步:")
    print("  1. 运行测试: python scripts/test_objnav_simple.py")
    print("  2. 开始训练: bash scripts/test_train_objnav_lerobot.sh")
    sys.exit(0)
else:
    print("✗ 部分依赖安装失败，请检查错误信息")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
