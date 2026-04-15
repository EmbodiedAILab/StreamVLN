# ObjectNav LeRobot 数据集依赖安装指南

## 环境要求

- Python: 3.9+
- CUDA: 12.1+
- PyTorch: 2.1+
- GPU: 至少16GB显存（建议32GB）

## 完整依赖安装步骤

### 1. 创建Conda环境

```bash
# 创建Python 3.9环境
conda create -n streamvln-train python=3.9 -y
conda activate streamvln-train
```

### 2. 安装PyTorch和CUDA依赖

```bash
# 安装PyTorch 2.1 with CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 验证CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 3. 安装核心依赖

```bash
# 安装transformers和LLaVA相关
pip install transformers==4.37.0
pip install accelerate==0.25.0
pip install peft==0.7.1
pip install deepspeed==0.12.6

# 安装数据处理库
pip install pandas==2.0.3
pip install fastparquet==2024.2.0
pip install pyarrow==14.0.1

# 安装视频处理库（关键依赖）
pip install av==15.1.0  # PyAV - 用于AV1视频解码
pip install opencv-python==4.8.1.78
pip install Pillow==10.2.0

# 安装训练相关
pip install tensorboard==2.15.1
pip install wandb==0.16.2
pip install tqdm==4.66.1
```

### 4. 安装ObjectNav特定依赖

```bash
# 安装用于R2R数据集处理的依赖（如果需要）
pip install networkx==3.2.1
pip install numpy==1.24.4
pip install scipy==1.11.4
```

### 5. 验证关键依赖

```bash
# 验证PyAV安装（必需）
python -c "import av; print(f'PyAV version: {av.__version__}')"

# 验证fastparquet安装（必需）
python -c "import pandas; df = pandas.read_parquet('test.parquet', engine='fastparquet')"

# 验证PyArrow安装（用于部分列读取）
python -c "import pyarrow; print(f'PyArrow version: {pyarrow.__version__}')"
```

## 最小化依赖安装（仅测试）

如果只想测试数据集加载而不进行完整训练：

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.0
pip install pandas==2.0.3
pip install fastparquet==2024.2.0
pip install av==15.1.0
pip install Pillow==10.2.0
pip install tqdm==4.66.1
```

## 依赖版本说明

### 关键依赖版本锁定

```
# 视频处理（关键）
av==15.1.0              # AV1视频解码，必须安装
opencv-python==4.8.1.78  # 备用视频解码
Pillow==10.2.0           # 图像处理

# 数据处理（关键）
pandas==2.0.3            # 数据加载
fastparquet==2024.2.0    # Parquet读取引擎（必需）
pyarrow==14.0.1          # Parquet列读取（可选但推荐）

# 深度学习框架
torch==2.1.0             # PyTorch
transformers==4.37.0     # HuggingFace Transformers
accelerate==0.25.0       # 分布式训练
deepspeed==0.12.6        # 混合精度训练
peft==0.7.1              # LoRA支持

# 训练辅助
tensorboard==2.15.1      # 训练可视化
wandb==0.16.2           # 实验跟踪
tqdm==4.66.1            # 进度条
```

## 常见依赖问题

### 1. PyAV安装失败

**问题**: 编译错误或找不到ffmpeg

**解决方案**:
```bash
# 方案1: 使用预编译包
pip install av --no-binary :all:

# 方案2: 安装ffmpeg系统依赖
sudo apt-get update
sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev

# 然后重新安装PyAV
pip install av==15.1.0
```

### 2. fastparquet安装失败

**问题**: 缺少numpy依赖

**解决方案**:
```bash
# 先安装numpy
pip install numpy==1.24.4

# 然后安装fastparquet
pip install fastparquet==2024.2.0
```

### 3. PyArrow兼容性问题

**问题**: `Repetition level histogram size mismatch`

**解决方案**: 使用fastparquet作为主要引擎，PyArrow仅用于特定列读取
```python
# 正确用法
df = pd.read_parquet(file_path, engine='fastparquet')  # 主要引擎
table = pq.ParquetFile(file_path).read(columns=[...])  # 特定列
```

### 4. CUDA版本不匹配

**问题**: `CUDA version mismatch`

**解决方案**:
```bash
# 检查系统CUDA版本
nvidia-smi

# 安装匹配的PyTorch版本
# 例如系统CUDA 12.4:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

## 环境验证脚本

创建 `verify_env.py` 脚本验证环境：

```python
#!/usr/bin/env python3
"""ObjectNav LeRobot环境验证脚本"""

import sys

def check_import(module_name, package_name=None):
    """检查模块是否可导入"""
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {module_name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        if package_name:
            print(f"  安装命令: pip install {package_name}")
        return False

def main():
    print("="*60)
    print("ObjectNav LeRobot 环境验证")
    print("="*60)

    checks = []

    # 核心依赖
    print("\n核心依赖:")
    checks.append(check_import("torch", "torch==2.1.0"))
    checks.append(check_import("transformers", "transformers==4.37.0"))
    checks.append(check_import("pandas", "pandas==2.0.3"))

    # 数据处理
    print("\n数据处理:")
    checks.append(check_import("pandas.core.frame", "fastparquet==2024.2.0"))
    checks.append(check_import("pyarrow", "pyarrow==14.0.1"))
    checks.append(check_import("fastparquet", "fastparquet==2024.2.0"))

    # 视频处理
    print("\n视频处理:")
    checks.append(check_import("av", "av==15.1.0"))  # PyAV
    checks.append(check_import("cv2", "opencv-python==4.8.1.78"))
    checks.append(check_import("PIL", "Pillow==10.2.0"))

    # 训练相关
    print("\n训练相关:")
    checks.append(check_import("accelerate", "accelerate==0.25.0"))
    checks.append(check_import("deepspeed", "deepspeed==0.12.6"))
    checks.append(check_import("peft", "peft==0.7.1"))

    # CUDA检查
    print("\nCUDA:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA not available")
            checks.append(False)
    except Exception as e:
        print(f"✗ CUDA check failed: {e}")
        checks.append(False)

    # 总结
    print("\n" + "="*60)
    if all(checks):
        print("✓ 所有依赖已正确安装！")
        return 0
    else:
        print("✗ 部分依赖缺失，请安装后重试")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

运行验证：
```bash
python verify_env.py
```

## Docker环境（可选）

如果使用Docker，创建 `Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 设置环境变量
ENV HF_HOME=/workspace/checkpoints/hf_home/
ENV HF_HUB_OFFLINE=1

CMD ["bash"]
```

对应的 `requirements.txt`:

```
torch==2.1.0
transformers==4.37.0
accelerate==0.25.0
deepspeed==0.12.6
peft==0.7.1

pandas==2.0.3
fastparquet==2024.2.0
pyarrow==14.0.1

av==15.1.0
opencv-python==4.8.1.78
Pillow==10.2.0

tensorboard==2.15.1
wandb==0.16.2
tqdm==4.66.1
```

## 性能优化建议

### 1. 使用SSD存储
- 将数据集放在SSD上可显著提升数据加载速度
- 视频文件读取是I/O密集型操作

### 2. 预加载数据
```bash
# 首次运行会自动缓存actions到内存
# 后续训练会更快
```

### 3. 调整dataloader workers
```bash
# 根据CPU核心数调整
--dataloader_num_workers 4
```

### 4. 使用梯度累积
```bash
# 模拟更大batch size
--per_device_train_batch_size 1
--gradient_accumulation_steps 8
```

## 更新依赖

定期更新依赖以获得bug修复和性能改进：

```bash
# 更新所有依赖
pip install --upgrade -r requirements.txt

# 或单独更新
pip install --upgrade av
pip install --upgrade transformers
```

## 获取帮助

如果遇到依赖问题：
1. 查看错误日志
2. 检查版本兼容性
3. 参考官方文档：
   - [PyTorch](https://pytorch.org/get-started/locally/)
   - [PyAV](https://pyav.org/docs/stable/installation.html)
   - [Transformers](https://huggingface.co/docs/transformers/installation)
