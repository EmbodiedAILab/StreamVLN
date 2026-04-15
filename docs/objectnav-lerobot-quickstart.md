# ObjectNav LeRobot 快速开始指南

本指南帮助你在5分钟内开始使用ObjectNav LeRobot数据集进行训练。

## 前置条件检查

```bash
# 1. 检查Python版本 (需要3.9+)
python --version

# 2. 检查CUDA (需要12.1+)
nvidia-smi

# 3. 检查GPU显存 (建议16GB+)
nvidia-smi --query-gpu=memory.total --format=csv
```

## 5分钟快速安装

### 步骤1: 创建环境 (1分钟)

```bash
# 创建conda环境
conda create -n streamvln-train python=3.9 -y
conda activate streamvln-train

# 安装PyTorch
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### 步骤2: 安装关键依赖 (2分钟)

```bash
# 安装核心依赖
pip install transformers==4.37.0 accelerate==0.25.0 deepspeed==0.12.6 peft==0.7.1

# 安装数据处理库
pip install pandas==2.0.3 fastparquet==2024.2.0 pyarrow==14.0.1

# 安装视频处理库 (关键!)
pip install av==15.1.0 opencv-python==4.8.1.78 Pillow==10.2.0

# 安装训练辅助
pip install tensorboard tqdm
```

### 步骤3: 验证安装 (1分钟)

```bash
python -c "
import torch, transformers, pandas, av
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ Transformers: {transformers.__version__}')
print(f'✓ Pandas: {pandas.__version__}')
print(f'✓ PyAV: {av.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
"
```

### 步骤4: 运行测试 (1分钟)

```bash
# 设置环境变量
export HF_HUB_OFFLINE=1
export HF_HOME=./checkpoints/hf_home/

# 运行快速测试
python scripts/test_objnav_simple.py
```

**预期输出**:
```
============================================================
✓ Test passed!
============================================================
Dataset initialized: 39146 samples
```

## 使用你的数据集

### 数据集准备

确保你的数据集按以下结构组织：

```
your_data/
├── episode_001/
│   ├── meta/info.json
│   ├── meta/tasks.parquet
│   ├── data/chunk-000/file-000.parquet
│   └── videos/observation.images.rgb/chunk-000/file-000.mp4
└── episode_002/
    └── ...
```

### 快速训练测试 (20步)

```bash
bash scripts/test_train_objnav_lerobot.sh
```

这个脚本会：
- ✅ 加载数据集
- ✅ 初始化模型
- ✅ 运行20个训练步
- ✅ 验证loss下降

**预期输出**:
```
{'loss': 0.6404, 'grad_norm': 1.504, 'learning_rate': 9.0e-05, 'epoch': 0.0}
{'loss': 0.3358, 'grad_norm': 0.714, 'learning_rate': 5.8e-05, 'epoch': 0.0}
```

### 完整训练

修改 `scripts/train_objnav_lerobot.sh` 中的数据集路径：

```bash
# 编辑脚本
vim scripts/train_objnav_lerobot.sh

# 修改第65行
OBJNAV_LEROBOT_ROOT="path/to/your/data"

# 运行训练
bash scripts/train_objnav_lerobot.sh
```

## 常见问题速查

### Q1: ImportError: No module named 'av'
```bash
pip install av==15.1.0
```

### Q2: Repetition level histogram size mismatch
```bash
# 确保使用fastparquet
pip install fastparquet==2024.2.0
```

### Q3: cannot find video stream
```bash
# 检查PyAV是否正确安装
python -c "import av; container = av.open('test.mp4'); print(container.streams.video[0].codec)"
# 应输出: libdav1d (AV1解码器)
```

### Q4: CUDA out of memory
```bash
# 减小batch size和gradient accumulation
--per_device_train_batch_size 1
--gradient_accumulation_steps 2
--num_frames 16  # 减少帧数
```

## 参数调优指南

### 快速测试参数
```bash
--num_frames 16              # 减少帧数
--per_device_train_batch_size 1
--gradient_accumulation_steps 2
--max_steps 20               # 限制步数
--model_max_length 8192      # 减少序列长度
```

### 标准训练参数
```bash
--num_frames 32              # 标准帧数
--per_device_train_batch_size 1
--gradient_accumulation_steps 4
--num_train_epochs 3
--learning_rate 1e-4
--model_max_length 32768
```

### 高质量训练参数
```bash
--num_frames 32
--per_device_train_batch_size 2  # 如果显存足够
--gradient_accumulation_steps 8
--num_train_epochs 5
--learning_rate 5e-5
--mm_projector_lr 1e-5
--warmup_ratio 0.1
```

## 下一步

1. **查看数据集统计**:
   ```bash
   python scripts/list_objnav_lerobot_episodes.py
   ```

2. **可视化训练**:
   ```bash
   tensorboard --logdir checkpoints/StreamVLN_ObjectNav_LeRobot_Test
   ```

3. **完整文档**:
   - 技术实现: `docs/objectnav-lerobot-implementation.md`
   - 依赖安装: `docs/objectnav-lerobot-dependencies.md`

## 获取帮助

遇到问题？检查：
1. ✅ Python版本是否为3.9+
2. ✅ PyAV是否正确安装
3. ✅ 数据集路径是否正确
4. ✅ GPU显存是否充足

更多帮助请参考完整文档。
