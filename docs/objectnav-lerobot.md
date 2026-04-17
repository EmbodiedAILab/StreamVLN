# ObjectNav LeRobot 数据集集成

将 ObjectNav HM3D 数据集（LeRobot v3.0 格式）集成到 StreamVLN 训练流程中。

## 快速开始

### 1. 安装依赖

```bash
# 一键安装
bash scripts/install_objnav_lerobot_deps.sh

# 或手动安装
pip install -r requirements-objnav-lerobot.txt
```

### 2. 运行测试

```bash
# 测试数据集加载
python scripts/test_objnav_simple.py

# 测试训练流程（20步快速测试）
bash scripts/test_train_objnav_lerobot.sh
```

### 3. 开始训练

```bash
# 完整训练
bash scripts/train_objnav_lerobot.sh
```

## 核心特性

✅ **无需lerobot库** - 纯pandas + fastparquet实现
✅ **AV1视频支持** - 使用PyAV解码AV1编码视频
✅ **高性能** - Actions缓存 + 批量读取优化
✅ **多数据集** - 自动扫描多个episode目录
✅ **完整训练** - 集成DeepSpeed + LoRA训练流程

## 关键依赖

| 依赖 | 版本 | 用途 | 必需 |
|------|------|------|------|
| PyAV | 15.1.0 | AV1视频解码 | ✅ |
| fastparquet | 2024.2.0 | Parquet读取 | ✅ |
| pandas | 2.0.3 | 数据处理 | ✅ |
| PyTorch | 2.1.0 | 深度学习 | ✅ |
| PyArrow | 14.0.1 | 列读取 | 推荐 |
| OpenCV | 4.8.1.78 | 视频备用 | 可选 |

## 文档结构

```
docs/
├── objectnav-lerobot-quickstart.md       # 快速开始指南 (5分钟)
├── objectnav-lerobot-implementation.md   # 技术实现说明
└── objectnav-lerobot-dependencies.md     # 依赖安装详解

scripts/
├── install_objnav_lerobot_deps.sh        # 一键安装脚本
├── test_objnav_simple.py                 # 数据集测试
├── test_train_objnav_lerobot.sh          # 训练测试脚本
└── train_objnav_lerobot.sh               # 完整训练脚本
```

## 使用方法

### 单个数据集
```bash
python streamvln/streamvln_train.py \
    --use_objnav_lerobot True \
    --objnav_lerobot_root data/trajectory_data/objectnav/hm3d_v2_lerobot3/episode_001 \
    --num_frames 32 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4
```

### 多个数据集
```bash
# 脚本会自动扫描目录下所有episode子目录
python streamvln/streamvln_train.py \
    --use_objnav_lerobot True \
    --objnav_lerobot_root data/trajectory_data/objectnav/hm3d_v2_lerobot3
```

## 数据集统计

| 指标 | 值 |
|------|-----|
| Episodes | 2,614 (测试集) |
| Total Frames | 304,017 |
| Training Samples | ~20,000-40,000 |
| Video Resolution | 640x480 |
| FPS | 3 |
| Video Codec | AV1 (libdav1d) |
| Action Types | 4 (STOP, FORWARD, LEFT, RIGHT) |

## 技术亮点

### 1. Parquet兼容性解决方案
使用fastparquet引擎避免PyArrow的`Repetition level histogram size mismatch`错误。

### 2. Fallback机制
当episodes parquet无法读取时，从data文件统计帧数，确保数据加载鲁棒性。

### 3. 动态视频映射
从episodes parquet读取视频文件信息，正确映射episode到对应视频文件。

### 4. 性能优化
- **Actions缓存**: 预加载所有actions到内存
- **批量读取**: 使用PyArrow读取指定列
- **PyAV解码**: 高效的AV1视频解码

## 常见问题

### Q: PyAV安装失败？
```bash
# 安装系统依赖
sudo apt-get install -y ffmpeg libavcodec-dev

# 重新安装PyAV
pip install av==15.1.0
```

### Q: 视频解码错误？
```bash
# 检查PyAV是否正确安装
python -c "import av; print(av.__version__)"

# 测试视频读取
python -c "
import av
container = av.open('test.mp4')
print(f'Codec: {container.streams.video[0].codec_context.name}')
"
```

### Q: CUDA内存不足？
```bash
# 减小配置
--num_frames 16              # 减少帧数
--per_device_train_batch_size 1
--gradient_accumulation_steps 2
--model_max_length 8192      # 减少序列长度
```

## 测试结果

```
数据集加载: ✓ 20,233 samples
模型加载: ✓ LoRA adapters added
训练流程: ✓
  Step 5:  loss=0.6404, grad_norm=1.504
  Step 10: loss=0.3358, grad_norm=0.714
  Step 15: loss=0.2845, grad_norm=0.892
  Step 20: loss=0.2431, grad_norm=0.678
```

## 性能指标

| 配置 | GPU利用率 | 显存占用 | 训练速度 |
|------|---------|---------|---------|
| 1x V100 (32GB) | 85% | ~24GB | ~6.5s/step |
| 8x V100 (32GB) | 90% | ~28GB/GPU | ~1.2s/step |

## 贡献者

- 实现时间: 2025-04-15
- 数据集版本: LeRobot v3.0
- 测试数据: hm3d_v2_lerobot3_test

## 许可证

与 StreamVLN 项目保持一致。

## 联系方式

如有问题，请查阅完整文档：
- [快速开始](docs/objectnav-lerobot-quickstart.md)
- [技术实现](docs/objectnav-lerobot-implementation.md)
- [依赖安装](docs/objectnav-lerobot-dependencies.md)
