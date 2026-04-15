# ObjectNav LeRobot 数据集集成技术说明

## 概述

本文档说明如何将 ObjectNav HM3D 数据集（LeRobot v3.0 格式）集成到 StreamVLN 训练流程中。

### 核心特性

- **数据格式**: LeRobot v3.0（MP4视频 + Parquet元数据）
- **视频编码**: AV1编解码器（使用PyAV解码）
- **数据组织**: 按episode分块存储
- **兼容性**: 无需依赖lerobot库，仅使用pandas + fastparquet

## 架构设计

### 1. 数据集结构

```
data/trajectory_data/objectnav/hm3d_v2_lerobot3/
├── episode_001/                    # 单个episode目录
│   ├── meta/
│   │   ├── info.json              # 数据集信息
│   │   ├── tasks.parquet          # 任务描述（DataFrame index存储）
│   │   └── episodes/              # Episode元数据
│   │       └── chunk-000/
│   │           └── file-000.parquet
│   ├── data/                      # 训练数据
│   │   └── chunk-000/
│   │       └── file-000.parquet   # actions, episode_index等
│   └── videos/                    # 视频文件
│       └── observation.images.rgb/
│           └── chunk-000/
│               ├── file-000.mp4   # AV1编码
│               ├── file-001.mp4
│               └── ...
├── episode_002/
└── ...
```

### 2. 关键技术挑战与解决方案

#### 挑战1: PyArrow兼容性问题
**问题**: PyArrow读取LeRobot parquet文件时报错：`Repetition level histogram size mismatch`

**解决方案**:
```python
# 使用fastparquet引擎
import pandas as pd
df = pd.read_parquet(file_path, engine='fastparquet')
```

#### 挑战2: Nested Columns问题
**问题**: episodes parquet包含大量嵌套列（`stats/observation.images.rgb/min`等），fastparquet无法处理

**解决方案**: 实现fallback机制
```python
def load_episodes_meta_from_data(data_dir: str) -> Dict[int, int]:
    """从data parquet文件统计每个episode的帧数"""
    # 读取data parquet的episode_index列
    df = pd.read_parquet(file_path, engine='fastparquet', columns=['episode_index'])
    # 统计每个episode的帧数
    for ep_idx in df['episode_index'].values:
        episode_lengths[ep_idx] = episode_lengths.get(ep_idx, 0) + 1
```

#### 挑战3: Tasks存储格式
**问题**: tasks.parquet将任务描述存储在DataFrame index中，而非列

**解决方案**:
```python
def load_tasks_file(file_path: str) -> Dict[int, str]:
    df = pd.read_parquet(file_path, engine='fastparquet')
    instructions = {}
    # 从index读取JSON任务描述
    for idx_str in df.index:
        task_dict = json.loads(idx_str)  # '{"instruction": "chair"}'
        task_idx = df.loc[idx_str, 'task_index']
        instructions[task_idx] = task_dict.get('instruction', 'Navigate...')
    return instructions
```

#### 挑战4: AV1视频解码
**问题**: MP4视频使用AV1编解码器，decord和opencv无法解码

**解决方案**: 使用PyAV库
```python
import av

def _load_frames_pyav(self, video_path: str, frame_indices: List[int]):
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    frames_needed = set(frame_indices)
    frames = {}
    current_frame = 0

    for frame in container.decode(video_stream):
        if current_frame in frames_needed:
            img = frame.to_ndarray(format='rgb24')
            frames[current_frame] = Image.fromarray(img)
        current_frame += 1

    return self.process_frames(frames)
```

#### 挑战5: 视频文件映射
**问题**: 需要确定每个episode对应哪个视频文件

**解决方案**: 从episodes parquet动态读取
```python
def _load_video_file_info(self, ep_data: Dict, ep_idx: int):
    """使用PyArrow读取指定列"""
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(file_path)

    table = parquet_file.read(columns=[
        'episode_index',
        'videos/observation.images.rgb/chunk_index',
        'videos/observation.images.rgb/file_index'
    ])

    # 查找匹配的episode
    for i, ep in enumerate(table['episode_index'].to_pylist()):
        if int(ep) == ep_idx:
            return (int(chunk_indices[i]), int(file_indices[i]))
```

## 实现文件

### 核心文件

#### 1. `streamvln/dataset/objectnav_lerobot_video_dataset.py`
**职责**: 数据集加载和预处理

**关键类和函数**:
- `ObjectNavLerobotVideoDataset`: 主数据集类
- `load_parquet_with_pandas()`: 使用fastparquet加载parquet
- `load_tasks_file()`: 加载任务描述
- `load_episodes_meta_from_data()`: 从data统计episode帧数
- `load_actions_data()`: 加载并缓存actions
- `_load_frames_pyav()`: 使用PyAV解码视频

#### 2. `streamvln/args.py` (修改)
**新增参数**:
```python
@dataclass
class DataArguments:
    # ... 现有参数 ...

    # ObjectNav LeRobot specific
    objnav_lerobot_root: Optional[str] = field(
        default=None,
        metadata={"help": "Path to ObjectNav LeRobot dataset root"}
    )
    use_objnav_lerobot: bool = field(
        default=False,
        metadata={"help": "Whether to use ObjectNav LeRobot dataset"}
    )
```

#### 3. `streamvln/streamvln_train.py` (修改)
**修改位置**: `make_supervised_data_module()` 函数

**修改内容**: 添加ObjectNav LeRobot数据集支持
```python
# Support for ObjectNav LeRobot dataset (highest priority)
if getattr(data_args, 'use_objnav_lerobot', False) and data_args.objnav_lerobot_root is not None:
    from streamvln.dataset.objectnav_lerobot_video_dataset import ObjectNavLerobotVideoDataset
    original_video_folder = data_args.video_folder
    data_args.video_folder = data_args.objnav_lerobot_root
    nav_dataset = ObjectNavLerobotVideoDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        task_id="objectnav_lerobot",
    )
    dataset.append(nav_dataset)
    data_args.video_folder = original_video_folder
```

### 辅助文件

#### 4. `scripts/train_objnav_lerobot.sh`
完整训练脚本（用于完整数据集训练）

#### 5. `scripts/test_train_objnav_lerobot.sh`
快速测试脚本（用于测试数据集）

#### 6. `scripts/test_objnav_simple.py`
单元测试脚本

## 数据流程

### 1. 初始化流程
```
1. 解析数据集路径（支持多个episode目录）
   ↓
2. 加载info.json获取总episode数
   ↓
3. 加载tasks.parquet获取任务描述
   ↓
4. 加载episode元数据（优先parquet，fallback到data统计）
   ↓
5. 预加载所有actions到内存缓存
   ↓
6. 构建训练样本列表
```

### 2. 训练样本生成
```python
# 每个episode根据num_frames和num_future_steps生成多个训练样本
# 例如: 116帧的episode，num_frames=16, num_future_steps=4
# 生成 floor((116-0) / 16) = 7个训练样本

for n in range(num_rounds + 1):
    start_idx = n * num_frames
    # 生成时间索引: [start_idx, start_idx+1, ..., start_idx+15]
    # 生成历史帧: 从0到start_idx，均匀采样num_history帧
    # 生成样本帧: 从start_idx开始，每隔num_future_steps采样一帧
```

### 3. 视频帧加载流程
```
1. 计算帧索引（历史帧 + 样本帧）
   ↓
2. 从episodes parquet获取视频文件信息
   ↓
3. 使用PyAV解码视频
   ↓
4. 提取指定帧
   ↓
5. 转换为RGB PIL Image
   ↓
6. 应用SigLIP图像预处理
   ↓
7. 返回tensor: [num_frames, 3, 384, 384]
```

## 性能优化

### 1. Actions缓存
**问题**: 每次构建样本都需要读取parquet获取actions

**优化**: 初始化时一次性加载所有actions到内存
```python
# 初始化时
for file_path in data_parquet_files:
    file_actions = load_actions_data(file_path)
    actions_cache.update(file_actions)

# 使用时
actions = ep_data['_actions_cache']  # 直接从内存读取
```

**效果**: 避免重复读取parquet，加速数据加载

### 2. 视频文件信息缓存
**优化**: 缓存episode到视频文件的映射
```python
if '_video_file_info' not in ep_data:
    video_file_info = self._load_video_file_info(ep_data, ep_idx)
    ep_data['_video_file_info'] = video_file_info
```

### 3. 批量读取优化
- 使用PyArrow的`read(columns=[...])`只读取需要的列
- 避免读取大量stats列

## 数据集统计

### 测试数据集 (`hm3d_v2_lerobot3_test`)
- **Episodes**: 2,614
- **Total Frames**: 304,017
- **Training Samples**: ~20,000-40,000（取决于num_frames配置）
- **Video Resolution**: 640x480
- **FPS**: 3
- **Video Codec**: AV1 (libdav1d)
- **Actions**: 4类 (STOP, MOVE FORWARD, TURN LEFT, TURN RIGHT)

### 完整数据集 (`hm3d_v2_lerobot3`)
- **Episodes**: ~数万个
- **Total Frames**: ~数百万帧
- **文件大小**: 每个episode约100-200MB

## 使用方法

### 1. 单个数据集训练
```bash
python streamvln/streamvln_train.py \
    --use_objnav_lerobot True \
    --objnav_lerobot_root data/trajectory_data/objectnav/hm3d_v2_lerobot3/episode_001 \
    --num_frames 32 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4
```

### 2. 多个数据集训练
```bash
python streamvln/streamvln_train.py \
    --use_objnav_lerobot True \
    --objnav_lerobot_root data/trajectory_data/objectnav/hm3d_v2_lerobot3 \
    --num_frames 32
```
脚本会自动扫描该目录下所有episode子目录。

### 3. 混合数据集训练
```bash
python streamvln/streamvln_train.py \
    --use_objnav_lerobot True \
    --objnav_lerobot_root data/trajectory_data/objectnav/hm3d_v2_lerobot3 \
    --r2r_dataset_root data/trajectory_data/R2R \
    --num_frames 32
```

## 故障排除

### 1. PyArrow兼容性错误
**错误**: `Repetition level histogram size mismatch`

**解决**: 确保使用fastparquet引擎
```python
df = pd.read_parquet(file_path, engine='fastparquet')
```

### 2. 视频解码失败
**错误**: `cannot find video stream with wanted index: -1`

**解决**: 确保安装PyAV
```bash
pip install av
```

### 3. 内存不足
**错误**: CUDA OOM

**解决**:
- 减小`--per_device_train_batch_size`
- 增加`--gradient_accumulation_steps`
- 减小`--num_frames`

### 4. 数据加载慢
**解决**:
- 检查actions缓存是否生效
- 考虑使用SSD存储
- 增加`--dataloader_num_workers`

## 技术限制

1. **视频文件查找**: 当前实现从episodes parquet读取映射，如果parquet损坏会fallback到file-000.mp4
2. **Episode顺序**: 视频帧顺序假设从0开始连续编号
3. **多chunk支持**: 当前主要测试了chunk-000的情况

## 未来改进方向

1. **性能优化**:
   - 实现视频帧缓存（对于小episode）
   - 使用更高效的视频编解码器

2. **功能增强**:
   - 支持多视频文件per episode
   - 添加数据增强选项

3. **兼容性**:
   - 支持更多视频编解码器
   - 优化PyArrow版本兼容性

## 参考资料

- [LeRobot文档](https://github.com/huggingface/lerobot)
- [PyAV文档](https://pyav.org/docs/stable/)
- [ObjectNav论文](https://arxiv.org/abs/1902.08986)
