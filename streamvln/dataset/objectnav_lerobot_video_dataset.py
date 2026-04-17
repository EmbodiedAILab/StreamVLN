"""
ObjectNav LeRobot Video Dataset - Fixed version using fastparquet

References lerobot's implementation but doesn't depend on it.
"""

import os
import torch
import json
import copy
import random
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Sequence, List, Tuple
from PIL import Image
from packaging import version
from tqdm import tqdm

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Install with: pip install pandas")

import tokenizers
import transformers

from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from streamvln.utils.utils import DEFAULT_IMAGE_TOKEN, MEMORY_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN
from streamvln.args import DataArguments

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    tokenizer = copy.deepcopy(tokenizer)
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        tokenizer.add_tokens(["<memory>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>") if "<image>" in tokenizer.get_vocab() else -1
    memory_token_index = tokenizer.convert_tokens_to_ids("<memory>") if "<memory>" in tokenizer.get_vocab() else -1

    # Handle additional_special_tokens_ids
    if hasattr(tokenizer, 'additional_special_tokens_ids') and len(tokenizer.additional_special_tokens_ids) >= 2:
        im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
        unmask_tokens_idx = [198, im_start, im_end]
    else:
        # Fallback if additional_special_tokens_ids is not available or has wrong length
        unmask_tokens_idx = [198]
    nl_tokens = tokenizer("\n").input_ids

    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
            role = roles.get(role, role)
            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index and image_token_index >= 0:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == memory_token_index and memory_token_index >= 0:
                input_id[idx] = MEMORY_TOKEN_INDEX

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    elif conversation_lib.default_conversation.version == "llama_v3":
        from streamvln.dataset.vln_action_dataset import preprocess_llama3
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    else:
        raise ValueError(f"Unsupported conversation version: {conversation_lib.default_conversation.version}")


def load_parquet_with_pandas(file_path: str, columns: Optional[List[str]] = None) -> Optional[Dict]:
    """
    Load parquet file using pandas with fastparquet engine.

    Args:
        file_path: Path to parquet file
        columns: Optional list of column names to read

    Returns:
        Dictionary with column names as keys and numpy arrays as values
    """
    if not HAS_PANDAS:
        return None

    try:
        # Use fastparquet engine to avoid pyarrow compatibility issues
        df = pd.read_parquet(file_path, engine='fastparquet')

        # Filter columns if specified
        if columns is not None:
            df = df[columns]

        # Convert to dictionary of numpy arrays
        return {col: df[col].values for col in df.columns}
    except Exception as e:
        print(f"    Warning: pandas failed to load {file_path}: {e}")
        return None


def load_tasks_file(file_path: str) -> Dict[int, str]:
    """
    Load tasks from parquet file.

    Note: In LeRobot v3.0 format, tasks are stored in the DataFrame index,
    not in a column. The index contains JSON strings like '{"instruction": "chair"}'

    Args:
        file_path: Path to tasks.parquet

    Returns:
        Dictionary mapping episode_index to instruction
    """
    if not HAS_PANDAS:
        return {}

    try:
        # Use fastparquet engine
        df = pd.read_parquet(file_path, engine='fastparquet')

        instructions = {}

        # In LeRobot v3.0, tasks are stored in the index
        # The index contains JSON strings like '{"instruction": "chair"}'
        # and task_index column maps to episode_index
        for idx_str in df.index:
            try:
                task_dict = json.loads(idx_str)

                # Get the task_index for this task
                task_idx = df.loc[idx_str, 'task_index']

                # Use first instruction if multiple exist
                if isinstance(task_dict.get('instruction'), list):
                    instructions[task_idx] = task_dict['instruction'][0]
                else:
                    instructions[task_idx] = task_dict.get('instruction', 'Navigate to the target location.')
            except:
                instructions[task_idx] = "Navigate to the target location."

        return instructions
    except Exception as e:
        print(f"    Warning: Failed to load tasks from {file_path}: {e}")
        return {}


def load_episodes_meta(file_path: str) -> List[Dict]:
    """
    Load episode metadata from parquet file.

    Args:
        file_path: Path to episodes parquet file

    Returns:
        List of episode dictionaries
    """
    result = load_parquet_with_pandas(file_path, columns=['episode_index', 'length'])

    if result is None:
        return []

    episodes = []
    for idx in range(len(result['episode_index'])):
        ep_idx = int(result['episode_index'][idx])
        length = int(result['length'][idx])

        episodes.append({
            'episode_index': ep_idx,
            'length': length,
        })

    return episodes


def load_episodes_meta_from_data(data_dir: str) -> Dict[int, int]:
    """
    Load episode metadata by counting frames from data parquet files.

    This is a fallback when episodes parquet cannot be read due to
    nested columns compatibility issues.

    Args:
        data_dir: Path to data directory containing parquet chunks

    Returns:
        Dictionary mapping episode_index to frame count
    """
    if not HAS_PANDAS:
        return {}

    episode_lengths = {}

    try:
        data_path = Path(data_dir)
        if not data_path.exists():
            return {}

        # Process each chunk directory
        for chunk_dir in sorted(data_path.iterdir()):
            if not chunk_dir.is_dir():
                continue

            for file_path in sorted(chunk_dir.glob('*.parquet')):
                try:
                    # Read only episode_index column
                    df = pd.read_parquet(file_path, engine='fastparquet', columns=['episode_index'])

                    # Count frames per episode
                    episode_indices = df['episode_index'].values
                    for ep_idx in episode_indices:
                        ep_idx = int(ep_idx)
                        episode_lengths[ep_idx] = episode_lengths.get(ep_idx, 0) + 1

                except Exception as e:
                    print(f"    Warning: Failed to process {file_path}: {e}")
                    continue

        return episode_lengths

    except Exception as e:
        print(f"    Warning: Failed to load episode metadata from data: {e}")
        return {}


def load_actions_data(file_path: str) -> Dict[int, np.ndarray]:
    """
    Load actions data from parquet file.

    Args:
        file_path: Path to data parquet file

    Returns:
        Dictionary mapping episode_index to actions array
    """
    result = load_parquet_with_pandas(file_path, columns=['action', 'episode_index', 'index'])

    if result is None or 'action' not in result:
        return {}

    # Group actions by episode_index
    actions_cache = {}

    # Process in batches for efficiency
    batch_size = 10000
    num_rows = len(result['episode_index'])

    for start_idx in range(0, num_rows, batch_size):
        end_idx = min(start_idx + batch_size, num_rows)

        for i in range(start_idx, end_idx):
            ep_idx = int(result['episode_index'][i])
            if ep_idx not in actions_cache:
                actions_cache[ep_idx] = []

            action = result['action'][i]

            # Handle numpy scalar
            if hasattr(action, 'item'):
                action = action.item()
            elif hasattr(action, 'dtype'):
                action = int(action)

            actions_cache[ep_idx].append(action)

    # Convert lists to numpy arrays
    for ep_idx in actions_cache:
        actions_cache[ep_idx] = np.array(actions_cache[ep_idx])

    return actions_cache


class ObjectNavLerobotVideoDataset(torch.utils.data.Dataset):
    """
    ObjectNav Video Dataset using pandas with fastparquet engine.

    Uses fastparquet to load parquet files, which avoids pyarrow
    compatibility issues with LeRobot v3.0 format.
    """

    def __init__(
        self,
        tokenizer,
        data_args,
        task_id: str = "objectnav_lerobot",
    ):
        super(ObjectNavLerobotVideoDataset, self).__init__()

        self.task_id = task_id
        self.image_size = data_args.image_size
        self.tokenizer = tokenizer
        self.transforms = data_args.transform_train
        self.image_processor = SigLipImageProcessor()

        self.num_frames = data_args.num_frames
        self.num_history = data_args.num_history
        self.num_future_steps = data_args.num_future_steps
        self.remove_init_turns = data_args.remove_init_turns

        # Parse lerobot dataset folders
        self.lerobot_folders = data_args.video_folder.split(',')
        print(f"Loading lerobot datasets from: {self.lerobot_folders}")

        # Load dataset info and metadata
        self.episodes_data = self.load_lerobot_data()

        # Build training samples
        self.data_list = self.build_data_list()

        # Action mappings
        self.idx2actions = {
            '0': 'STOP',
            '1': "↑",
            '2': "←",
            '3': "→",
        }

        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]

        # Base prompt template
        prompt = f"You are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversations = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]

        print(f"Dataset initialized: {len(self.data_list)} training samples")

    def __len__(self):
        return len(self.data_list)

    @property
    def task(self):
        return self.task_id

    def load_lerobot_data(self) -> List[Dict]:
        """Load episode data using datasets library (inspired by lerobot)."""
        episodes_data = []

        for folder_path in self.lerobot_folders:
            folder_path = folder_path.strip()
            if not os.path.exists(folder_path):
                print(f"Warning: Folder not found: {folder_path}")
                continue

            # Check if this is a specific episode directory or parent directory
            info_path = os.path.join(folder_path, 'meta', 'info.json')
            if not os.path.exists(info_path):
                # Try to find episode directories if this is a parent directory
                episode_dirs = [d for d in os.listdir(folder_path)
                               if os.path.isdir(os.path.join(folder_path, d))]
                print(f"Found {len(episode_dirs)} episode subdirectories in {folder_path}, processing each...")

                # Process each episode subdirectory
                for episode_dir_name in sorted(episode_dirs):
                    episode_path = os.path.join(folder_path, episode_dir_name)
                    episode_info_path = os.path.join(episode_path, 'meta', 'info.json')

                    if os.path.exists(episode_info_path):
                        # Recursively load this episode directory
                        episodes_data.extend(self._load_single_lerobot_dataset(episode_path))
                    else:
                        print(f"Warning: No info.json found in {episode_path}, skipping")
                continue

            # Load this single directory as a dataset
            episodes_data.extend(self._load_single_lerobot_dataset(folder_path))

        print(f"Loaded {len(episodes_data)} episodes from {len(self.lerobot_folders)} dataset folder(s)")
        return episodes_data

    def _load_single_lerobot_dataset(self, folder_path: str) -> List[Dict]:
        """Load data from a single lerobot dataset directory."""
        episodes_data = []

        info_path = os.path.join(folder_path, 'meta', 'info.json')

        # Load info.json
        with open(info_path, 'r') as f:
            info = json.load(f)

        total_episodes = info['total_episodes']
        print(f"    Loading {total_episodes} episodes from {os.path.basename(folder_path)}")

        # Load tasks (instructions)
        tasks_path = os.path.join(folder_path, 'meta', 'tasks.parquet')
        instructions = load_tasks_file(tasks_path)

        if not instructions:
            print(f"    Warning: Failed to load tasks from {tasks_path}")

        # Load episode metadata
        episodes_meta_dir = os.path.join(folder_path, 'meta', 'episodes')
        episode_lengths = {}

        if os.path.exists(episodes_meta_dir):
            # Try to load from episodes parquet files
            success = False
            for chunk_dir in sorted(os.listdir(episodes_meta_dir)):
                chunk_path = os.path.join(episodes_meta_dir, chunk_dir)
                if os.path.isdir(chunk_path):
                    for file_name in sorted(os.listdir(chunk_path)):
                        if file_name.endswith('.parquet'):
                            file_path = os.path.join(chunk_path, file_name)

                            episodes_meta = load_episodes_meta(file_path)

                            if episodes_meta:
                                success = True
                                for ep_meta in episodes_meta:
                                    ep_idx = ep_meta['episode_index']
                                    length = ep_meta['length']
                                    episode_lengths[ep_idx] = length
                            else:
                                # If parquet loading fails, use fallback
                                print(f"    Warning: Episodes parquet has nested columns, using fallback method...")
                if success:
                    break

            # If parquet loading completely failed, use data files to count frames
            if not success or not episode_lengths:
                print(f"    Loading episode metadata from data files (fallback)...")
                data_dir = os.path.join(folder_path, 'data')
                episode_lengths = load_episodes_meta_from_data(data_dir)
        else:
            print(f"    Warning: Episode metadata directory not found: {episodes_meta_dir}")
            # Try to load from data files
            data_dir = os.path.join(folder_path, 'data')
            episode_lengths = load_episodes_meta_from_data(data_dir)

        # Create episode data entries
        for ep_idx, length in episode_lengths.items():
            episodes_data.append({
                'episode_index': ep_idx,
                'lerobot_folder': folder_path,
                'video_key': 'observation.images.rgb',
                'instruction': instructions.get(ep_idx, "Navigate to the target location."),
                'num_frames': length,
                'info_path': info_path,
            })

        print(f"    Loaded metadata for {len(episodes_data)} episodes")

        # Pre-load all actions from data parquet files
        print(f"    Loading actions cache...")
        actions_cache = {}
        data_dir = os.path.join(folder_path, 'data')

        if os.path.exists(data_dir):
            for chunk_dir in sorted(os.listdir(data_dir)):
                chunk_path = os.path.join(data_dir, chunk_dir)
                if os.path.isdir(chunk_path):
                    for file_name in sorted(os.listdir(chunk_path)):
                        if file_name.endswith('.parquet'):
                            file_path = os.path.join(chunk_path, file_name)

                            # Load actions from this parquet file
                            file_actions = load_actions_data(file_path)

                            # Merge into main cache
                            for ep_idx, actions in file_actions.items():
                                if ep_idx not in actions_cache:
                                    actions_cache[ep_idx] = []
                                actions_cache[ep_idx].extend(actions)

        # Convert lists to numpy arrays
        for ep_idx in actions_cache:
            actions_cache[ep_idx] = np.array(actions_cache[ep_idx])

        # Attach cache to each episode's data
        cache_count = 0
        for ep_data in episodes_data:
            ep_idx = ep_data['episode_index']
            if ep_idx in actions_cache:
                ep_data['_actions_cache'] = actions_cache[ep_idx]
                cache_count += 1

        print(f"    Attached actions cache to {cache_count}/{len(episodes_data)} episodes")

        return episodes_data

    def build_data_list(self) -> List[Tuple]:
        """Build training samples from episodes."""
        data_list = []

        for ep_data_idx, ep_data in enumerate(tqdm(self.episodes_data, desc="Building data list")):
            actions = ep_data.get('_actions_cache')

            if actions is None or len(actions) < 4:
                continue

            # Check for initial rotations
            valid_idx = 0
            if self.remove_init_turns:
                valid_idx = self.clean_initial_rotations(ep_data['instruction'], actions)

            if len(actions) - valid_idx < 4:
                continue

            # Create training samples
            num_rounds = (len(actions) - valid_idx) // self.num_frames
            for n in range(num_rounds + 1):
                if n * self.num_frames == len(actions) - valid_idx:
                    continue
                data_list.append((ep_data_idx, n * self.num_frames, valid_idx))

        print(f"Built {len(data_list)} training samples")
        return data_list

    def load_episode_actions(self, ep_data: Dict) -> Optional[np.ndarray]:
        """Load actions for a specific episode. Uses pre-loaded cache if available."""
        # Check if actions were pre-loaded during dataset initialization
        if '_actions_cache' in ep_data:
            return ep_data['_actions_cache']
        return None

    def clean_initial_rotations(self, instruction: str, actions: np.ndarray) -> int:
        """Skip initial rotation actions at the beginning of an episode."""
        valid_idx = 0
        for i, action in enumerate(actions):
            if action in [2, 3]:  # TURN LEFT or TURN RIGHT
                valid_idx = i + 1
            else:
                break
        return valid_idx

    def actions2text(self, actions: List[int]) -> str:
        """Convert action indices to text representation."""
        converted_sequence = []
        for action in actions:
            act_text = self.idx2actions.get(str(action), '↑')
            if isinstance(act_text, list):
                act_text = random.choice(act_text)
            converted_sequence.append(act_text)
        text = ''.join(converted_sequence)
        return text

    def prepare_conversation(self, conversation, actions: List[int]) -> List[Dict]:
        """Prepare interleaved conversation with action sequences."""
        i = 0
        sources = []
        while i < len(actions):
            source = copy.deepcopy(conversation)
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            step_actions = actions[i:i + self.num_future_steps]
            answer = self.actions2text(step_actions)
            if i == 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            source[1]["value"] = answer
            i += len(step_actions)
            sources.extend(source)
        return sources

    def load_video_frames(self, video_path: str, frame_indices: List[int]) -> torch.Tensor:
        """Load video frames from MP4 file."""
        try:
            import av
            return self._load_frames_pyav(video_path, frame_indices)
        except ImportError:
            # Fallback to opencv if PyAV not available
            return self._load_frames_opencv(video_path, frame_indices)

    def _load_frames_pyav(self, video_path: str, frame_indices: List[int]) -> torch.Tensor:
        """Load frames using PyAV (supports AV1 codec)."""
        import av

        container = av.open(video_path)
        video_stream = container.streams.video[0]

        # Get the maximum frame index we need
        max_idx = max(frame_indices)

        # Decode frames and collect the ones we need
        frames_needed = set(frame_indices)
        frames = {}

        current_frame = 0
        for frame in container.decode(video_stream):
            if current_frame in frames_needed:
                # Convert YUV420P to RGB
                img = frame.to_ndarray(format='rgb24')
                frames[current_frame] = Image.fromarray(img)

                # Check if we have all frames we need
                if len(frames) == len(frames_needed):
                    break

            current_frame += 1
            if current_frame > max_idx:
                break

        container.close()

        # Reorder frames to match original frame_indices order
        ordered_frames = [frames[idx] for idx in frame_indices if idx in frames]

        # Apply image processor to all frames
        frames_processed = []
        for frame in ordered_frames:
            processed = self.image_processor.preprocess(
                images=frame,
                return_tensors='pt'
            )['pixel_values'][0]
            frames_processed.append(processed)

        return torch.stack(frames_processed)

    def _load_frames_opencv(self, video_path: str, frame_indices: List[int]) -> torch.Tensor:
        """Load frames using opencv (slower fallback)."""
        import cv2
        frames = []
        for idx in frame_indices:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transforms is not None:
                    frame = self.transforms(frame)
                frame = self.image_processor.preprocess(images=frame, return_tensors='pt')['pixel_values'][0]
                frames.append(frame)
        return torch.stack(frames)

    def get_video_path(self, ep_data: Dict) -> str:
        """Get the video file path for an episode.

        Uses episode metadata to determine the correct video file.
        """
        folder_path = ep_data['lerobot_folder']
        ep_idx = ep_data['episode_index']

        # Check if we have video file info cached
        if '_video_file_info' not in ep_data:
            # Try to load video file info from episodes parquet
            video_file_info = self._load_video_file_info(ep_data, ep_idx)
            ep_data['_video_file_info'] = video_file_info
        else:
            video_file_info = ep_data['_video_file_info']

        chunk_idx, file_idx = video_file_info

        video_path = os.path.join(
            folder_path,
            'videos',
            ep_data['video_key'],
            f'chunk-{chunk_idx:03d}',
            f'file-{file_idx:03d}.mp4'
        )

        return video_path

    def _load_video_file_info(self, ep_data: Dict, ep_idx: int) -> Tuple[int, int]:
        """Load video file info for an episode.

        Returns (chunk_idx, file_idx) tuple.
        """
        folder_path = ep_data['lerobot_folder']
        episodes_meta_dir = os.path.join(folder_path, 'meta', 'episodes')

        if not os.path.exists(episodes_meta_dir):
            # Default to chunk-000, file-000
            return (0, 0)

        # Try to read from episodes parquet
        for chunk_dir in sorted(os.listdir(episodes_meta_dir)):
            chunk_path = os.path.join(episodes_meta_dir, chunk_dir)
            if os.path.isdir(chunk_path):
                for file_name in sorted(os.listdir(chunk_path)):
                    if file_name.endswith('.parquet'):
                        file_path = os.path.join(chunk_path, file_name)

                        try:
                            # Try using pyarrow to read specific columns
                            import pyarrow.parquet as pq
                            parquet_file = pq.ParquetFile(file_path)

                            # Get column indices
                            schema = parquet_file.schema_arrow
                            ep_idx_col = schema.get_field_index('episode_index')

                            # Try to get video file column indices
                            chunk_col = schema.get_field_index('videos/observation.images.rgb/chunk_index')
                            file_col = schema.get_field_index('videos/observation.images.rgb/file_index')

                            if chunk_col >= 0 and file_col >= 0:
                                # Read all episode indices and video file info
                                table = parquet_file.read(columns=[
                                    'episode_index',
                                    'videos/observation.images.rgb/chunk_index',
                                    'videos/observation.images.rgb/file_index'
                                ])

                                ep_indices = table['episode_index'].to_pylist()
                                chunk_indices = table['videos/observation.images.rgb/chunk_index'].to_pylist()
                                file_indices = table['videos/observation.images.rgb/file_index'].to_pylist()

                                # Find matching episode
                                for i, ep in enumerate(ep_indices):
                                    if int(ep) == ep_idx:
                                        return (int(chunk_indices[i]), int(file_indices[i]))
                        except Exception as e:
                            # If pyarrow fails, continue to next file
                            pass

        # Default to chunk-000, file-000 if not found
        return (0, 0)

    def __getitem__(self, i: int):
        ep_data_idx, start_idx, valid_idx = self.data_list[i]
        ep_data = self.episodes_data[ep_data_idx]

        # Load actions (from cache)
        actions = ep_data.get('_actions_cache')
        actions_len = len(actions)

        # Determine time indices for this sample
        time_ids = np.arange(start_idx, min(start_idx + self.num_frames, actions_len))
        assert len(time_ids) > 0
        actions = np.array(actions)[time_ids]

        # Calculate frame indices to load
        sample_start_frame = time_ids[0] + valid_idx
        sample_end_frame = time_ids[-1] + 1 + valid_idx
        sample_interval = self.num_future_steps

        sample_frame_indices = np.arange(
            sample_start_frame,
            sample_end_frame,
            sample_interval,
            dtype=np.int32
        )

        # History frames
        if time_ids[0] != 0:
            history_interval = max(time_ids[0] // self.num_history, 1)
            history_frame_indices = np.arange(
                valid_idx,
                time_ids[0] + valid_idx,
                history_interval,
                dtype=np.int32
            )
        else:
            history_frame_indices = np.array([], dtype=np.int32)

        # Combine history and sample frames
        all_frame_indices = np.concatenate([history_frame_indices, sample_frame_indices])

        # Load video frames
        video_path = self.get_video_path(ep_data)
        images = self.load_video_frames(video_path, all_frame_indices.tolist())

        # Prepare conversation
        sources = copy.deepcopy(self.conversations)

        if start_idx != 0:
            sources[0]["value"] += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'

        sources[0]["value"] = sources[0]["value"].replace('<instruction>.', ep_data['instruction'])
        interleave_sources = self.prepare_conversation(sources, list(actions))

        # Preprocess
        data_dict = preprocess([interleave_sources], self.tokenizer, True)

        return (
            data_dict["input_ids"][0],
            data_dict["labels"][0],
            images,
            torch.tensor(time_ids),
            self.task
        )


def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """Pad tensors to same length."""
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output


def collate_fn(batch, tokenizer):
    """Collate function for DataLoader."""
    input_ids_batch, labels_batch, image_batch, time_ids_batch, task_type_batch = zip(*batch)
    input_ids_batch = torch.nn.utils.rnn.pad_sequence(
        input_ids_batch, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels_batch = torch.nn.utils.rnn.pad_sequence(
        labels_batch, batch_first=True, padding_value=IGNORE_INDEX
    )
    input_ids_batch = input_ids_batch[:, :tokenizer.model_max_length]
    labels_batch = labels_batch[:, :tokenizer.model_max_length]
    attention_mask = input_ids_batch.ne(tokenizer.pad_token_id)

    img_lens = np.array([i.size(0) for i in image_batch])
    if time_ids_batch[0] is not None:
        time_ids_batch = torch.nn.utils.rnn.pad_sequence(
            time_ids_batch, batch_first=True, padding_value=-1
        )
    image_batch = pad_tensors(image_batch, img_lens)

    return {
        'images': image_batch,
        'time_ids': time_ids_batch,
        'attention_mask': attention_mask,
        'input_ids': input_ids_batch,
        'labels': labels_batch,
        'task_type': task_type_batch
    }
