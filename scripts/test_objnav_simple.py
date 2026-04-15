#!/usr/bin/env python3
"""
Simple focused test for ObjectNav LeRobot dataset
"""

import sys
import os

# Set offline mode before importing
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = './checkpoints/hf_home/'

sys.path.insert(0, '/root/workspace/StreamVLN_up')

import torch
from dataclasses import dataclass
from transformers import AutoTokenizer
from torchvision import transforms

from streamvln.dataset.objectnav_lerobot_video_dataset import ObjectNavLerobotVideoDataset


@dataclass
class MockDataArgs:
    """Mock data arguments"""
    image_size: int = 384
    num_frames: int = 8  # Reduce for faster testing
    num_history: int = 4  # Reduce for faster testing
    num_future_steps: int = 4
    remove_init_turns: bool = False
    is_multimodal: bool = True
    mm_use_im_start_end: bool = False
    video_folder: str = "./data/trajectory_data/objectnav/hm3d_v2_lerobot3_test"
    transform_train: object = None


def main():
    print("=" * 60)
    print("ObjectNav LeRobot Dataset - Simple Test")
    print("=" * 60)

    # Test with single episode
    data_root = "./data/trajectory_data/objectnav/hm3d_v2_lerobot3_test/1S7LAXRdDqK"

    if not os.path.exists(data_root):
        print(f"\n✗ Test directory not found: {data_root}")
        return False

    print(f"\n1. Testing with: {data_root}")

    # Initialize tokenizer
    print("\n2. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "checkpoints/lmms-lab/LLaVA-Video-7B-Qwen2",
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Set conversation version
        from llava import conversation as conversation_lib
        conversation_lib.default_conversation.version = "qwen"

        print(f"   ✓ Tokenizer loaded")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Create data arguments
    print("\n3. Creating data arguments...")
    data_args = MockDataArgs()
    data_args.video_folder = data_root  # Use single episode
    data_args.transform_train = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
    ])
    print(f"   ✓ video_folder: {data_args.video_folder}")
    print(f"   ✓ num_frames: {data_args.num_frames}")

    # Initialize dataset
    print("\n4. Initializing dataset...")
    try:
        dataset = ObjectNavLerobotVideoDataset(
            tokenizer=tokenizer,
            data_args=data_args,
            task_id="test"
        )
        print(f"   ✓ Dataset initialized: {len(dataset)} samples")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    if len(dataset) == 0:
        print("\n✗ No training samples generated")
        return False

    # Test loading one sample
    print("\n5. Loading one sample...")
    try:
        input_ids, labels, images, time_ids, task = dataset[0]
        print(f"   ✓ Sample loaded!")
        print(f"   - input_ids: {input_ids.shape}")
        print(f"   - labels: {labels.shape}")
        print(f"   - images: {images.shape}")
        print(f"   - time_ids: {time_ids.shape}")
        print(f"   - task: {task}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ Test passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
