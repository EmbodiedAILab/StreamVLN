#!/usr/bin/env python3
"""
Convert ObjectNav JSON.gz format to R2R annotations format.

Source Format (ObjectNav):
{
  "episode_id": "0",
  "scene_id": "data/scene_datasets/cloudrobo_v1/train/.../scene.glb",
  "object_category": "sofa",
  "reference_replay": [
    {"action": "STOP"},
    {"action": "LOOK_DOWN", "agent_state": {...}},
    {"action": "MOVE_FORWARD", "agent_state": {...}},
    ...
  ]
}

Target Format (R2R):
{
  "id": 0,
  "video": "images/suzhou-room-shengwei-metacam_cloudrobov1_0",
  "instructions": ["sofa"],
  "actions": [-1, 1, 3, 2, ...]
}
"""

import gzip
import json
import argparse
from pathlib import Path
from collections import Counter

# Action mapping (skip LOOK_UP/LOOK_DOWN)
ACTION_MAP = {
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}


def extract_scene_name(scene_id: str) -> str:
    """Extract scene name from scene_id path.

    Input:  data/scene_datasets/cloudrobo_v1/train/suzhou-room-shengwei-metacam-2025-07-09_01-13-22/suzhou-room-shengwei-metacam-2025-07-09_01-13-22.glb
    Output: suzhou-room-shengwei-metacam-2025-07-09_01-13-22
    """
    parts = scene_id.split("/")
    glb_file = parts[-1]
    return glb_file.replace(".glb", "")


def convert_reference_replay(reference_replay: list) -> list:
    """Convert reference_replay to R2R action format.

    - First step (index 0): Always convert to -1 (dummy action)
    - Skip LOOK_UP and LOOK_DOWN actions entirely
    - Skip last action if it's STOP
    - Map other actions to their integer IDs
    """
    actions = []
    last_idx = len(reference_replay) - 1
    for i, step in enumerate(reference_replay):
        action_str = step["action"]

        # Skip last action if it's STOP
        if i == last_idx and action_str == "STOP":
            continue

        # First step -> dummy -1
        if i == 0:
            actions.append(-1)
            continue

        # Skip LOOK_UP and LOOK_DOWN
        if action_str in ["LOOK_UP", "LOOK_DOWN"]:
            continue

        # Map action to ID
        if action_str in ACTION_MAP:
            actions.append(ACTION_MAP[action_str])
        else:
            print(f"Warning: Unknown action '{action_str}', skipping")

    return actions


def convert_episode(episode: dict) -> dict:
    """Convert single ObjectNav episode to R2R format."""
    scene_name = extract_scene_name(episode["scene_id"])
    actions = convert_reference_replay(episode["reference_replay"])
    episode_id = episode['episode_id']

    return {
        "id": int(episode_id),
        "video": f"images/{scene_name}_cloudrobov1_{episode_id}",
        "instructions": [episode["object_category"]],
        "actions": actions
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert ObjectNav JSON.gz to R2R annotations format"
    )
    parser.add_argument("--input", required=True, help="Input ObjectNav JSON.gz file")
    parser.add_argument("--output", required=True, help="Output R2R JSON file")
    args = parser.parse_args()

    # Load ObjectNav data
    print(f"Loading {args.input}...")
    with gzip.open(args.input, "rt") as f:
        data = json.load(f)

    # Convert episodes
    print(f"Converting {len(data['episodes'])} episodes...")
    r2r_data = []
    skipped_actions = Counter()

    for i, episode in enumerate(data['episodes']):
        converted = convert_episode(episode)

        # Track skipped actions for stats
        for step in episode['reference_replay']:
            if step['action'] in ['LOOK_UP', 'LOOK_DOWN']:
                skipped_actions[step['action']] += 1

        r2r_data.append(converted)

    # Save output
    print(f"Saving to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(r2r_data, f, indent=2)

    # Print stats
    print(f"\nConversion complete!")
    print(f"  原本数据量: {len(data['episodes'])}")
    print(f"  Episodes converted: {len(r2r_data)}")
    print(f"  Skipped actions: {dict(skipped_actions)}")


if __name__ == "__main__":
    main()
