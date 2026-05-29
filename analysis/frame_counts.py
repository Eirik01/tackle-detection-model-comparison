"""Analysis script to count frames per class in TACDEC label files."""

import json
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict


def analyze_label_file(label_path: str) -> Tuple[Dict[str, int], int]:
    """
    Analyze a TACDEC label file and count frames per class.

    Merges:
    - tackle-live + tackle-live-incomplete → 'tackle-live'
    - tackle-replay + tackle-replay-incomplete → 'tackle-replay'

    Background frames are calculated as:
    total_frames - sum(all annotated event frames)

    Args:
        label_path: Path to a JSON label file

    Returns:
        Tuple of (frame_counts dict, total_frames)
    """
    with open(label_path) as f:
        data = json.load(f)

    total_frames = data["media_attributes"]["frame_count"]
    frame_counts = defaultdict(int)
    annotated_frames = 0

    # Count frames per event type
    for event in data["events"]:
        frame_start = event["frame_start"]
        frame_end = event["frame_end"]
        event_type = event["type"]

        # Calculate frames in this event (inclusive range)
        event_frames = frame_end - frame_start + 1
        annotated_frames += event_frames

        # Normalize class names (merge variants)
        if event_type.startswith("tackle-live"):
            normalized_type = "tackle-live"
        elif event_type.startswith("tackle-replay"):
            normalized_type = "tackle-replay"
        else:
            normalized_type = event_type

        frame_counts[normalized_type] += event_frames

    # Add background frames
    frame_counts["background"] = total_frames - annotated_frames

    return dict(frame_counts), total_frames


def analyze_folder(folder_path: str) -> None:
    """
    Analyze all label files in a folder and accumulate frame counts.

    Args:
        folder_path: Path to folder containing JSON label files
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: {folder_path} is not a directory")
        return

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return

    accumulated_counts = defaultdict(int)
    total_frames = 0

    for json_file in json_files:
        counts, file_total = analyze_label_file(str(json_file))
        total_frames += file_total
        for class_name, count in counts.items():
            accumulated_counts[class_name] += count

    print(f"\nAnalysis of {len(json_files)} files in {folder.name}/")
    print(f"Total frames across all files: {total_frames:,}")
    print(f"\nFrame counts per class:")

    for class_name in sorted(accumulated_counts.keys()):
        count = accumulated_counts[class_name]
        percentage = (count / total_frames) * 100
        print(f"  {class_name:20s}: {count:10,d} ({percentage:6.2f}%)")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analysis_frame_counts.py <path_to_labels_folder>")
        print("\nExample:")
        print("  python analysis_frame_counts.py data/TACDEC/labels")
        sys.exit(1)

    folder = sys.argv[1]
    analyze_folder(folder)
