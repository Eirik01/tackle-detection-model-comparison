"""
TACDEC Spotting Dataset
Loads pre-extracted features and frame-level labels for tackle detection.

Backbone-agnostic: Automatically detects feature files from different extractors.
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import re


class TACDECSpottingDataset(Dataset):
    """
    Dataset for TACDEC tackle spotting task.
    
    Loads pre-extracted visual features and corresponding frame-level annotations.
    Handles padding/truncation for batch processing.
    
    Args:
        features_dir (str/Path): Directory containing .npz feature files
        labels_dir (str/Path): Directory containing .json label files
        max_sequence_length (int): Fixed length for batching (pad/truncate)
        extraction_fps (float): FPS used during feature extraction
        backbone (str): Backbone identifier to filter features (e.g., 'dinov3_b', 'vjepa2_b')
                       If None, uses all available features
        tolerance_sec (float): Temporal tolerance window in seconds. Set to 0 for exact point/interval.
        num_classes (int): Number of output classes (3 or 5)
        labeling_mode (str): Labeling strategy:
            - 'anchor': Labels only center point ± tolerance (action spotting)
            - 'interval': Labels full event range + tolerance padding (temporal detection)
    
    Label Mapping:
        0: tackle-live
        1: tackle-replay  
        2: tackle-live-incomplete
        3: tackle-replay-incomplete
        4: background
    """
    
    def __init__(self, features_dir, labels_dir, max_sequence_length=256,
                 extraction_fps=5.0, backbone=None, tolerance_sec=0.0, num_classes=3, labeling_mode='anchor',
                 feature_type='cls'):
        self.features_dir = Path(features_dir)
        self.labels_dir = Path(labels_dir)
        self.max_len = max_sequence_length
        self.extraction_fps = extraction_fps  # Must match the fps suffix in feature filenames
        self.backbone = backbone
        self.tolerance_sec = tolerance_sec  # Tolerance padding (seconds). Set to 0 for exact point/interval only.
        self.num_classes = num_classes  # 5 or 3 classes
        self.labeling_mode = labeling_mode  # 'anchor' (center ± tolerance) or 'interval' (full range + padding)
        if feature_type not in ('cls', 'dense'):
            raise ValueError(f"feature_type must be 'cls' or 'dense', got '{feature_type}'")
        self.feature_type = feature_type
        # File-naming convention from the extractor:
        #   cls   -> *_<fps>fps_features.npz, key 'cls',   shape [Seq, D]
        #   dense -> *_<fps>fps_dense.npz,    key 'dense', shape [Seq, T*H*W, D]
        self._file_suffix = '_dense.npz' if feature_type == 'dense' else '_features.npz'
        self._file_stem_tail = 'dense' if feature_type == 'dense' else 'features'
        self._npz_key = feature_type
        
        if labeling_mode not in ['anchor', 'interval']:
            raise ValueError(f"labeling_mode must be 'anchor' or 'interval', got '{labeling_mode}'")
        
        # Label mapping for TACDEC (always start with 5-class mapping)
        self.label_map = {
            "tackle-live": 0,
            "tackle-replay": 1,
            "tackle-live-incomplete": 2,
            "tackle-replay-incomplete": 3,
            "background": 4,
        }
        
        # Reverse mapping for display
        self.idx_to_label = {v: k for k, v in self.label_map.items()}
        
        # Set up label remapping if using 3 classes
        if self.num_classes == 3:
            # Merge incomplete classes into parent classes
            # 0: Tackle-Live (Live + Live-Incomplete)
            # 1: Tackle-Replay (Replay + Replay-Incomplete)  
            # 2: Background
            self.remap_labels = lambda x: self._remap_to_3_classes(x)
        else:
            # No remapping for 5 classes
            self.remap_labels = lambda x: x
        
        # Find matching feature-label pairs
        self.samples = self._build_sample_list()
        
        if len(self.samples) == 0:
            print("⚠️  Warning: No matching feature-label pairs found!")
            print(f"   Features dir: {self.features_dir}")
            print(f"   Labels dir: {self.labels_dir}")
            if backbone:
                print(f"   Looking for backbone: {backbone}")
    
    def _remap_to_3_classes(self, labels):
        """
        Remap 5-class labels to 3-class labels.
        Merges incomplete classes into their parent classes.
        
        5-class: [Live=0, Replay=1, Live-Inc=2, Replay-Inc=3, Background=4]
        3-class: [Live=0, Replay=1, Background=2]
        """
        remapped = labels.copy()
        remapped[labels == 2] = 0  # Live-Incomplete -> Live
        remapped[labels == 3] = 1  # Replay-Incomplete -> Replay
        remapped[labels == 4] = 2  # Background -> Background
        return remapped
    
    def _build_sample_list(self):
        """
        Build list of (feature_file, label_file) pairs.
        Automatically detects backbone from feature filename and selects by fps.
        
        Feature file format: {video_id}_{backbone}_{fps}fps_features.npz
        Example: video001_dinov3_l_25fps_features.npz
        """
        samples = []
        fps_str = f"{self.extraction_fps}fps"  # Format fps for matching (e.g., "5fps", "25fps")

        glob_pattern = f"*_{fps_str}{self._file_suffix}"
        id_regex = re.compile(
            rf'(.+?)_(?:dinov3|vjepa2)_[bl]_\d+(?:\.\d+)?fps_{self._file_stem_tail}'
        )

        for feat_file in self.features_dir.glob(glob_pattern):
            # Extract video ID by removing backbone+fps suffix
            filename = feat_file.stem  # Remove .npz

            # If backbone filter is specified, check if it matches
            if self.backbone and self.backbone not in filename:
                continue

            match = id_regex.match(filename)

            if match:
                video_id = match.group(1)
            else:
                # Fallback: try removing common suffixes
                video_id = filename
                for pattern in ["_dinov3_b_", "_dinov3_l_", "_vjepa2_b_", "_vjepa2_l_"]:
                    if pattern in video_id:
                        video_id = video_id.split(pattern)[0]
                        break
            
            # Look for corresponding label file
            label_file = self.labels_dir / f"{video_id}.json"
            
            if label_file.exists():
                samples.append((feat_file, label_file))
        
        return sorted(samples, key=lambda x: x[0].stem)
    
    def __len__(self):
        return len(self.samples)
    
    def get_config(self):
        """
        Returns dataset configuration as a dictionary.
        
        Returns:
            dict: Dataset configuration parameters
        """
        return {
            'features_dir': str(self.features_dir),
            'labels_dir': str(self.labels_dir),
            'max_sequence_length': self.max_len,
            'extraction_fps': self.extraction_fps,
            'backbone_filter': self.backbone,
            'tolerance_sec': self.tolerance_sec,
            'tolerance_frames': int(self.tolerance_sec * self.extraction_fps),
            'num_samples': len(self.samples),
            'num_classes': self.num_classes,
            'labeling_mode': self.labeling_mode,
        }
    
    def __getitem__(self, idx):
        """
        Load and process a single sample.
        
        Returns:
            dict: {
                'features': Tensor [max_len, feature_dim]
                'labels': Tensor [max_len] (int64)
                'mask': Tensor [max_len] (float32) - 1.0 for valid, 0.0 for padding
            }
        """
        feat_path, json_path = self.samples[idx]

        filename = feat_path.stem
        vid_match = re.match(
            rf'(.+?)_(?:dinov3|vjepa2)_[bl]_\d+(?:\.\d+)?fps_{self._file_stem_tail}',
            filename,
        )
        video_id = vid_match.group(1) if vid_match else filename

        # Load features. Shape depends on feature_type:
        #   cls   -> [Seq_Len, Feature_Dim]
        #   dense -> [Seq_Len, num_tokens, Feature_Dim]   (full T*H*W per clip)
        data = np.load(feat_path)
        cls_features = data[self._npz_key]
        
        # Load metadata and labels
        with open(json_path, 'r') as f:
            label_data = json.load(f)
        
        original_fps = label_data['media_attributes']['frame_rate']
        num_sampled_frames = len(cls_features)
        
        # Initialize all frames as background
        frame_labels = np.full(num_sampled_frames, self.label_map["background"], dtype=np.int64)
        
        # Store original event centers (for ground-truth evaluation)
        gt_event_centers = []
        
        # Map event annotations to sampled frame indices
        for event in label_data['events']:
            label_idx = self.label_map.get(event['type'], self.label_map["background"])
            
            # Convert original frame indices to sampled frame indices
            # accounting for FPS conversion from original video fps to extraction fps
            ratio = self.extraction_fps / original_fps  # Map from original to extraction FPS
            
            # Convert original frame range to sampled indices
            start_frame_orig = event['frame_start']
            end_frame_orig = event['frame_end']
            
            # Convert to extraction FPS indices
            start_idx = int(start_frame_orig * ratio)
            end_idx = int(end_frame_orig * ratio)
            
            # Store the true event center (before any padding)
            center_idx = (start_idx + end_idx) // 2
            gt_event_centers.append({
                'class': label_idx,
                'frame': center_idx
            })
            
            if self.labeling_mode == 'anchor':
                # ACTION SPOTTING: Label only around center point ± tolerance
                
                if self.tolerance_sec > 0:
                    tolerance_frames = int(self.tolerance_sec * self.extraction_fps)
                    label_start = center_idx - tolerance_frames
                    label_end = center_idx + tolerance_frames
                else:
                    # No tolerance: label only the center frame
                    label_start = center_idx
                    label_end = center_idx
            else:
                # TEMPORAL DETECTION: Label full interval ± tolerance padding
                if self.tolerance_sec > 0:
                    tolerance_frames = int(self.tolerance_sec * self.extraction_fps)
                    label_start = start_idx - tolerance_frames
                    label_end = end_idx + tolerance_frames
                else:
                    # No tolerance: use exact interval
                    label_start = start_idx
                    label_end = end_idx
            
            # Clamp to valid range
            label_start = max(0, min(label_start, num_sampled_frames - 1))
            label_end = max(0, min(label_end, num_sampled_frames - 1))
            
            # Apply label
            frame_labels[label_start:label_end + 1] = label_idx
        
        # Pad or truncate to fixed length. Per-frame feature shape is either
        # (D,) for cls or (num_tokens, D) for dense -- preserve trailing dims.
        seq_len = min(num_sampled_frames, self.max_len)
        per_frame_shape = cls_features.shape[1:]   # () not allowed; always at least (D,)

        padded_features = np.zeros((self.max_len, *per_frame_shape), dtype=np.float32)
        padded_labels = np.full(self.max_len, self.label_map["background"], dtype=np.int64)

        padded_features[:seq_len] = cls_features[:seq_len]
        padded_labels[:seq_len] = frame_labels[:seq_len]
        
        # Apply label remapping if needed (5-class -> 3-class)
        padded_labels = self.remap_labels(padded_labels)
        
        # Create mask (1.0 for valid frames, 0.0 for padding)
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:seq_len] = 1.0
        
        # Extract team metadata for bias analysis
        team_home_id = label_data.get('metadata', {}).get('team_home', {}).get('id', -1)
        team_away_id = label_data.get('metadata', {}).get('team_away', {}).get('id', -1)
        game_id = label_data.get('metadata', {}).get('game_id', -1)
        
        return {
            "features": torch.from_numpy(padded_features),
            "labels": torch.from_numpy(padded_labels),
            "mask": torch.from_numpy(mask),
            "gt_event_centers": gt_event_centers,
            "team_home_id": team_home_id,
            "team_away_id": team_away_id,
            "game_id": game_id,
            "video_id": video_id,
        }
    
    def get_label_distribution(self):
        """
        Analyze label distribution across entire dataset.
        
        Returns:
            dict: Label counts and percentages
        """
        total_labels = {i: 0 for i in range(5)}
        total_frames = 0
        
        for i in range(len(self)):
            sample = self[i]
            labels = sample['labels'][sample['mask'] == 1].numpy()
            total_frames += len(labels)
            
            for label_id in range(5):
                total_labels[label_id] += (labels == label_id).sum()
        
        # Convert to percentages
        distribution = {}
        for label_id, count in total_labels.items():
            label_name = self.idx_to_label[label_id]
            percentage = (count / total_frames * 100) if total_frames > 0 else 0
            distribution[label_name] = {
                'count': int(count),
                'percentage': percentage
            }
        
        return distribution


if __name__ == "__main__":
    """
    Test the dataset loading and display statistics.
    """
    from config import TACDEC_FEATURES, TACDEC_LABELS
    
    print("="*60)
    print("Testing TACDEC Spotting Dataset")
    print("="*60)
    
    # Test without backbone filter (loads all features)
    print("\n📂 Loading dataset (all backbones)...")
    dataset = TACDECSpottingDataset(
        features_dir=TACDEC_FEATURES,
        labels_dir=TACDEC_LABELS,
        extraction_fps=5.0,
        tolerance_sec=1.0,
        labeling_mode='anchor'  # Try 'anchor' or 'interval'
    )
    
    print(f"\n🎯 Labeling Strategy:")
    print(f"   Mode: {dataset.labeling_mode.upper()}")
    if dataset.labeling_mode == 'anchor':
        if dataset.tolerance_sec > 0:
            print(f"   Labels center point ± {dataset.tolerance_sec}s (±{int(dataset.tolerance_sec * dataset.extraction_fps)} frames)")
        else:
            print(f"   Labels only center frame (no tolerance)")
    else:  # interval
        if dataset.tolerance_sec > 0:
            print(f"   Labels full interval + {dataset.tolerance_sec}s padding (±{int(dataset.tolerance_sec * dataset.extraction_fps)} frames)")
        else:
            print(f"   Labels exact interval (no padding)")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Max sequence length: {dataset.max_len}")
    print(f"   Extraction FPS: {dataset.extraction_fps}")
    
    if len(dataset) > 0:
        # Examine first sample
        print(f"\n🔍 First sample:")
        sample = dataset[0]
        print(f"   Features shape: {sample['features'].shape}")
        print(f"   Labels shape: {sample['labels'].shape}")
        print(f"   Mask shape: {sample['mask'].shape}")
        print(f"   Valid frames: {int(sample['mask'].sum())}")
        
        # Get label distribution
        print(f"\n📈 Full dataset label distribution:")
        dist = dataset.get_label_distribution()
        for label_name, stats in dist.items():
            print(f"   {label_name:25s}: {stats['count']:6d} frames ({stats['percentage']:5.2f}%)")
    else:
        print("\n⚠️  No samples found!")
    
    print("\n" + "="*60)
