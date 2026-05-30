from abc import ABC, abstractmethod
from pathlib import Path
import torch
import cv2
import numpy as np
import csv
import os
import random


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extraction from video clips.
    
    REQUIRED to implement (abstract methods):
    - get_model_name(): Returns identifier for file naming (e.g., 'dinov3_l', 'vjepa2_l')
    - load_model(): Initializes the backbone model, should set self.feature_dim
    - extract_features(): Main extraction loop defining your extraction strategy
      (Extractors can use frame-by-frame batching, sliding windows, or custom logic)
    
    OPTIONAL utilities (inherited from base):
    - _get_all_videos(): Crawl input directory for video files
    - _compute_stride(): Calculate FPS sampling stride
    
    Args:
        input_dir (str/Path): Directory containing video files
        output_dir (str/Path): Directory to save extracted features
        device (str): 'cuda' or 'cpu'
    """
    
    def __init__(self, input_dir, output_dir, device="cuda"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
        
        # Set seed for reproducibility (imported from config)
        try:
            from ..config import EXTRACTION_SEED
            self.seed = EXTRACTION_SEED
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            self.seed = 42  # Fallback
        
        print(f"📍 Initialized {self.__class__.__name__}")
        print(f"   Input:  {self.input_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Device: {self.device}")
        print(f"   Seed:   {self.seed} (for reproducibility)")
    
    @abstractmethod
    def get_model_name(self):
        """
        Return a short identifier for this backbone (e.g., 'dinov3_b', 'vjepa2').
        Used for naming output files.
        
        Returns:
            str: Model identifier
        """
        pass
    
    @abstractmethod
    def load_model(self):
        """
        Load and initialize the backbone model.
        Should set self.model and any preprocessing components.
        """
        pass
    
    @abstractmethod
    def extract_features(self, fps=None, batch_size=32, start_idx=None, end_idx=None, override=False, profile_efficiency=False):
        """
        Main extraction loop - processes all videos in input directory.
        Each extractor must define its own strategy (frame-by-frame batching, sliding windows, etc.).
        
        Args:
            fps (float): Target FPS for frame sampling (None = use original FPS)
            batch_size (int): Batch size (may not apply to all strategies)
            start_idx (int): Start index for video processing (inclusive, 0-based)
            end_idx (int): End index for video processing (exclusive, 0-based)
            override (bool): If True, re-extract and overwrite existing feature files
            profile_efficiency (bool): If True, measure end-to-end and GPU throughput
        """
        pass
    
    def _get_all_videos(self):
        """
        Crawl input directory for video files.

        Returns:
            list: Paths to all .mp4 and .mkv files
        """
        video_files = (list(self.input_dir.rglob("*.mp4"))
                       + list(self.input_dir.rglob("*.mkv")))
        return sorted(video_files)

    def _compute_stride(self, original_fps, target_fps):
        """
        Compute a deterministic sampling stride from original and target FPS.
        Uses the same rounding behavior across extractors.
        """
        if target_fps is None:
            return 1
        return max(1, int(original_fps / target_fps))

    @staticmethod
    def _square_with_reflect(rgb_frame):
        """
        Pad the shorter side of an RGB frame with border-reflected pixels until
        it matches the longer side, producing a square image without cropping
        away geometry or stretching the aspect ratio. Uses BORDER_REFLECT_101
        (gfedcb|abcdefgh|gfedcba) so the seam pixel is not duplicated.
        """
        h, w = rgb_frame.shape[:2]
        if h == w:
            return rgb_frame
        if h < w:
            pad = w - h
            top = pad // 2
            bot = pad - top
            return cv2.copyMakeBorder(rgb_frame, top, bot, 0, 0, cv2.BORDER_REFLECT_101)
        pad = h - w
        left = pad // 2
        right = pad - left
        return cv2.copyMakeBorder(rgb_frame, 0, 0, left, right, cv2.BORDER_REFLECT_101)

    def _apply_crop(self, rgb_frame, x1, y1, x2, y2):
        """
        Crop frame region and zero-pad to square, preserving aspect ratio.
        Resizing to 256x256 after padding avoids distorting player proportions.
        """
        crop = rgb_frame[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return rgb_frame
        size = max(h, w)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        dy, dx = (size - h) // 2, (size - w) // 2
        padded[dy:dy + h, dx:dx + w] = crop
        return padded

    def get_feature_dim(self):
        """
        Returns the feature dimension for this model.
        Assumes self.feature_dim is set by load_model().
        
        Returns:
            int: Feature dimension (e.g., 768 for DINOv3-base, 1024 for DINOv3-large/V-JEPA2)
        """
        if not hasattr(self, 'feature_dim'):
            raise RuntimeError("feature_dim not set. Call load_model() first.")
        return self.feature_dim
    
    def _log_extraction_metrics(self, video_path, batch_size_label, num_features, gpu_compute_sec, total_end_to_end_sec, video_duration_sec):
        """
        Log extraction performance metrics to CSV for profiling.
        
        Args:
            video_path (Path): Path to video file
            batch_size_label (str): Batch configuration label (e.g., '32' for DINOv3, '1x16' for V-JEPA2)
            num_features (int): Number of features extracted
            gpu_compute_sec (float): GPU compute time in seconds
            total_end_to_end_sec (float): Total end-to-end time in seconds
            video_duration_sec (float): Original video duration in seconds
        """
        fps_achieved = num_features / gpu_compute_sec if gpu_compute_sec > 0 else 0
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        rtf = total_end_to_end_sec / video_duration_sec if video_duration_sec > 0 else 0
        
        csv_path = "extraction_throughput.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["model_name", "video", "batch_size", "total_frames", "gpu_compute_sec", "throughput_fps", "peak_gpu_gb", "end_to_end_sec", "rtf"])
            writer.writerow([
                self.get_model_name(),
                video_path.name,
                batch_size_label,
                num_features,
                round(gpu_compute_sec, 2),
                round(fps_achieved, 1),
                round(peak_memory, 2),
                round(total_end_to_end_sec, 2),
                round(rtf, 4)
            ])

