"""
DINOv3 Feature Extractor
Concrete implementation using Meta's DINOv3 vision transformer.
"""

import torch
import numpy as np
import time
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
import cv2
from .base_extractor import BaseFeatureExtractor
from ..config import HF_TOKEN


class DINOv3Extractor(BaseFeatureExtractor):
    """
    DINOv3-based feature extractor for video frames.
    
    Supports both Base (768-dim) and Large (1024-dim) variants.
    Extracts both global (CLS token) and dense (patch) features.
    
    Args:
        input_dir (str/Path): Directory containing video files
        output_dir (str/Path): Directory to save extracted features
        model_size (str): 'base' or 'large'
        device (str): 'cuda' or 'cpu'
    """
    
    def __init__(self, input_dir, output_dir, model_size="base", device="cuda",
                 padding_mode="center_crop"):
        self.model_size = model_size.lower()

        if self.model_size not in ["base", "large"]:
            raise ValueError(f"model_size must be 'base' or 'large', got '{model_size}'")

        if padding_mode not in ("center_crop", "reflect"):
            raise ValueError(
                f"padding_mode must be 'center_crop' or 'reflect', got '{padding_mode}'"
            )
        self.padding_mode = padding_mode

        # Initialize parent class
        super().__init__(input_dir, output_dir, device)

        # Load model
        self.load_model()
    
    def get_model_name(self):
        """Returns 'dinov3_b' or 'dinov3_l'"""
        return f"dinov3_{self.model_size[0]}"
    
    def load_model(self):
        """
        Load DINOv3 model from Hugging Face.
        Requires HF_TOKEN in config for model access.

        Preprocessing pipeline (applied per-call in extract_frame_features):
        shortest_edge=256 → centre-crop 256×256, preserves aspect ratio.
        """

        # Model mapping
        model_mapping = {
            "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "large": "facebook/dinov3-vitl16-pretrain-lvd1689m"
        }

        model_name = model_mapping[self.model_size]

        print(f"🔄 Loading {model_name}...")

        # Resize/crop kwargs are passed at call time in extract_frame_features
        # so the preprocessing contract lives in one place and we don't mutate
        # processor state.
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            token=HF_TOKEN,
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            token=HF_TOKEN
        ).to(self.device)

        self.model.eval()

        # Store model config info
        self.patch_size = self.model.config.patch_size
        self.num_registers = getattr(self.model.config, "num_register_tokens", 4)
        self.feature_dim = self.model.config.hidden_size

        print(f"✅ Model loaded successfully")
        print(f"   Feature dimension: {self.feature_dim}")
        print(f"   Patch size: {self.patch_size}")
        if self.padding_mode == "reflect":
            print(f"   Preprocessing: reflect-pad to square → resize 256×256 (per-call)")
        else:
            print(f"   Preprocessing: shortest_edge=256 → centre-crop 256×256 (per-call)")
    
    def extract_features(self, fps=None, batch_size=32, start_idx=None, end_idx=None, override=False, profile_efficiency=False, save_dense=False, skip_cls=False):
        """
        Frame-by-frame batch extraction strategy for DINOv3.

        Processes all videos with uniform frame sampling and batch processing.

        Args:
            fps (float): Target FPS for frame sampling (None = use original FPS)
            batch_size (int): Number of frames to process at once
            start_idx (int): Start index for video processing (inclusive, 0-based)
            end_idx (int): End index for video processing (exclusive, 0-based)
            override (bool): If True, re-extract and overwrite existing feature files
            profile_efficiency (bool): If True, measure end-to-end and GPU throughput
            save_dense (bool): If True, also save dense patch features to a sibling
                file ({video_id}_..._dense_features.npy, uncompressed numpy array,
                shape (T, num_patches, feature_dim)). Load with np.load(...,
                mmap_mode='r') for window-level random access.
            skip_cls (bool): If True, do not write the CLS .npz output. Used when
                the run only exists to produce a dense file whose padding mode
                doesn't match the linear probe's CLS preprocessing (so the CLS
                file would be dead weight).
        """
        if skip_cls and not save_dense:
            raise ValueError("skip_cls=True requires save_dense=True (otherwise "
                             "the extractor would produce no outputs).")
        videos = self._get_all_videos()
        total_videos = len(videos)

        # Handle index-based slicing
        if start_idx is not None or end_idx is not None:
            start = start_idx if start_idx is not None else 0
            end = end_idx if end_idx is not None else total_videos
            videos = videos[start:end]
            print(f"🎯 Processing videos [{start}:{end}] of {total_videos} total")

        print(f"📹 Found {len(videos)} video files to process")
        if save_dense:
            print(f"📦 save_dense=True: dense patch features will be written alongside CLS")

        if len(videos) == 0:
            print("⚠️  No videos found! Check input directory.")
            return

        for video_path in tqdm(videos, desc="Processing videos"):
            # Filename format: {video_id}_{backbone}_{fps}fps_features.npz
            # Dense (when save_dense=True): same stem, suffix _dense_features.npz
            pad_tag = "_reflect" if self.padding_mode == "reflect" else ""
            stem = f"{video_path.stem}_{self.get_model_name()}_{fps}fps{pad_tag}"
            output_path = self.output_dir / f"{stem}_features.npz"
            dense_path  = self.output_dir / f"{stem}_dense_features.npy"

            if not override:
                cls_done   = skip_cls or output_path.exists()
                dense_done = (not save_dense) or dense_path.exists()
                if cls_done and dense_done:
                    continue

            self._process_video(video_path, output_path, fps, batch_size,
                                profile_efficiency,
                                save_dense=save_dense, dense_path=dense_path,
                                skip_cls=skip_cls)
    
    def _process_video(self, video_path, output_path, fps, batch_size, profile_efficiency=False, save_dense=False, dense_path=None, skip_cls=False):
        """
        Process a single video file with frame-by-frame batch extraction and save features.

        Args:
            video_path (Path): Path to video file
            output_path (Path): Where to save CLS features (.npz, key='cls')
            fps (float): Target FPS for sampling
            batch_size (int): Batch size for processing
            profile_efficiency (bool): If True, measure GPU performance
            save_dense (bool): If True, also save dense patch features.
            dense_path (Path): Where to save dense features (.npy, uncompressed),
                shape (T, num_patches, feature_dim). Required if save_dense=True.
        """
        if save_dense and dense_path is None:
            raise ValueError("save_dense=True requires dense_path")
        video_start_time = time.perf_counter()
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Failed to open: {video_path}")
            return
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   Original: {width}×{height} @ {original_fps:.1f} FPS")
        
        # Determine frame sampling interval
        skip_interval = self._compute_stride(original_fps, fps)

        all_cls_features = []
        all_dense_features = [] if save_dense else None

        def flush(batch):
            if save_dense:
                cls_features, dense_features = self.extract_frame_features(batch, return_dense=True)
                all_cls_features.append(cls_features)
                # Cast to fp16 at accumulation time: the on-disk dense file is fp16
                # anyway (np.save below), and the fp32 accumulator was the cause of
                # the host-RAM OOM on long SoccerNet halves (~13.5 GiB at 45 min /
                # 5 FPS, doubled again by np.concatenate). Halves steady-state RSS.
                all_dense_features.append(dense_features.astype(np.float16))
            else:
                cls_features = self.extract_frame_features(batch)
                all_cls_features.append(cls_features)

        if profile_efficiency and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        extraction_start = time.perf_counter()

        # Stream-decode: read and process one batch at a time so the raw-frame
        # buffer stays at O(batch_size) instead of O(num_sampled_frames). The
        # output lists accumulate compact CLS/dense tensors, not HxWx3 uint8
        # frames, so peak RSS is dominated by the dense accumulator (~6 GB for
        # a 45-min half at 5 FPS) rather than the raw decode (~37 GB).
        batch = []
        num_sampled = 0
        success, frame = cap.read()
        count = 0
        while success:
            if count % skip_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                batch.append(rgb_frame)
                if len(batch) == batch_size:
                    flush(batch)
                    num_sampled += len(batch)
                    batch = []
            success, frame = cap.read()
            count += 1

        cap.release()

        if batch:
            flush(batch)
            num_sampled += len(batch)

        if num_sampled == 0:
            print(f"⚠️  No frames extracted from: {video_path}")
            return

        if profile_efficiency and torch.cuda.is_available():
            torch.cuda.synchronize()
        extraction_end = time.perf_counter()
        end_to_end_time = extraction_end - video_start_time
        video_duration_sec = count / original_fps if original_fps > 0 else 0

        total_extraction_time = extraction_end - extraction_start

        if profile_efficiency:
            self._log_extraction_metrics(
                video_path=video_path,
                batch_size_label=str(batch_size),
                num_features=num_sampled,
                gpu_compute_sec=total_extraction_time,
                total_end_to_end_sec=end_to_end_time,
                video_duration_sec=video_duration_sec
            )

        # Save CLS features (used for training)
        if not skip_cls:
            cls_output = np.vstack(all_cls_features)
            np.savez_compressed(output_path, cls=cls_output)

        # Save dense patch features (for attentive probes / dense baselines).
        # Uncompressed .npy so consumers can use np.load(path, mmap_mode='r')
        # to slice windows without materialising the full (T, P, D) tensor.
        # Stored as fp16 — halves disk vs fp32, lossless for downstream probes.
        if save_dense:
            # Chunks are already fp16 (cast at accumulation time in flush()), so
            # concatenate without a further dtype change.
            dense_output = np.concatenate(all_dense_features, axis=0)
            np.save(dense_path, dense_output)

    def extract_frame_features(self, frames, return_dense=False):
        """
        Extract CLS (and optionally dense patch) features from a batch of RGB frames.

        Preprocessing (master spec): shortest_edge=256 → centre-crop 256×256
        → ImageNet-normalise. Aspect ratio preserved.

        Args:
            frames (list): List of numpy arrays [H, W, 3] in RGB format
            return_dense (bool): If True, also return dense patch features.

        Returns:
            cls_features (np.ndarray): [batch_size, feature_dim]
            dense_features (np.ndarray, optional): [batch_size, num_patches, feature_dim]
                Only returned when return_dense=True. Register tokens are dropped;
                num_patches = (256 / patch_size)**2 = 256 for ViT-L/16.
        """
        if self.padding_mode == "reflect":
            # Make each frame square via border-reflected padding, resize the
            # square to 256x256 in numpy, then let the processor only do the
            # ImageNet normalisation. No pixels are cropped away.
            squared = [self._square_with_reflect(f) for f in frames]
            resized = [
                cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA)
                for f in squared
            ]
            pil_images = [Image.fromarray(f) for f in resized]
            inputs = self.processor(
                images=pil_images,
                do_resize=False,
                do_center_crop=False,
                return_tensors="pt",
            ).to(self.device)
        else:
            pil_images = [Image.fromarray(frame) for frame in frames]
            inputs = self.processor(
                images=pil_images,
                size={"shortest_edge": 256},
                do_center_crop=True,
                crop_size={"height": 256, "width": 256},
                return_tensors="pt",
            ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            h = outputs.last_hidden_state            # (B, 1+R+P, D)
            cls_features = h[:, 0, :].cpu().numpy()

            if return_dense:
                patch_start = 1 + self.num_registers
                dense_features = h[:, patch_start:, :].cpu().numpy()
                return cls_features, dense_features

        return cls_features
