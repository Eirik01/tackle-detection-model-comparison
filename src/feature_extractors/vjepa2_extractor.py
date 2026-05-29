"""
V-JEPA2 Feature Extractor
Implementation for Meta's V-JEPA2 (Video Joint-Embedding Predictive Architecture).

V-JEPA2 is a self-supervised video encoder that processes temporal sequences.
Reference: https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
"""

import torch
import numpy as np
import cv2
import time
from pathlib import Path
from transformers import AutoModel, AutoVideoProcessor
from tqdm import tqdm
from .base_extractor import BaseFeatureExtractor
from .yolo_crop_extractor import YOLOCropExtractor


class VJEPA2Extractor(BaseFeatureExtractor):
    """
    V-JEPA2-based feature extractor for video frames.
    
    V-JEPA2 is designed for video understanding and processes temporal sequences.
    Uses 16-frame sliding windows (0.64s at 25 FPS) to leverage V-JEPA2's temporal 
    pretraining, providing richer temporal context than single-frame extraction.
    
    Window extraction strategy:
    - Reads all frames from video at original FPS
    - Creates 16-frame sliding windows centered on each output frame
    - Pads with zero frames at video boundaries for alignment
    - Stride is auto-calculated to match target output FPS
    
    Args:
        input_dir (str/Path): Directory containing video files
        output_dir (str/Path): Directory to save extracted features
        model_size (str): 'large', 'huge', or 'giant'
        device (str): 'cuda' or 'cpu'
    """
    
    def __init__(self, input_dir, output_dir, model_size="large", device="cuda", crop_dir=None,
                 feature_type="cls", padding_mode="center_crop"):
        self.model_size = model_size.lower()

        if self.model_size not in ["large", "huge", "giant"]:
            raise ValueError(f"model_size must be 'large', 'huge', or 'giant', got '{model_size}'")

        if feature_type not in ("cls", "dense"):
            raise ValueError(f"feature_type must be 'cls' or 'dense', got '{feature_type}'")
        self.feature_type = feature_type

        if padding_mode not in ("center_crop", "reflect"):
            raise ValueError(
                f"padding_mode must be 'center_crop' or 'reflect', got '{padding_mode}'"
            )
        self.padding_mode = padding_mode

        self.crop_dir = Path(crop_dir) if crop_dir else None

        # Initialize parent class
        super().__init__(input_dir, output_dir, device)
        
        # Load model
        self.load_model()
    
    def get_model_name(self):
        """Returns 'vjepa2_b' or 'vjepa2_l'"""
        return f"vjepa2_{self.model_size[0]}"
    
    def load_model(self):
        """
        Load V-JEPA2 model from Hugging Face.
        
        V-JEPA2 uses AutoVideoProcessor for video preprocessing.
        All V-JEPA2 models output 1024-dim features regardless of size.
        """
        # Model mapping
        model_mapping = {
            "large": "facebook/vjepa2-vitl-fpc64-256",
            "huge": "facebook/vjepa2-vith-fpc64-256",
            "giant": "facebook/vjepa2-vitg-fpc64-256"
        }
        
        model_name = model_mapping[self.model_size]
        
        print(f"🔄 Loading {model_name}...")
        
        # Load video processor
        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_name,
        ).to(self.device)  # Manual device placement
        
        self.model.eval()
        
        # Store model config info
        self.feature_dim = self.model.config.hidden_size
        self.frames_per_clip = self.model.config.frames_per_clip
        
        print(f"✅ Model loaded successfully")
        print(f"   Feature dimension: {self.feature_dim}")
        if self.padding_mode == "reflect":
            print(f"   Preprocessing: reflect-pad to square -> resize 256x256 "
                  f"(matches DINOv3 reflect mode, applied per-call)")
        else:
            print(f"   Preprocessing: shortest_edge=256 -> center-crop 256x256 "
                  f"(matches DINOv3, applied per-call)")
        print(f"   Frames per clip (config): {self.frames_per_clip}")
    
    
    def _apply_crop(self, rgb_frame, x1, y1, x2, y2):
        """Crop frame region and zero-pad to square, preserving aspect ratio."""
        crop = rgb_frame[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return rgb_frame
        size = max(h, w)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        dy, dx = (size - h) // 2, (size - w) // 2
        padded[dy:dy + h, dx:dx + w] = crop
        return padded

    def _process_video_with_windows(self, video_path, output_path, target_fps=2.0,
                                    window_size=16, stride=None,
                                    intra_window_stride=None,
                                    profile_efficiency=False, crops=None):
        """
        Process video using sliding temporal windows (V-JEPA2's strength).

        Strategy:
        - Read all frames at original FPS.
        - For each anchor (one V-JEPA2 forward), pick ``window_size`` source
          frames at ``intra_window_stride`` between them. Anchors step by
          ``stride`` source frames between successive forwards.
        - Frame indices come from the shared ``window_protocol`` so DINOv3
          and V-JEPA2 see byte-identical source frames at every anchor.

        Args:
            video_path (Path): Path to video file
            output_path (Path): Where to save features
            target_fps (float): Target output FPS / anchor rate (default: 2.0)
            window_size (int): Number of frames per V-JEPA2 forward (default: 16)
            stride (int): Anchor stride in source frames between windows. If
                None, auto-computed from target_fps via _compute_stride.
            intra_window_stride (int): Stride between adjacent frames *inside*
                one V-JEPA2 input clip. If None, falls back to the anchor
                ``stride`` (Claude Desktop spec: anchor and intra strides are
                the same so each output tick covers one frame at target FPS).
                Set to 1 explicitly for the legacy "consecutive raw frames"
                behaviour.
        """
        
        video_start_time = time.perf_counter()
        
        # Load video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Failed to open: {video_path}")
            return
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate stride to match target_fps directly (no subsampling needed)
        if stride is None:
            stride = self._compute_stride(original_fps, target_fps)
        # Default intra-window stride to the anchor stride. Old behaviour
        # (consecutive raw frames) is opt-in via intra_window_stride=1.
        if intra_window_stride is None:
            intra_window_stride = stride

        window_span_src = (window_size - 1) * intra_window_stride
        print(f"   Original: {width}×{height} @ {original_fps:.1f} FPS")
        print(f"   Target FPS: {target_fps:.1f}")
        print(f"   Window size: {window_size} frames at intra_stride={intra_window_stride} "
              f"-> spans {window_span_src} source frames "
              f"({window_span_src / max(original_fps, 1e-9):.2f}s)")
        print(f"   Anchor stride: {stride} frames "
              f"({stride/original_fps:.3f}s between forwards = "
              f"{original_fps / max(stride, 1):.2f} forwards/sec)")
        
        # Stream the source frames instead of buffering the whole video. A
        # 45-min broadcast half at 25 FPS is ~67,500 frames * 1280*720*3 bytes
        # ~= 185 GiB — way over any reasonable Slurm allocation, even though
        # the per-anchor work only needs ~(W-1)*intra_stride + 1 frames
        # (~46 frames @ default W=10/stride=5, ~140 MiB).
        #
        # n_source_frames comes from cv2's stream metadata; on SoccerNet MKVs
        # and TACDEC MP4s it matches a full sequential read exactly. We use it
        # up front to compute valid_anchor_range without having to materialise
        # the frames first.
        n_source_frames = int(total_frames)
        if n_source_frames <= 0:
            print(f"⚠️  Could not determine frame count from {video_path}")
            cap.release()
            return
        if n_source_frames < window_size:
            print(f"⚠️  Video shorter than window size ({n_source_frames} frames, "
                  f"window={window_size}). No valid anchors; skipping.")
            cap.release()
            return

        # Extract features with sliding windows.
        #
        # For feature_type=="dense" we stream rows directly into a memmap'd
        # .npy on disk via np.lib.format.open_memmap. This caps RSS at the
        # streaming source-frame cache (~140 MiB) + the model, instead of
        # accumulating an in-RAM array that scales with sequence length
        # (~35 GiB at 13.5k windows × 1280 tokens × 1024 dim × fp16). For
        # feature_type=="cls" the per-row payload is tiny (4 KB) so the old
        # in-RAM list is still fine.
        all_features = [] if self.feature_type != "dense" else None
        dense_mmap = None
        dense_path = None
        meta_path = None

        extraction_start = time.perf_counter()
        if profile_efficiency and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Build windows via the shared protocol (one source-of-truth function;
        # see src/window_protocol.py). Kassab-style boundary handling: emit
        # only anchors whose window fits entirely inside the video. No zero-
        # padding, no edge-replication. Matches TempTAC.ipynb cell 15
        # (`len(y) - W + 1` windows).
        from ..window_protocol import select_source_frames, valid_anchor_range

        valid_lo, valid_hi = valid_anchor_range(
            video_length=n_source_frames,
            anchor_stride=stride,
            intra_window_stride=intra_window_stride,
            window_length=window_size,
        )
        if valid_hi < valid_lo:
            print(f"   ⚠️  Video has {n_source_frames} frames but window span = "
                  f"{(window_size - 1) * intra_window_stride + 1} source frames; "
                  "no valid anchors. Skipping.")
            cap.release()
            return
        n_valid = valid_hi - valid_lo + 1
        print(f"   Valid anchor range: [{valid_lo}, {valid_hi}]  "
              f"({n_valid} windows; dropped {valid_lo + (((n_source_frames + stride - 1) // stride) - 1 - valid_hi)} boundary anchors)")

        # Allocate the streaming dense output up front: a memmap'd .npy of
        # the full shape, fp16 on disk. Each window writes one row through
        # the OS page cache; no accumulator. The output path is split into
        # a `.npy` for the dense bytes and a `.meta.npz` sidecar for the
        # protocol metadata (loader prefers this new layout, falls back to
        # the legacy single-.npz path for older extractions).
        if self.feature_type == "dense":
            tokens_per_window = (window_size // 2) * 16 * 16
            dense_path = output_path.with_suffix(".npy")
            meta_path = output_path.parent / (output_path.stem + ".meta.npz")
            print(f"   Streaming dense rows to {dense_path.name}  "
                  f"(shape=({n_valid}, {tokens_per_window}, {self.feature_dim}), "
                  f"fp16, ~{n_valid * tokens_per_window * self.feature_dim * 2 / 1e9:.1f} GB)")
            dense_mmap = np.lib.format.open_memmap(
                dense_path,
                mode="w+",
                dtype=np.float16,
                shape=(n_valid, tokens_per_window, self.feature_dim),
            )

        # Rolling source-frame cache. Frames are read sequentially via cap.read()
        # and dropped as soon as the next pending anchor no longer needs them.
        frame_cache: dict[int, np.ndarray] = {}
        next_source_idx = 0  # next index cap.read() will return

        def ensure_frames_up_to(target_idx: int) -> bool:
            """Read frames from cap until frame_cache contains target_idx.
            Returns False on EOF before target_idx is reached."""
            nonlocal next_source_idx
            while next_source_idx <= target_idx:
                ok, bgr = cap.read()
                if not ok:
                    return False
                frame_cache[next_source_idx] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                next_source_idx += 1
            return True

        for out_row, anchor_idx in enumerate(range(valid_lo, valid_hi + 1)):
            # Determine crop for this window (from center frame's crop entry).
            # Crop list is anchor-aligned: index by anchor_idx, not by output row.
            crop_box = None
            if crops is not None and anchor_idx < len(crops):
                x1, y1, x2, y2 = crops[anchor_idx]
                if x1 >= 0:
                    crop_box = (x1, y1, x2, y2)

            # 'clamp' is a safety net; valid anchors never trigger it.
            indices = select_source_frames(
                anchor_idx=anchor_idx,
                video_length=n_source_frames,
                anchor_stride=stride,
                intra_window_stride=intra_window_stride,
                window_length=window_size,
                boundary="clamp",
            )
            # Read up to the latest source frame this anchor needs. If the
            # stream ends short, truncate valid_hi so the .npz reflects only
            # the anchors we could actually score.
            if not ensure_frames_up_to(max(indices)):
                truncated_hi = anchor_idx - 1
                if truncated_hi < valid_lo:
                    print(f"   ⚠️  EOF before any valid anchor could be built "
                          f"(read {next_source_idx} of declared {n_source_frames} frames). Skipping.")
                    cap.release()
                    return
                print(f"   ⚠️  Short read at source idx {next_source_idx} "
                      f"(declared {n_source_frames}); truncating valid_hi "
                      f"{valid_hi} -> {truncated_hi}.")
                valid_hi = truncated_hi
                n_valid = valid_hi - valid_lo + 1
                break

            window_frames = []
            for i in indices:
                frame = frame_cache[i]
                if crop_box is not None:
                    frame = self._apply_crop(frame, *crop_box)
                if self.padding_mode == "reflect":
                    # Square the frame with border-reflected padding before the
                    # processor sees it, then resize to 256x256 so no pixels
                    # are cropped along the shorter axis.
                    frame = self._square_with_reflect(frame)
                    frame = cv2.resize(frame, (256, 256),
                                       interpolation=cv2.INTER_AREA)
                window_frames.append(frame)

            # Evict frames that no anchor will need again.
            if anchor_idx + 1 <= valid_hi:
                next_min = min(select_source_frames(
                    anchor_idx=anchor_idx + 1,
                    video_length=n_source_frames,
                    anchor_stride=stride,
                    intra_window_stride=intra_window_stride,
                    window_length=window_size,
                    boundary="clamp",
                ))
                for k in [k for k in frame_cache if k < next_min]:
                    del frame_cache[k]
            else:
                frame_cache.clear()

            # Convert to tensor: [T, H, W, C] -> [T, C, H, W]
            window_tensor = torch.stack([
                torch.from_numpy(f).permute(2, 0, 1) for f in window_frames
            ])  # [16, C, H, W]

            # Process through V-JEPA2.
            # Default path matches DINOv3's preprocessing (`dinov3_extractor.py`):
            # resize shorter edge to 256, then centre-crop to 256x256. With
            # padding_mode="reflect" the frames are already 256x256 squares, so
            # we skip both processor-side resize and centre-crop.
            if self.padding_mode == "reflect":
                inputs = self.processor(
                    window_tensor,
                    do_resize=False,
                    do_center_crop=False,
                    return_tensors="pt",
                ).to(self.model.device)
            else:
                inputs = self.processor(
                    window_tensor,
                    size={"shortest_edge": 256},
                    do_center_crop=True,
                    crop_size={"height": 256, "width": 256},
                    return_tensors="pt",
                ).to(self.model.device)
            
            with torch.inference_mode():
                outputs = self.model(**inputs, skip_predictor=True)
                last_hidden_state = outputs.last_hidden_state  # [1, num_patches, feature_dim]

                if self.feature_type == "dense":
                    # Stream the full spatio-temporal token grid straight to
                    # the memmap'd .npy: write goes through OS page cache, so
                    # RSS stays flat regardless of clip length.
                    feat = last_hidden_state[0].cpu().to(torch.float16).numpy()
                    dense_mmap[out_row] = feat
                else:
                    # Mean pooling over all patch tokens -> [feature_dim]
                    feat = last_hidden_state[0].mean(dim=0).cpu().numpy()
                    all_features.append(feat)
        
        cap.release()
        frame_cache.clear()

        if profile_efficiency and torch.cuda.is_available():
            torch.cuda.synchronize()
        extraction_end = time.perf_counter()
        end_to_end_time = extraction_end - video_start_time
        video_duration_sec = n_source_frames / original_fps if original_fps > 0 else 0
        
        total_extraction_time = extraction_end - extraction_start
        
        num_features = n_valid if self.feature_type == "dense" else len(all_features)
        if profile_efficiency:
            self._log_extraction_metrics(
                video_path=video_path,
                batch_size_label=f"1x{window_size}",
                num_features=num_features,
                gpu_compute_sec=total_extraction_time,
                total_end_to_end_sec=end_to_end_time,
                video_duration_sec=video_duration_sec
            )

        if self.feature_type == "dense":
            # Flush the memmap and persist the protocol metadata sidecar. The
            # `.npy` already holds the full (n_valid, tokens, D) fp16 grid in
            # the canonical row order [valid_lo, valid_hi]. Loader maps anchor
            # -> row via row = anchor - valid_lo and prefers this layout over
            # the legacy single-.npz path.
            dense_mmap.flush()
            del dense_mmap  # closes the underlying mmap
            np.savez(
                meta_path,
                valid_lo=np.int32(valid_lo),
                valid_hi=np.int32(valid_hi),
                anchor_stride=np.int32(stride),
                intra_window_stride=np.int32(intra_window_stride),
                window_length=np.int32(window_size),
                n_source_frames=np.int32(n_source_frames),
            )
            # Best-effort: clear the now-stale legacy .npz so the loader does
            # not see two candidates for the same video on re-runs.
            if output_path.exists():
                output_path.unlink()
            print(f"   ✓ Extracted {n_valid} dense features at {target_fps} FPS"
                  f"  (per-frame shape: ({(window_size // 2) * 16 * 16}, {self.feature_dim}), dtype=float16)")
            print(f"     dense -> {dense_path.name}")
            print(f"     meta  -> {meta_path.name}")
        else:
            # [Seq_Len, feature_dim] -- mean-pooled per-clip summary
            stacked = np.stack(all_features)
            np.savez_compressed(output_path, cls=stacked)
            print(f"   ✓ Extracted {len(stacked)} cls features at {target_fps} FPS"
                  f"  (per-frame shape: {stacked.shape[1:]}, dtype={stacked.dtype})")
    
    def extract_features(self, fps=2.0, batch_size=32, start_idx=None, end_idx=None,
                         override=False, profile_efficiency=False, crop_suffix="yolo",
                         window_size=16, stride=None, intra_window_stride=None):
        """
        Override base class to use sliding window extraction.

        Args:
            fps (float): Target FPS for output features (default: 2.0).
            window_size (int): Number of raw frames per V-JEPA2 forward (default 16).
                For the patch-token attentive probe protocol set this to W (e.g. 50).
            stride (int): Stride between successive windows in raw frames. If None,
                auto-computed as max(1, original_fps / target_fps). Set stride=1 for
                stride-1 sliding feature extraction (one V-JEPA2 forward per frame).
            batch_size (int): Not used for V-JEPA2 (processes windows individually)
            start_idx (int): Start index for video processing (inclusive, 0-based)
            end_idx (int): End index for video processing (exclusive, 0-based)
            override (bool): If True, re-extract and overwrite existing feature files
            profile_efficiency (bool): If True, measure end-to-end and GPU throughput
            crop_suffix (str): Suffix added to filename when crops are used
        """
        videos = self._get_all_videos()
        total_videos = len(videos)

        # Handle index-based slicing
        if start_idx is not None or end_idx is not None:
            start = start_idx if start_idx is not None else 0
            end = end_idx if end_idx is not None else total_videos
            videos = videos[start:end]
            print(f"🎯 Processing videos [{start}:{end}] of {total_videos} total")

        print(f"📹 Found {len(videos)} video files to process")
        print(f"🎬 Using {window_size}-frame sliding windows for temporal context"
              + ("" if stride is None else f" (stride={stride})"))

        if len(videos) == 0:
            print("⚠️  No videos found! Check input directory.")
            return

        # Disk-budget heads-up for non-default dense extraction. Token math:
        # tubelet 2x16x16 over 256x256 input -> (W/2) * 16*16 = W*128 tokens.
        # fp16 -> 2 bytes/elem. So per-window bytes = W * 128 * feature_dim * 2.
        if self.feature_type == "dense" and window_size != 16:
            tokens_per_window = (window_size // 2) * 16 * 16
            mb_per_window = tokens_per_window * self.feature_dim * 2 / 1e6
            print(f"⚠️  Non-default window_size={window_size} with feature_type=dense "
                  f"-- output files will be ~{mb_per_window:.1f} MB per window @ fp16 "
                  f"({tokens_per_window} tokens × {self.feature_dim} dim). Confirm storage budget.")

        for video_path in tqdm(videos, desc="Processing videos"):
            suffix = f"_{crop_suffix}" if self.crop_dir else ""
            kind_suffix = "_dense" if self.feature_type == "dense" else "_features"
            # Tag dense files with the protocol params so different runs don't collide.
            # Include intra-window stride when set explicitly (it's distinct from
            # anchor stride only when the new shared-protocol path is used).
            if self.feature_type == "dense" and (window_size != 16
                                                  or stride is not None
                                                  or intra_window_stride is not None):
                stride_tag = f"_s{stride}" if stride is not None else ""
                iw_tag = (f"_iw{intra_window_stride}"
                          if intra_window_stride is not None else "")
                kind_suffix = f"_dense_w{window_size}{stride_tag}{iw_tag}"
            pad_tag = "_reflect" if self.padding_mode == "reflect" else ""
            output_filename = f"{video_path.stem}_{self.get_model_name()}_{fps}fps{suffix}{pad_tag}{kind_suffix}.npz"
            output_path = self.output_dir / output_filename

            if output_path.exists() and not override:
                continue

            crops = None
            if self.crop_dir:
                crop_path = YOLOCropExtractor.get_crop_path(self.crop_dir, video_path.stem, fps)
                if crop_path.exists():
                    crops = YOLOCropExtractor.load_crops(crop_path)
                else:
                    print(f"  WARNING: no crop file found for {video_path.stem}, using full frame")

            self._process_video_with_windows(
                video_path, output_path, target_fps=fps,
                window_size=window_size, stride=stride,
                intra_window_stride=intra_window_stride,
                profile_efficiency=profile_efficiency, crops=crops,
            )


if __name__ == "__main__":
    """
    Standalone test for V-JEPA2 extractor.
    """
    from config import TACDEC_VIDEOS, TACDEC_FEATURES
    
    print("="*60)
    print("Testing V-JEPA2 Feature Extractor")
    print("="*60)
    
    extractor = VJEPA2Extractor(
        input_dir=TACDEC_VIDEOS,
        output_dir=TACDEC_FEATURES,
        model_size="large",
        device="cuda"
    )
    
    print(f"\nModel: {extractor.get_model_name()}")
    print(f"Feature dimension: {extractor.get_feature_dim()}")
    print("\n✅ V-JEPA2 extractor initialized successfully!")

