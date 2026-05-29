"""Overlay the attentive probe's cross-attention on the window it predicts on.

Usage:
    python -m analysis.attn_window <clip> <anchor_idx> [options]

Loads the trained probe (same path resolution as eval_temporal.py), runs the
pre-extracted features through the probe's self-attention stack, then captures
the cross-attention weights from the single learnable query token to the
W*16*16 = 2560 patch tokens. The per-frame 16x16 maps are upsampled to 256x256
and alpha-blended onto the contact-sheet tiles from analysis/show_window.

The script prints the predicted class and softmax probabilities, the
ground-truth class of the anchor frame, and writes a contact-sheet PNG
(and optional mp4) annotated with attention heatmaps.

The script needs the trained checkpoint AND the pre-extracted dense features
to exist on the machine you run it on. On Fox both live under
/cluster/work/projects/ec12/ec-eirikto/TACDEC/{models,features}/. Override
with --checkpoint and --features-dir if either is somewhere else.

Examples:
    # default DINOv3 large run, reflect padding
    python -m analysis.attn_window 3266_ag95e7qiyar9f 116

    # V-JEPA 2 large attentive probe
    python -m analysis.attn_window 3266_ag95e7qiyar9f 116 --backbone-type vjepa2 \\
        --model-suffix optimised
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Reuse show_frame / show_window helpers for frame rendering and the sheet.
from analysis import show_frame as _show_frame
from analysis.show_frame import REPO_ROOT, VIEWS, read_frame_rgb, resolve_video
from analysis.show_window import (
    DEFAULT_LABELS_DIR,
    annotate,
    load_frame_labels,
    make_contact_sheet,
    video_frame_count,
)

sys.path.insert(0, str(REPO_ROOT / "src"))
from window_protocol import select_source_frames
from config import ( 
    TACDEC_FEATURES, TACDEC_LABELS, TACDEC_MODELS, TACDEC_VIDEOS,
)
from data.temporal_loaders import (
    CLASS_NAMES, DINOv3DenseLoader, VJEPA2DenseLoader,
)

from models.dinov3.attentive_probe import DINOv3AttentiveProbe
from models.vjepa2.attentive_pooler import AttentiveClassifier

# --- Probe rebuild (mirrors eval_temporal.rebuild_probe) ---------------------

def _resolve_checkpoint(backbone_type: str, backbone_size: str, model_suffix: str,
                        override: Path | None) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"--checkpoint not found: {override}")
        return override
    backbone_id = f"{backbone_type}_{backbone_size[0]}"
    name = f"{backbone_id}_{model_suffix}"
    for p in (TACDEC_MODELS / backbone_id / f"best_attn_{name}.pth",
              TACDEC_MODELS / backbone_id / f"{name}.pth",
              TACDEC_MODELS / f"best_attn_{name}.pth",
              TACDEC_MODELS / f"{name}.pth"):
        if p.exists():
            return p
    raise FileNotFoundError(f"Checkpoint not found for {name} under {TACDEC_MODELS}")


def _build_probe(ckpt: dict):
    backbone = ckpt["backbone_type"]
    feat_dim = int(ckpt["feature_dim"])
    num_classes = int(ckpt["num_classes"])
    if backbone == "dinov3":
        m = DINOv3AttentiveProbe(
            in_dim=feat_dim, probe_dim=feat_dim, num_classes=num_classes,
            num_heads=int(ckpt.get("probe_num_heads", 16)),
            num_blocks=int(ckpt.get("probe_depth", 4)),
            t_size=int(ckpt["window_size"]),
            h_size=int(ckpt.get("patch_h", 16)),
            w_size=int(ckpt.get("patch_w", 16)),
        )
    elif backbone == "vjepa2":
        m = AttentiveClassifier(
            embed_dim=feat_dim,
            num_heads=int(ckpt.get("probe_num_heads", 16)),
            depth=int(ckpt.get("probe_depth", 4)),
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"unknown backbone in checkpoint: {backbone!r}")
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, backbone


# --- Forward + cross-attention capture ---------------------------------------

def _self_attn_blocks(model, backbone: str):
    """Return the list of self-attn blocks for the probe (backbone-aware).
    V-JEPA 2's AttentivePooler may have `blocks is None` when probe_depth == 1.
    """
    blocks = model.blocks if backbone == "dinov3" else model.pooler.blocks
    return list(blocks) if blocks is not None else []


def _recompute_block_attn_meanheads(attn_mod, x_norm: torch.Tensor) -> torch.Tensor:
    """Mirror Block.attn's softmax weights, averaged over heads. Returns (N, N).

    Works for both the V-JEPA 2 Attention and DINOv3 _RoPEAttention because
    both have .qkv, .num_heads, .scale and optionally .rope.
    """
    B, N, C = x_norm.shape
    H = attn_mod.num_heads
    Hd = C // H
    qkv = attn_mod.qkv(x_norm).reshape(B, N, 3, H, Hd).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv[0], qkv[1], qkv[2]
    rope = getattr(attn_mod, "rope", None)
    if rope is not None:
        q, k = rope(q, k)
    w = (q @ k.transpose(-2, -1)) * attn_mod.scale         # (B, H, N, N)
    w = w.softmax(dim=-1)
    return w.mean(dim=1).squeeze(0)                         # (N, N), batch=1


def _attention_rollout(sa_mats: list[torch.Tensor], ca: torch.Tensor,
                       discard_ratio: float = 0.0) -> torch.Tensor:
    """Abnar & Zuidema attention rollout combined with the cross-attn pooler.

    Parameters
    ----------
    sa_mats : list of (N, N) tensors, one per self-attn block, head-averaged.
    ca      : cross-attn weights (H, 1, N) from the pooler's query to tokens.
    discard_ratio : fraction in [0, 1) of the smallest attention values to zero
        out per block before adding identity. Common values 0.0..0.9.

    Returns
    -------
    relevance : (N,) per-token relevance.
    """
    if not sa_mats:
        return ca.mean(dim=0).squeeze(0)                   # cross-attn only
    device = sa_mats[0].device
    N = sa_mats[0].shape[-1]
    I = torch.eye(N, device=device)
    R = I.clone()
    for A in sa_mats:
        if discard_ratio > 0.0:
            flat = A.flatten()
            n_drop = int(flat.numel() * discard_ratio)
            if 0 < n_drop < flat.numel():
                thresh = flat.kthvalue(n_drop).values
                A = torch.where(A >= thresh, A, torch.zeros_like(A))
        A = 0.5 * A + 0.5 * I
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        R = A @ R
    ca_mean = ca.mean(dim=0)                                # (1, N)
    return (ca_mean @ R).squeeze(0)                         # (N,)


@torch.no_grad()
def _forward_with_xattn(model, feats: torch.Tensor, backbone: str,
                        capture_self_attn: bool = False):
    """Run the probe forward and return (logits, cross_attn_weights[, sa_mats]).

    cross_attn_weights has shape (num_heads, N) where N = T*H*W (= 2560 for
    the default 5 fps W=10 16x16 protocol). Captured BEFORE head averaging.
    When capture_self_attn=True also returns a list of (N, N) head-averaged
    self-attention matrices, one per block, in shallow-to-deep order.
    """
    sa_inputs: list[tuple[int, torch.Tensor]] = []
    handles: list = []
    if capture_self_attn:
        for i, blk in enumerate(_self_attn_blocks(model, backbone)):
            def _hook(_mod, inputs, _out, _i=i):
                sa_inputs.append((_i, inputs[0].detach()))
            handles.append(blk.attn.register_forward_hook(_hook))
    B, N, _ = feats.shape
    if backbone == "dinov3":
        # Mirror DINOv3AttentiveProbe.forward up to the cross-attn pool.
        x = model.input_proj(feats)
        x = x + model.pos_embed.to(x.dtype).unsqueeze(0)
        for blk in model.blocks:
            x = blk(x)
        block = model.cross_attn_block
        q_in = model.query_token.expand(B, -1, -1)
    elif backbone == "vjepa2":
        # AttentiveClassifier wraps an AttentivePooler; we walk its parts.
        # Note: AttentivePooler builds `depth - 1` self-attn blocks (and
        # `pooler.blocks` is None if depth == 1), then a cross-attn pooler
        # with a learnable query (same convention as DINOv3).
        pooler = model.pooler
        x = feats
        if pooler.blocks is not None:
            for blk in pooler.blocks:
                x = blk(x)
        block = pooler.cross_attention_block
        q_in = pooler.query_tokens.expand(B, -1, -1)
    else:
        raise ValueError(backbone)

    # Manual cross-attention (mirrors CrossAttention.forward but exposes weights).
    xattn = block.xattn
    H = xattn.num_heads
    D = q_in.shape[-1]
    scale = xattn.scale

    x_norm = block.norm1(x)                                # (B, N, D)
    q = xattn.q(q_in).reshape(B, 1, H, D // H).permute(0, 2, 1, 3)        # (B, H, 1, d)
    kv = xattn.kv(x_norm).reshape(B, N, 2, H, D // H).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]                                                    # (B, H, N, d)

    attn = (q @ k.transpose(-2, -1)) * scale                              # (B, H, 1, N)
    attn = attn.softmax(dim=-1)
    out = attn @ v                                                         # (B, H, 1, d)
    out = out.transpose(1, 2).reshape(B, 1, D)

    q_after = q_in + out
    q_after = q_after + block.mlp(block.norm2(q_after))
    # Classifier head differs: DINOv3 names it `.classifier`, V-JEPA 2 `.linear`.
    head = model.classifier if backbone == "dinov3" else model.linear
    logits = head(q_after.squeeze(1))

    # cross-attn weights (H, N) for the single sample in this batch
    ca_weights = attn.squeeze(0).squeeze(1)

    sa_mats: list[torch.Tensor] | None = None
    if capture_self_attn:
        # Recompute head-averaged (N, N) self-attn per block from captured inputs.
        # Hooks fire in block order; we sort by the captured index to be safe.
        blocks = _self_attn_blocks(model, backbone)
        sa_mats = []
        for idx, x_in in sorted(sa_inputs, key=lambda t: t[0]):
            sa_mats.append(_recompute_block_attn_meanheads(blocks[idx].attn, x_in))
    for h in handles:
        h.remove()

    if capture_self_attn:
        return logits, ca_weights, sa_mats
    return logits, ca_weights


# --- Heatmap overlay ---------------------------------------------------------

def _overlay_heatmap(tile_bgr: np.ndarray, attn_2d: np.ndarray, vmin: float,
                     vmax: float, alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend a normalized 2D attention map onto a BGR tile.
    vmin/vmax fix the colour scale across all tiles in the window."""
    h, w = tile_bgr.shape[:2]
    a = (attn_2d - vmin) / max(vmax - vmin, 1e-9)
    a = np.clip(a, 0.0, 1.0)
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)
    cmap = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(cmap, alpha, tile_bgr, 1.0 - alpha, 0.0)


# --- Main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", help="clip id or path to .mp4")
    ap.add_argument("anchor_idx", type=int,
                    help="centre-frame index in the 5 FPS grid (CSV frame_idx)")
    ap.add_argument("model", nargs="?", default=None,
                    help="optional: model suffix (e.g. 'optimised') OR a full "
                         ".pth path. If it ends with .pth or contains a path "
                         "separator it's treated as a checkpoint path, "
                         "otherwise as a model suffix. Overrides --model-suffix "
                         "/ --checkpoint when given.")
    ap.add_argument("--backbone-type", choices=["dinov3", "vjepa2"], default="dinov3")
    ap.add_argument("--backbone-size", default="large",
                    choices=["base", "large", "huge", "giant"])
    ap.add_argument("--model-suffix", default="optimised",
                    help="suffix used at training time (default: optimised)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="explicit .pth path (overrides the default resolver)")
    ap.add_argument("--features-dir", type=Path, default=None,
                    help="dense features dir (default: TACDEC_FEATURES/<backbone>_<size>)")
    ap.add_argument("--video-dir", type=Path, default=None,
                    help="directory containing the source .mp4 files; clip id "
                         "is resolved against this dir. Defaults to "
                         "<repo>/data/TACDEC/videos/. Ignored if `clip` is "
                         "already a full .mp4 path.")
    ap.add_argument("--labels-dir", type=Path, default=None,
                    help="directory containing TACDEC label JSONs. Defaults "
                         "to config.TACDEC_LABELS when it exists (Fox), else "
                         "to <repo>/data/TACDEC/labels/ (laptop).")
    ap.add_argument("--padding-mode", choices=["center_crop", "reflect"],
                    default="reflect",
                    help="must match the extraction (default: reflect, matches the run)")
    ap.add_argument("--target-fps", type=float, default=5.0)
    ap.add_argument("--source-fps", type=float, default=25.0)
    ap.add_argument("--window-size", type=int, default=10)
    ap.add_argument("--view", choices=list(VIEWS), default="reflect",
                    help="frame view for the contact sheet (default: reflect)")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="heatmap blend strength on each tile (default: 0.45)")
    ap.add_argument("--no-overlay", action="store_true",
                    help="write the bare attention maps (no frame underneath)")
    ap.add_argument("--rollout", action="store_true",
                    help="combine the cross-attn weights with Abnar & Zuidema "
                         "rollout over the self-attn blocks (more localised "
                         "and more defensible than cross-attn alone)")
    ap.add_argument("--rollout-discard-ratio", type=float, default=0.0,
                    help="if --rollout, zero out the smallest fraction of "
                         "per-block self-attn weights before adding identity "
                         "(common values 0.0..0.9; default 0.0)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--also-anchor", action="store_true",
                    help="additionally write the single anchor (centre) tile as "
                         "<stem>_anchor.png, suitable as a standalone thesis "
                         "figure beside the window contact sheet.")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/attn_windows"),
                    help="where to write the PNG / MP4 (default: "
                         "results/attn_windows, relative to CWD — matches "
                         "the run_*.sh convention; created if missing)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.source_fps % args.target_fps != 0:
        ap.error("source-fps must be an integer multiple of target-fps")
    stride = int(args.source_fps // args.target_fps)

    # Where clip ids resolve to .mp4 files. Precedence:
    #   1. --video-dir (explicit override)
    #   2. config.TACDEC_VIDEOS if it exists on this machine (Fox)
    #   3. show_frame.DEFAULT_VIDEO_DIR = <repo>/data/TACDEC/videos (laptop)
    # resolve_video reads DEFAULT_VIDEO_DIR at call time.
    if args.video_dir is not None:
        _show_frame.DEFAULT_VIDEO_DIR = args.video_dir
    elif TACDEC_VIDEOS.exists():
        _show_frame.DEFAULT_VIDEO_DIR = TACDEC_VIDEOS

    # Same precedence for the per-frame label JSONs.
    if args.labels_dir is not None:
        labels_dir = args.labels_dir
    elif TACDEC_LABELS.exists():
        labels_dir = TACDEC_LABELS
    else:
        labels_dir = DEFAULT_LABELS_DIR

    # If the optional 3rd positional was given, route it to either --checkpoint
    # or --model-suffix based on whether it looks like a path.
    if args.model is not None:
        looks_like_path = args.model.endswith(".pth") or ("/" in args.model)
        if looks_like_path:
            args.checkpoint = Path(args.model)
        else:
            args.model_suffix = args.model

    # 1) Load checkpoint + rebuild probe
    ckpt_path = _resolve_checkpoint(args.backbone_type, args.backbone_size,
                                    args.model_suffix, args.checkpoint)
    print(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, backbone = _build_probe(ckpt)
    model.to(args.device)

    # 2) Build feature loader and pull the window's tokens
    features_dir = args.features_dir or (
        TACDEC_FEATURES / f"{args.backbone_type}_{args.backbone_size}")
    dense_tag = "reflect" if args.padding_mode == "reflect" else ""
    if backbone == "dinov3":
        loader = DINOv3DenseLoader(features_dir, args.target_fps, args.window_size,
                                   source_fps=args.source_fps, dense_tag=dense_tag)
    else:
        loader = VJEPA2DenseLoader(features_dir, args.target_fps, args.window_size,
                                   dense_tag=dense_tag)
    video = resolve_video(args.clip)
    clip_name = video.stem
    feats = loader.get_feature(clip_name, args.anchor_idx)         # (N, D)
    feats_t = torch.from_numpy(np.ascontiguousarray(feats)).unsqueeze(0).to(args.device)

    # 3) Forward + capture attention (cross-attn always; self-attn only for rollout)
    if args.rollout:
        logits, ca_weights, sa_mats = _forward_with_xattn(
            model, feats_t, backbone, capture_self_attn=True)
        relevance = _attention_rollout(sa_mats, ca_weights,
                                       discard_ratio=args.rollout_discard_ratio)
        attn = relevance.cpu().numpy()                                # (N,)
    else:
        logits, ca_weights = _forward_with_xattn(model, feats_t, backbone)
        attn = ca_weights.mean(dim=0).cpu().numpy()                   # (N,)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred = int(probs.argmax())

    # 4) Reshape (T, H, W). For DINOv3 we know patch_h/w from the checkpoint.
    if backbone == "dinov3":
        T = int(ckpt["window_size"])
        H_p = int(ckpt.get("patch_h", 16))
        W_p = int(ckpt.get("patch_w", 16))
    else:
        # V-JEPA 2 uses tubelet=2, so T_token = W//2; default spatial 16x16.
        T = int(ckpt["window_size"]) // 2
        H_p = int(ckpt.get("patch_h", 16))
        W_p = int(ckpt.get("patch_w", 16))
    expected = T * H_p * W_p
    if attn.shape[0] != expected:
        raise RuntimeError(
            f"attention length {attn.shape[0]} != T*H*W = {expected} "
            f"(T={T}, H={H_p}, W={W_p}); check window-size / patch dims.")
    attn_map = attn.reshape(T, H_p, W_p)
    vmin, vmax = float(attn.min()), float(attn.max())

    # 5) Render the 10 source frames using show_window helpers
    n_src = video_frame_count(video)
    src_frames = select_source_frames(
        args.anchor_idx, n_src,
        anchor_stride=stride, intra_window_stride=stride,
        window_length=args.window_size, boundary="clamp",
    )
    centre_src = args.anchor_idx * stride
    classify = None if args.no_labels else load_frame_labels(clip_name, labels_dir)
    view_fn = VIEWS[args.view]

    tiles, raw_tiles = [], []
    for k, f in enumerate(src_frames):
        rgb = read_frame_rgb(video, f)
        bgr = cv2.cvtColor(view_fn(rgb), cv2.COLOR_RGB2BGR)

        # Map source-frame index to attention frame index. DINOv3: 1:1.
        # V-JEPA 2 tubelet=2 collapses pairs -> use k // 2 for the W=10 case.
        attn_idx = k if backbone == "dinov3" else min(k // 2, T - 1)
        if args.no_overlay:
            blended = _overlay_heatmap(np.zeros_like(bgr), attn_map[attn_idx],
                                       vmin, vmax, alpha=1.0)
        else:
            blended = _overlay_heatmap(bgr, attn_map[attn_idx], vmin, vmax,
                                       alpha=args.alpha)
        label = classify(f) if classify else None
        annotated = annotate(blended, f, f / args.source_fps, label, f == centre_src)
        tiles.append(annotated)
        raw_tiles.append(annotated.copy())   # mp4 carries the same annotations

    # 6) Sheet + mp4
    # Per-pipeline subfolder so DINOv3 and V-JEPA 2 outputs stay separate
    # (e.g. results/attn_windows/dinov3_l/, results/attn_windows/vjepa2_l/).
    backbone_id = f"{args.backbone_type}_{args.backbone_size[0]}"
    out_dir = args.out_dir / backbone_id
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "rollout" if args.rollout else "xattn"
    stem = (f"{clip_name}_a{args.anchor_idx}_{args.view}_attn_{mode}"
            f"_{backbone_id}_{args.model_suffix}")
    sheet_path = out_dir / f"{stem}.png"
    cv2.imwrite(str(sheet_path), make_contact_sheet(tiles, cols=min(args.cols, len(tiles))))
    outputs = [sheet_path]
    if args.also_anchor:
        anchor_k = next((k for k, f in enumerate(src_frames) if f == centre_src), None)
        if anchor_k is None:
            raise RuntimeError(f"could not locate anchor frame {centre_src} in "
                               f"window {src_frames}")
        anchor_path = out_dir / f"{stem}_anchor.png"
        cv2.imwrite(str(anchor_path), tiles[anchor_k])
        outputs.append(anchor_path)
    if not args.no_video:
        h, w = raw_tiles[0].shape[:2]
        mp4_path = out_dir / f"{stem}.mp4"
        vw = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.target_fps, (w, h))
        for bgr in raw_tiles:
            vw.write(bgr)
        vw.release()
        outputs.append(mp4_path)

    # 7) Summary
    span = (src_frames[-1] - src_frames[0]) / args.source_fps
    true_label = classify(centre_src) if classify else "?"
    print(f"clip={clip_name} anchor={args.anchor_idx} (src centre={centre_src}, "
          f"t={centre_src / args.source_fps:.2f}s)")
    print(f"true (anchor) class = {true_label}")
    print(f"predicted class     = {CLASS_NAMES[pred]}  (probs: "
          + ", ".join(f"{CLASS_NAMES[i]}={probs[i]:.3f}" for i in range(len(probs))) + ")")
    print(f"window source frames ({args.window_size}): {src_frames}")
    mode_str = (f"rollout (discard={args.rollout_discard_ratio:.2f})"
                if args.rollout else "cross-attn (pooler only)")
    print(f"attn map shape per frame: {H_p}x{W_p}   span={span:.2f}s   "
          f"view={args.view}   mode={mode_str}")
    for p in outputs:
        print(f"  -> {p}")
    if not args.no_open and sys.platform == "darwin":
        for p in outputs:
            subprocess.run(["open", str(p)], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
