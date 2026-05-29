"""
DINOv3 video-classification attentive probe (Sec. 6.1.6 / App. D.6).

Pipeline
--------
1. Discard the global CLS token; keep dense (T x H_p x W_p) patch features.
2. Linearly project per-patch features to ``probe_dim`` (1024 in the paper).
3. Add additive 3D sin-cos positional embeddings (Vaswani et al., 2017).
4. Pass the sequence through ``num_blocks`` self-attention blocks. Each block
   applies 3D factorized RoPE to its Q and K projections; spatial coords are
   rotated by a per-head, *fixed* random angle so different heads see relative
   position from different orientations.
5. Aggregate with a single cross-attention block driven by one position-less
   learnable query token.
6. Apply a final Linear projection to obtain class logits.

For ``num_blocks=4`` (the DINOv3 paper default), the total depth is 4
self-attn blocks + 1 cross-attn block = 5 layers, exactly as App. D.6
describes: "four self-attention blocks ... After the four blocks, we apply
a cross-attention block".

Methodology
-----------
This file is paired with the V-JEPA 2 attentive probe (used unmodified for
comparison) on TACDEC. To make the two probes maximally comparable, we re-use
V-JEPA 2's ``Block``, ``CrossAttentionBlock``, ``MLP``, ``trunc_normal_``,
``get_3d_sincos_pos_embed`` and the per-block weight rescaling from V-JEPA 2's
``AttentivePooler`` *unchanged*. The pieces that the DINOv3 paper specifies
differently from V-JEPA 2 are localized in this file:

  * Linear input projection from backbone_dim to probe_dim (paper: 1024).
  * Additive 3D sin-cos positional embedding before the self-attn stack
    (V-JEPA 2's probe does not have this).
  * 3D factorized RoPE with per-head random spatial rotations applied inside
    each self-attn block (V-JEPA 2's RoPEAttention has 3D factorized RoPE
    but with fixed axis-aligned rotations).
  * Five-layer topology: 4 self-attn + 1 cross-attn (V-JEPA 2's
    ``AttentivePooler(depth=4)`` is 3 self-attn + 1 cross-attn = 4 layers).

Imports
-------
This file expects the V-JEPA 2 source tree on ``sys.path`` (e.g. via a git
submodule at ``vjepa2/`` plus ``PYTHONPATH=$PWD/vjepa2:$PYTHONPATH``):
    from src.models.utils.modules import Block, CrossAttentionBlock
    from src.models.utils.pos_embs import get_3d_sincos_pos_embed
    from src.utils.tensors import trunc_normal_
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- V-JEPA 2 building blocks (used UNCHANGED) -----------------------------
# Block:                    pre-norm self-attn block (we override its .attn
#                           below with our random-rotated 3D RoPE attention).
# CrossAttentionBlock:      pre-norm cross-attn + FFN, used for the pooler.
# trunc_normal_:            V-JEPA 2's truncated-normal init helper.
# get_3d_sincos_pos_embed:  the additive 3D sin-cos PE used by the V-JEPA 2
#                           backbone itself; we re-use it here so both probes
#                           share an identical positional-embedding helper.
from models.vjepa2.utils.modules import Block, CrossAttentionBlock
from models.vjepa2.utils.pos_embs import get_3d_sincos_pos_embed
from models.vjepa2.utils.tensors import trunc_normal_


# ===========================================================================
# Helpers
# ===========================================================================
def _split_three_even(total: int) -> Tuple[int, int, int]:
    """Split ``total`` into three positive even integers as evenly as possible.

    Used for the per-axis dim split in both the additive 3D sin-cos PE and
    the 3D factorized RoPE when no explicit split is provided.
    """
    if total < 6 or total % 2 != 0:
        raise ValueError(
            f"Cannot split {total} into three positive even integers."
        )
    if total % 6 == 0:
        d = total // 3
        return (d, d, d)
    # total % 6 in {2, 4}
    base = (total // 6) * 2                          # largest even <= total/3
    remainder = total - 3 * base
    if remainder == 2:
        return (base, base, base + 2)                # +2 to W
    return (base, base + 2, base + 2)                # +4 to H and W


def build_3d_sincos_pos_embed(
    embed_dim: int,
    t_size: int,
    h_size: int,
    w_size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    uniform_power: bool = True,
) -> torch.Tensor:
    """Additive 3D (T, H, W) sin-cos positional embedding.

    Thin wrapper around V-JEPA 2's ``get_3d_sincos_pos_embed`` (in
    ``src/models/utils/pos_embs.py``). We re-use the V-JEPA 2 helper verbatim
    so the additive PE is bit-identical to the one used by the V-JEPA 2
    backbone itself.

    With ``uniform_power=True`` (default), each axis gets
    ``ceil(embed_dim / 6) * 2`` channels, and the helper truncates the last
    block to ``embed_dim`` total. With ``uniform_power=False``, the split is
    ``(d_t = D/2, d_h = D/4, d_w = D/4)`` — favouring the temporal axis.
    The DINOv3 paper does not specify the split; we default to uniform_power
    since DINOv3 has no temporal bias built into its image backbone.

    Note: V-JEPA 2's helper requires ``h_size == w_size`` (one ``grid_size``
    argument). For non-square spatial grids you would need a different PE.

    Returns ``(T*H*W, embed_dim)`` torch.Tensor in (t, h, w) row-major order
    with w fastest, on ``device`` and cast to ``dtype``.
    """
    if h_size != w_size:
        raise ValueError(
            f"V-JEPA 2's get_3d_sincos_pos_embed assumes a square spatial grid; "
            f"got h_size={h_size} != w_size={w_size}."
        )
    pe_np = get_3d_sincos_pos_embed(
        embed_dim=embed_dim,
        grid_size=h_size,
        grid_depth=t_size,
        cls_token=False,
        uniform_power=uniform_power,
    )                                                          # (T*H*W, D), numpy
    pe = torch.from_numpy(pe_np).to(dtype=dtype)
    if device is not None:
        pe = pe.to(device)
    return pe


# ===========================================================================
# 3D factorized RoPE with per-head random spatial rotations
# ===========================================================================
class FactorizedRoPE3D(nn.Module):
    """3D factorized Rotary Position Embedding with per-head random spatial rotations.

    The per-head channel dim is split into three contiguous slices ``(d_t,
    d_h, d_w)``. Standard 1D RoPE (RoFormer, Su et al. 2024) is applied to
    each slice using the corresponding axis coordinate.

    Per-head random spatial rotation
    --------------------------------
    DINOv3 App. D.6: "3D factorized RoPE with random spatial rotations applied
    independently in each attention head".

    We interpret this as: at construction time, sample one random angle
    ``alpha_h ~ U[0, 2*pi)`` per head, register it as a fixed buffer (so it
    is identical at train and eval, and survives checkpointing). The 2D
    spatial coordinates ``(w, h)`` of every token are rotated by ``alpha_h``
    *before* being fed to the spatial RoPE; temporal coords are unchanged.
    Different heads therefore see relative spatial position from different
    orientations, breaking the strict axis-aligned bias of plain factorized
    RoPE.

    The cos/sin tables are precomputed at construction with shape
    ``(num_heads, T*H*W, head_dim)`` so every forward is just a fused gather +
    rotate-half multiply, matching the standard LLaMA / RoFormer pattern.

    Spatial coordinate normalization
    --------------------------------
    The DINOv3 backbone (App. C) places patch coordinates in a normalized
    box ``[-1, 1]`` and applies "RoPE-box jittering" by random rescaling.
    We follow the same convention here by dividing raw integer coords by
    ``max(H, W)`` so they live in ``[0, 1]``, paired with a smaller RoPE
    base of 100 (matched to the smaller coordinate range). Set
    ``normalize_spatial=False`` to use raw integer coords, in which case
    ``base=10000`` (RoFormer default) is more appropriate.

    Bug-fix vs V-JEPA 2
    -------------------
    V-JEPA 2's ``rotate_queries_or_keys`` has a documented bug (PR #15)
    where ``repeat`` duplicates rather than interleaves the cos/sin pairs.
    We use ``repeat_interleave``, which is the correct RoFormer convention.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        t_size: int,
        h_size: int,
        w_size: int,
        base: float = 100.0,
        axis_dims: Optional[Sequence[int]] = None,
        use_random_rotations: bool = True,
        seed: int = 0,
        normalize_spatial: bool = True,
    ):
        super().__init__()
        if axis_dims is None:
            axis_dims = _split_three_even(head_dim)
        if sum(axis_dims) != head_dim:
            raise ValueError(
                f"axis_dims {tuple(axis_dims)} must sum to head_dim={head_dim}."
            )
        if any(d % 2 != 0 or d <= 0 for d in axis_dims):
            raise ValueError(
                f"Each axis dim must be a positive even integer. Got {tuple(axis_dims)}."
            )

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.axis_dims = tuple(axis_dims)
        self.t_size, self.h_size, self.w_size = t_size, h_size, w_size
        self.use_random_rotations = use_random_rotations
        self.normalize_spatial = normalize_spatial

        # Sample (and freeze) per-head spatial rotation angles.
        gen = torch.Generator().manual_seed(seed)
        if use_random_rotations:
            alphas = torch.rand(num_heads, generator=gen) * (2 * math.pi)
        else:
            alphas = torch.zeros(num_heads)
        self.register_buffer("rotation_angles", alphas, persistent=True)

        cos, sin = self._build_cache(base)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    @staticmethod
    def _rope_cos_sin_continuous(
        dim: int, positions: torch.Tensor, base: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """1D RoPE cos/sin tables for arbitrary (real-valued) positions.

        Pairs channels (2k, 2k+1) so each pair shares one frequency, matching
        the standard rotate-half formula.
        """
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        angles = positions[:, None] * freqs[None, :]                    # (N, dim/2)
        cos = torch.repeat_interleave(angles.cos(), repeats=2, dim=-1)  # (N, dim)
        sin = torch.repeat_interleave(angles.sin(), repeats=2, dim=-1)
        return cos, sin

    def _build_cache(self, base: float) -> Tuple[torch.Tensor, torch.Tensor]:
        T, H, W = self.t_size, self.h_size, self.w_size
        N = T * H * W
        d_t, d_h, d_w = self.axis_dims

        # ---- Temporal RoPE: shared across heads --------------------------
        t_idx = torch.arange(T, dtype=torch.float32)
        cos_t_1d, sin_t_1d = self._rope_cos_sin_continuous(d_t, t_idx, base)

        # ---- Spatial coords (raw or [0,1]-normalized) --------------------
        h_idx = torch.arange(H, dtype=torch.float32)
        w_idx = torch.arange(W, dtype=torch.float32)
        if self.normalize_spatial:
            scale = float(max(H, W))
            h_idx = h_idx / scale
            w_idx = w_idx / scale
        grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing="ij")
        h_flat = grid_h.reshape(-1)
        w_flat = grid_w.reshape(-1)

        # ---- Per-head spatial RoPE with rotation -------------------------
        cos_caches, sin_caches = [], []
        for h in range(self.num_heads):
            alpha = float(self.rotation_angles[h])
            ca, sa = math.cos(alpha), math.sin(alpha)
            x_rot = w_flat * ca - h_flat * sa
            y_rot = w_flat * sa + h_flat * ca

            cos_h, sin_h = self._rope_cos_sin_continuous(d_h, y_rot, base)
            cos_w, sin_w = self._rope_cos_sin_continuous(d_w, x_rot, base)

            ct = cos_t_1d[:, None, :].expand(T, H * W, d_t)
            st = sin_t_1d[:, None, :].expand(T, H * W, d_t)
            ch = cos_h[None, :, :].expand(T, H * W, d_h)
            sh = sin_h[None, :, :].expand(T, H * W, d_h)
            cw = cos_w[None, :, :].expand(T, H * W, d_w)
            sw = sin_w[None, :, :].expand(T, H * W, d_w)

            cos_full = torch.cat([ct, ch, cw], dim=-1).reshape(N, self.head_dim)
            sin_full = torch.cat([st, sh, sw], dim=-1).reshape(N, self.head_dim)
            cos_caches.append(cos_full)
            sin_caches.append(sin_full)

        return torch.stack(cos_caches, dim=0), torch.stack(sin_caches, dim=0)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # Treats the last dim as pairs (a, b) -> (-b, a).
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply per-head factorized RoPE to query and key tensors.

        q, k : (B, num_heads, N, head_dim) where N == T*H*W.
        """
        B, H, N, _ = q.shape
        if H != self.num_heads:
            raise ValueError(
                f"Got tensor with {H} heads but RoPE was built for {self.num_heads}."
            )
        if N > self.cos_cache.shape[1]:
            raise ValueError(
                f"Sequence length N={N} exceeds RoPE cache "
                f"(T*H*W = {self.cos_cache.shape[1]})."
            )
        cos = self.cos_cache[:, :N].to(q.dtype)[None]                  # (1, H, N, Hd)
        sin = self.sin_cache[:, :N].to(q.dtype)[None]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ===========================================================================
# RoPE attention layer that plugs into V-JEPA 2's Block.attn slot
# ===========================================================================
class _RoPEAttention(nn.Module):
    """Self-attention layer with externally-provided 3D RoPE.

    Has the exact forward signature V-JEPA 2's ``Block`` expects:
        forward(x, mask=None, attn_mask=None) -> Tensor
    so we can swap it into a stock V-JEPA 2 ``Block`` after construction.

    Internally identical to V-JEPA 2's ``Attention`` (modules.py:390) except
    that, after computing Q/K/V, we route Q and K through the provided RoPE
    module before scaled-dot-product attention. There is no positional logic
    in this class itself; all of it lives in ``FactorizedRoPE3D``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope: nn.Module,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa
        self.rope = rope

    def forward(self, x: torch.Tensor, mask=None, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                                # (3,B,H,N,Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.rope(q, k)

        if attn_mask is not None or self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.proj_drop_prob if self.training else 0.0,
                    attn_mask=attn_mask,
                )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ===========================================================================
# DINOv3 attentive video probe
# ===========================================================================
class DINOv3AttentiveProbe(nn.Module):
    """DINOv3 video-classification attentive probe.

    Default config follows the paper recipe: 16 frames x 16 x 16 patches =
    4096 tokens at probe_dim=1024 with 16 heads.

    Parameters
    ----------
    in_dim : int
        Per-patch feature dim from the backbone (1024 for ViT-L/16, etc.).
    probe_dim : int
        Probe internal dimension. Default 1024 (paper).
    num_classes : int
    num_heads : int
        Default 16 (paper does not specify; matches V-JEPA 2 SSv2 config).
    num_blocks : int
        Number of self-attention blocks. The cross-attention pooler is a
        separate layer that always sits on top. Default 4 -> 4 self-attn +
        1 cross-attn = 5 layers, matching DINOv3 App. D.6: "four
        self-attention blocks ... After the four blocks, we apply a
        cross-attention block".
    t_size, h_size, w_size : int
        Patch grid. Used to build the sin-cos PE and RoPE caches.
    mlp_ratio : float
        FFN hidden multiplier (4.0 standard).
    qkv_bias : bool
        V-JEPA 2 default is True; we keep it.
    use_rope : bool
        Whether to apply 3D factorized RoPE inside each self-attn block.
    rope_axis_dims : Sequence[int], optional
        Explicit ``(d_t, d_h, d_w)`` head_dim split. Default: even split.
    use_random_rope_rotations : bool
        Per-head random spatial rotation (paper default True).
    rope_seed : int
    rope_normalize_spatial : bool
        Use [0,1]-normalized spatial coords (matches DINOv3 backbone RoPE box
        in App. C). Default True; pairs with ``rope_base=100``.
    rope_base : float
        RoPE frequency base. 100 with normalized coords, 10000 with raw ints.
    init_std : float
        Trunc-normal std for Linear / query token init.
    """

    def __init__(
        self,
        in_dim: int,
        probe_dim: int = 1024,
        num_classes: int = 3,
        num_heads: int = 16,
        num_blocks: int = 4,
        t_size: int = 16,
        h_size: int = 16,
        w_size: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        use_rope: bool = True,
        rope_axis_dims: Optional[Sequence[int]] = None,
        use_random_rope_rotations: bool = True,
        rope_seed: int = 0,
        rope_normalize_spatial: bool = True,
        rope_base: float = 100.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        if probe_dim % num_heads != 0:
            raise ValueError(f"probe_dim={probe_dim} not divisible by num_heads={num_heads}")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1.")
        head_dim = probe_dim // num_heads

        self.num_blocks = num_blocks
        self.t_size, self.h_size, self.w_size = t_size, h_size, w_size
        self.init_std = init_std

        # 1) Linear projection of per-patch features to probe_dim.
        self.input_proj = nn.Linear(in_dim, probe_dim)

        # 2) Additive 3D sin-cos positional embedding (non-trainable buffer).
        pe = build_3d_sincos_pos_embed(probe_dim, t_size, h_size, w_size)
        self.register_buffer("pos_embed", pe, persistent=False)

        # 3) 3D factorized RoPE shared across self-attn blocks (Q/K only).
        if use_rope:
            self.rope: Optional[nn.Module] = FactorizedRoPE3D(
                head_dim=head_dim, num_heads=num_heads,
                t_size=t_size, h_size=h_size, w_size=w_size,
                base=rope_base,
                axis_dims=rope_axis_dims,
                use_random_rotations=use_random_rope_rotations,
                seed=rope_seed,
                normalize_spatial=rope_normalize_spatial,
            )
        else:
            self.rope = None

        # 4) num_blocks V-JEPA 2 self-attn Blocks. We construct each Block
        # with its default Attention, then overwrite .attn with a
        # _RoPEAttention so the surrounding pre-norm/MLP/residual layout is
        # exactly V-JEPA 2's. If use_rope is False we leave the default
        # attention in place (no positional info beyond the additive sin-cos).
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            blk = Block(
                dim=probe_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=nn.LayerNorm,
                use_rope=False,    # we plug in our own RoPE attention below
            )
            if self.rope is not None:
                blk.attn = _RoPEAttention(
                    dim=probe_dim,
                    num_heads=num_heads,
                    rope=self.rope,
                    qkv_bias=qkv_bias,
                )
            self.blocks.append(blk)

        # 5) Cross-attention pooler with a single position-less learnable
        # query, using V-JEPA 2's CrossAttentionBlock unchanged (pre-norm
        # cross-attn + residual + pre-norm MLP + residual). The query token
        # gets no positional info.
        self.query_token = nn.Parameter(torch.zeros(1, 1, probe_dim))
        self.cross_attn_block = CrossAttentionBlock(
            dim=probe_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=nn.LayerNorm,
        )

        # 6) Final classifier head.
        self.classifier = nn.Linear(probe_dim, num_classes)

        self._init_weights_all()
        self._rescale_blocks()

    # ----- Initialization (V-JEPA 2 AttentivePooler convention) -----
    def _init_weights_all(self) -> None:
        trunc_normal_(self.query_token, std=self.init_std)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self) -> None:
        """V-JEPA 2 AttentivePooler weight rescaling.

        Scales attn.proj and mlp.fc2 weights of deeper layers by
        ``1 / sqrt(2 * layer_id)`` to compensate for residual variance growth.
        Self-attn blocks are layer_id 1..num_blocks; cross-attn block is the
        deepest at layer_id == num_blocks + 1. CrossAttention has no .proj
        (commented out in V-JEPA 2's source), so we rescale its FFN only.
        """
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, blk in enumerate(self.blocks, start=1):
            rescale(blk.attn.proj.weight.data, layer_id)
            rescale(blk.mlp.fc2.weight.data, layer_id)

        cross_id = len(self.blocks) + 1
        rescale(self.cross_attn_block.mlp.fc2.weight.data, cross_id)

    # ----- Forward -----
    def forward(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_feats : (B, N, in_dim)
            Dense patch features from the FROZEN DINOv3 backbone with the
            CLS token already discarded. The backbone's final LayerNorm
            (App. A.2) should be applied before this. Tokens must be in
            (t, h, w) order with w fastest. ``N`` must equal
            ``t_size * h_size * w_size``.

        Returns
        -------
        logits : (B, num_classes)
        """
        B, N, _ = patch_feats.shape
        if N != self.pos_embed.shape[0]:
            raise ValueError(
                f"Expected N={self.pos_embed.shape[0]} (T*H*W used at "
                f"construction), got N={N}."
            )

        # (1) project + (2) additive 3D sin-cos PE
        x = self.input_proj(patch_feats)
        x = x + self.pos_embed.to(x.dtype).unsqueeze(0)

        # (3,4) self-attn stack with 3D factorized RoPE inside each block
        for blk in self.blocks:
            x = blk(x)

        # (5) cross-attention pool with a single learnable query
        q = self.query_token.expand(B, -1, -1)
        q = self.cross_attn_block(q, x).squeeze(1)

        # (6) classify
        return self.classifier(q)


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(0)

    def _count(m: nn.Module) -> str:
        return f"{sum(p.numel() for p in m.parameters() if p.requires_grad):,}"

    # Paper config: 16 frames x 16x16 patches = 4096 tokens, head_dim=64
    # -> RoPE auto-split (20, 22, 22). 4 self-attn + 1 cross-attn = 5 layers.
    print("[1] DINOv3AttentiveProbe (paper config)")
    feats = torch.randn(2, 16 * 16 * 16, 1024)
    probe = DINOv3AttentiveProbe(
        in_dim=1024, probe_dim=1024, num_classes=3, num_heads=16,
        num_blocks=4, t_size=16, h_size=16, w_size=16, use_rope=True,
    )
    out = probe(feats)
    print(f"    in {tuple(feats.shape)} -> out {tuple(out.shape)}   params: {_count(probe)}")
    print(f"    self-attn blocks: {len(probe.blocks)}   "
          f"+ 1 cross-attn block = {len(probe.blocks) + 1} total layers")
    print(f"    RoPE cache shape: {tuple(probe.rope.cos_cache.shape)}  "
          f"(num_heads, T*H*W, head_dim)")
    print(f"    per-head rotation angles (rad, first 4): "
          f"{probe.rope.rotation_angles[:4].tolist()}")

    # Determinism: angles are buffers, so eval == eval and train == train
    # (no per-step resampling).
    probe.eval()
    with torch.no_grad():
        y1 = probe(feats); y2 = probe(feats)
    assert torch.allclose(y1, y2), "Eval forward must be deterministic"
    print("    eval determinism: OK")

    # Ablation: random rotations off
    print("[2] DINOv3AttentiveProbe (RoPE on, random rotations OFF)")
    probe_no_rot = DINOv3AttentiveProbe(
        in_dim=1024, probe_dim=1024, num_classes=3, num_heads=16,
        num_blocks=4, t_size=16, h_size=16, w_size=16,
        use_rope=True, use_random_rope_rotations=False,
    )
    print(f"    out {tuple(probe_no_rot(feats).shape)}   params: {_count(probe_no_rot)}")

    # Ablation: no RoPE at all (additive sin-cos only)
    print("[3] DINOv3AttentiveProbe (no RoPE)")
    probe_no_rope = DINOv3AttentiveProbe(
        in_dim=1024, probe_dim=1024, num_classes=3, num_heads=16,
        num_blocks=4, t_size=16, h_size=16, w_size=16, use_rope=False,
    )
    print(f"    out {tuple(probe_no_rope(feats).shape)}   params: {_count(probe_no_rope)}")

    # Gradient check
    out.sum().backward()
    grads_ok = all(p.grad is not None for p in probe.parameters() if p.requires_grad)
    print(f"[grad-check] all parameters got gradients: {grads_ok}")