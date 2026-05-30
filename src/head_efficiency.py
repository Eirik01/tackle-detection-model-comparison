"""Minimal head-only efficiency profiler shared by eval_spatial / eval_temporal.

Measures three quantities for the classification head (backbone features are
already cached on disk, so the head consumes pre-extracted tensors):

  1. Trainable parameters (M)          -- sum of `numel()` for params with
                                          `requires_grad=True`, divided by 1e6.
  2. Mean head latency (ms per batch)  -- wall-clock per head forward pass on
                                          GPU at batch size 16, fp32, averaged
                                          over `n_timed=200` batches after
                                          `n_warmup=20` warmup batches.
  3. Peak head VRAM (MiB)              -- `torch.cuda.max_memory_allocated()`
                                          measured around the timed batches,
                                          reset just before.

Latency percentiles, FLOPs, MACs, throughput and RTF are intentionally not
computed. Results are appended to a single CSV at
``tackle-detection-model-comparison/results/head_efficiency.csv``.

The profiler is GPU-only: calling it without CUDA raises RuntimeError with a
clear message. It does not interfere with training or the existing eval loop.
"""

from __future__ import annotations

import csv
import datetime
import time
from pathlib import Path
from typing import Iterable, List

import torch
import torch.nn as nn


CSV_COLUMNS = [
    "pipeline",
    "trainable_params_M",
    "mean_batch_latency_ms",
    "peak_vram_mib",
    "batch_size",
    "n_warmup",
    "n_timed",
    "device",
    "torch_version",
    "timestamp",
]

PROFILE_BATCH_SIZE = 16
N_WARMUP = 20
N_TIMED = 200
# Distinct profile batches kept resident on GPU; the timing loop cycles
# through them. Bounded so attentive-head dense-token features (~5-10 MiB
# per window) do not blow up host/GPU memory.
MAX_DISTINCT_BATCHES = 8


def count_trainable_params_m(module: nn.Module) -> float:
    """Trainable-parameter count of `module` (params with ``requires_grad=True``)
    in millions."""
    n = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return n / 1e6


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--profile-efficiency requires CUDA. No CUDA device is available; "
            "run this on a GPU node (or remove the flag)."
        )
    return torch.device("cuda")


def _collect_input_batches(
    batch_iter: Iterable[torch.Tensor],
    n_distinct: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Pull pre-extracted feature tensors from an iterable until `n_distinct`
    distinct GPU batches at ``PROFILE_BATCH_SIZE`` are available.

    The timing loop in :func:`measure_head_efficiency` cycles through the
    returned list, so we only ever hold a small number of batches resident on
    GPU (8 by default). This keeps memory bounded for attentive-head dense-
    token inputs (~5-10 MiB per window) while still giving the head varied
    inputs across timed iterations.

    `batch_iter` yields CPU tensors shaped ``[B_src, ...]`` (any source batch
    size). Buffered samples are concatenated along dim 0 and re-chunked to
    ``PROFILE_BATCH_SIZE``. Trailing partial batches are dropped.
    """
    target_samples = n_distinct * PROFILE_BATCH_SIZE
    buf: List[torch.Tensor] = []
    collected_samples = 0
    iterator = iter(batch_iter)
    while collected_samples < target_samples:
        try:
            t = next(iterator)
        except StopIteration:
            break
        # Defensive copy to detach from any underlying DataLoader pinned buffers.
        buf.append(t.detach().to(dtype=torch.float32, device="cpu", copy=True))
        collected_samples += buf[-1].shape[0]

    if not buf:
        raise RuntimeError(
            "No feature batches available to profile the head. "
            "Check that the eval DataLoader / feature cache yields data."
        )

    pool = torch.cat(buf, dim=0)
    total = (pool.shape[0] // PROFILE_BATCH_SIZE) * PROFILE_BATCH_SIZE
    if total == 0:
        raise RuntimeError(
            f"Need at least {PROFILE_BATCH_SIZE} samples to profile, got "
            f"{pool.shape[0]}."
        )

    # Cap at the requested distinct-batch count; do NOT tile to fill n_warmup +
    # n_timed -- the timing loop cycles through whatever distinct batches we
    # have. Source pools smaller than n_distinct just get fewer distinct
    # batches; head latency depends on input shape, not content.
    pool = pool[:total]
    n_available = pool.shape[0] // PROFILE_BATCH_SIZE
    n_keep = min(n_distinct, n_available)
    pool = pool[: n_keep * PROFILE_BATCH_SIZE]
    batches = list(
        pool.view(n_keep, PROFILE_BATCH_SIZE, *pool.shape[1:]).unbind(dim=0)
    )
    # Move to GPU once; we want to time the forward only.
    return [b.to(device=device, non_blocking=False).contiguous() for b in batches]


@torch.no_grad()
def measure_head_efficiency(
    *,
    head: nn.Module,
    input_batches: List[torch.Tensor],
    device: torch.device,
    n_warmup: int = N_WARMUP,
    n_timed: int = N_TIMED,
) -> dict:
    """Run warmup + timed forward passes through `head`. Returns a dict with
    `mean_batch_latency_ms`, `peak_vram_mib`."""
    assert device.type == "cuda", "measure_head_efficiency requires CUDA"
    assert len(input_batches) >= 1, "need at least one input batch"

    head.eval()
    n_distinct = len(input_batches)

    # Warmup. Cycle through the distinct batches so the very first ones still
    # see the head with caches/jit/cuBLAS heuristics warmed.
    for i in range(n_warmup):
        _ = head(input_batches[i % n_distinct])
    torch.cuda.synchronize()

    # Reset peak VRAM right before the timed section.
    torch.cuda.reset_peak_memory_stats()

    # Timed batches.
    latencies_ms: List[float] = []
    for i in range(n_timed):
        batch = input_batches[i % n_distinct]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = head(batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return {
        "mean_batch_latency_ms": float(mean_ms),
        "peak_vram_mib": float(peak_mib),
    }


def append_csv_row(
    csv_path: Path,
    *,
    pipeline: str,
    trainable_params_m: float,
    mean_batch_latency_ms: float,
    peak_vram_mib: float,
    n_warmup: int = N_WARMUP,
    n_timed: int = N_TIMED,
    batch_size: int = PROFILE_BATCH_SIZE,
    device_name: str = "cuda",
) -> None:
    """Append one row to the shared head-efficiency CSV. Creates the file
    with the header if it does not exist."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_COLUMNS)
        w.writerow([
            pipeline,
            f"{trainable_params_m:.6f}",
            f"{mean_batch_latency_ms:.4f}",
            f"{peak_vram_mib:.2f}",
            batch_size,
            n_warmup,
            n_timed,
            device_name,
            torch.__version__,
            datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        ])


def profile_head(
    *,
    head: nn.Module,
    feature_batch_iter: Iterable[torch.Tensor],
    pipeline: str,
    csv_path: Path,
    n_warmup: int = N_WARMUP,
    n_timed: int = N_TIMED,
) -> dict:
    """End-to-end profiler entry point.

    Parameters
    ----------
    head : nn.Module
        The classification head (already loaded with checkpoint state, .eval()).
    feature_batch_iter : Iterable[torch.Tensor]
        Iterable yielding pre-extracted feature tensors on CPU. Each tensor is
        a (sub)batch shaped ``[B_src, ...]`` where the trailing dims match the
        head's expected input. ``B_src`` may be anything; batches are re-chunked
        to ``PROFILE_BATCH_SIZE``.
    pipeline : str
        One of ``dinov3_linear``, ``dinov3_attentive``, ``vjepa2_attentive``.
        Written to the CSV.
    csv_path : Path
        Output CSV path (appended; header written if missing).
    """
    device = _require_cuda()
    head.to(device)
    head.eval()

    # Trainable params first; this is independent of the timing.
    params_m = count_trainable_params_m(head)

    input_batches = _collect_input_batches(
        feature_batch_iter, MAX_DISTINCT_BATCHES, device,
    )

    timing = measure_head_efficiency(
        head=head,
        input_batches=input_batches,
        device=device,
        n_warmup=n_warmup,
        n_timed=n_timed,
    )

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    append_csv_row(
        csv_path,
        pipeline=pipeline,
        trainable_params_m=params_m,
        mean_batch_latency_ms=timing["mean_batch_latency_ms"],
        peak_vram_mib=timing["peak_vram_mib"],
        n_warmup=n_warmup,
        n_timed=n_timed,
        batch_size=PROFILE_BATCH_SIZE,
        device_name=device_name,
    )

    result = {
        "pipeline": pipeline,
        "trainable_params_M": params_m,
        "mean_batch_latency_ms": timing["mean_batch_latency_ms"],
        "peak_vram_mib": timing["peak_vram_mib"],
        "batch_size": PROFILE_BATCH_SIZE,
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "device": device_name,
        "csv_path": str(csv_path),
    }
    return result
