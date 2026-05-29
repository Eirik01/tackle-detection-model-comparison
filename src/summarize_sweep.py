"""Extended sweep summary that handles both pipelines.

`select_hparams.py` ranks by val macro-F1. This script adds:
  - last_epoch  : last epoch the run reached
  - early_stop  : whether the patience trigger fired
  - train_loss  : train loss at the saved (best macro-F1) epoch
  - gap         : val_loss - train_loss at the saved epoch  (overfit depth)
  - f1_win      : mean val_f1 over [best_ep-2, best_ep+2]   (peak stability)
  - ep_diff     : best_f1_epoch - best_val_loss_epoch       (how far past the
                  loss minimum the F1-best checkpoint sits)

Auto-detects which run-file layout each combo uses:
  * spatial pipeline writes per-combo `config.json` + `metrics.json`
    (metrics.json embeds the full per-epoch history -- no slurm log needed)
  * temporal pipeline writes `train.json` only; per-epoch values are scraped
    from the matching slurm .out file (matched on the "LR / WD: ..." line)

Run from tackle-detection-model-comparison/ as:
    # spatial (no log needed)
    python -m src.summarize_sweep \\
        --sweep-dir /cluster/.../sweeps/dinov3_linear_spatial/seed42

    # temporal (logs needed for the per-epoch fields)
    python -m src.summarize_sweep \\
        --sweep-dir sweeps/dinov3_l/seed42 \\
        --log-dir   slurm_logs/sweep/temporal \\
        --backbone  dinov3
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

EPOCH_RE = re.compile(
    r"Epoch +(\d+)/\d+\s+train_loss=([\d.eE+-]+)\s+val_loss=([\d.eE+-]+)"
    r"\s+val_acc=[\d.]+\s+val_f1=([\d.]+)"
)
LR_WD_RE = re.compile(r"LR / WD:\s+(\S+)\s*/\s*(\S+)")
EARLY_STOP_RE = re.compile(r"[Ee]arly[- ]stop")


def parse_log(path: Path) -> dict:
    epochs: list[tuple[int, float, float, float]] = []
    early_stop = False
    try:
        text = path.read_text(errors="ignore")
    except (FileNotFoundError, OSError):
        return {"last_epoch": None, "early_stop": None, "epochs": []}
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            epochs.append((int(m[1]), float(m[2]), float(m[3]), float(m[4])))
            continue
        if EARLY_STOP_RE.search(line):
            early_stop = True
    return {
        "last_epoch": epochs[-1][0] if epochs else None,
        "early_stop": early_stop,
        "epochs": epochs,
    }


def _close(a: str, b: str, rel: float = 1e-6) -> bool:
    """Numeric comparison so '0.001' matches '1e-3', '0.1' matches '1e-1', etc."""
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if fa == fb:
        return True
    denom = max(abs(fa), abs(fb), 1e-30)
    return abs(fa - fb) / denom < rel


def find_log(log_dir: Optional[Path], backbone: Optional[str],
             lr: str, wd: str) -> Optional[Path]:
    if log_dir is None:
        return None
    pattern = f"{backbone}_*.out" if backbone else "*.out"
    candidates: list[Path] = []
    for p in sorted(log_dir.glob(pattern)):
        try:
            text = p.read_text(errors="ignore")
        except (FileNotFoundError, OSError):
            continue
        m = LR_WD_RE.search(text)
        if not m:
            continue
        if _close(m.group(1), lr) and _close(m.group(2), wd):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def gather_run(combo_dir: Path, log_dir: Optional[Path],
               backbone: Optional[str]) -> Optional[dict]:
    """Return a unified run dict (spatial or temporal), or None if neither
    schema matches."""
    metrics_path = combo_dir / "metrics.json"
    config_path = combo_dir / "config.json"
    train_path = combo_dir / "train.json"

    # Spatial: metrics.json carries the full history, so no log scrape needed.
    if metrics_path.exists() and config_path.exists():
        metrics = json.loads(metrics_path.read_text())
        config = json.loads(config_path.read_text())
        history = metrics.get("history", [])
        epochs = [
            (int(h["epoch"]), float(h["train_loss"]),
             float(h["val_loss"]), float(h["val_macro_f1"]))
            for h in history
            if all(k in h for k in ("epoch", "train_loss", "val_loss", "val_macro_f1"))
        ]
        last_epoch = epochs[-1][0] if epochs else None
        total_epochs = config.get("epochs")
        early_stop: Optional[bool] = None
        if last_epoch is not None and isinstance(total_epochs, int):
            early_stop = last_epoch < total_epochs
        return {
            "lr": str(config.get("lr")),
            "wd": str(config.get("weight_decay")),
            "best_val_macro_f1": metrics.get("best_val_macro_f1"),
            "best_val_macro_f1_epoch": (metrics.get("best_val_macro_f1_epoch")
                                        or metrics.get("best_epoch")),
            "val_loss_at_best_f1": metrics.get("val_loss_at_best_f1",
                                               metrics.get("best_val_loss")),
            "best_val_loss": metrics.get("best_val_loss"),
            "best_val_loss_epoch": metrics.get("best_val_loss_epoch"),
            "epochs": epochs,
            "last_epoch": last_epoch,
            "early_stop": early_stop,
            "log": None,
        }

    # Temporal: train.json + slurm log scrape for per-epoch values.
    if train_path.exists():
        tj = json.loads(train_path.read_text())
        targs = tj.get("args", {})
        lr = str(targs.get("learning_rate"))
        wd = str(targs.get("weight_decay"))
        log = find_log(log_dir, backbone, lr, wd)
        info = parse_log(log) if log else {"last_epoch": None,
                                           "early_stop": None,
                                           "epochs": []}
        return {
            "lr": lr,
            "wd": wd,
            "best_val_macro_f1": tj.get("best_val_macro_f1"),
            "best_val_macro_f1_epoch": tj.get("best_val_macro_f1_epoch"),
            "val_loss_at_best_f1": tj.get("val_loss_at_best_f1",
                                          tj.get("best_val_loss")),
            "best_val_loss": tj.get("best_val_loss"),
            "best_val_loss_epoch": tj.get("best_val_loss_epoch"),
            "epochs": info["epochs"],
            "last_epoch": info["last_epoch"],
            "early_stop": info["early_stop"],
            "log": str(log) if log else None,
        }

    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="Directory containing slurm .out files. Required only "
                         "for the temporal pipeline (where train.json lacks "
                         "per-epoch history); spatial sweeps already embed it "
                         "in metrics.json.")
    ap.add_argument("--backbone", choices=["dinov3", "vjepa2"], default=None,
                    help="Optional log-filename prefix used to narrow which "
                         "*.out files are scanned. Only relevant when "
                         "--log-dir is given.")
    args = ap.parse_args()

    rows: list[dict] = []
    pipelines = set()
    for combo_dir in sorted(args.sweep_dir.iterdir()):
        if not combo_dir.is_dir():
            continue
        run = gather_run(combo_dir, args.log_dir, args.backbone)
        if run is None:
            continue
        val_f1 = run["best_val_macro_f1"]
        if val_f1 is None:
            continue
        pipelines.add("spatial" if (combo_dir / "metrics.json").exists() else "temporal")

        best_f1_epoch = run["best_val_macro_f1_epoch"]
        val_loss_at_best_f1 = run["val_loss_at_best_f1"]
        best_val_loss_epoch = run["best_val_loss_epoch"]

        train_loss_at_best_f1: Optional[float] = None
        f1_window_mean: Optional[float] = None
        if best_f1_epoch is not None and run["epochs"]:
            by_ep = {ep: (tl, vl, vf) for ep, tl, vl, vf in run["epochs"]}
            if best_f1_epoch in by_ep:
                train_loss_at_best_f1 = by_ep[best_f1_epoch][0]
            window = [by_ep[e][2]
                      for e in range(best_f1_epoch - 2, best_f1_epoch + 3)
                      if e in by_ep]
            if window:
                f1_window_mean = sum(window) / len(window)

        gap: Optional[float] = None
        if val_loss_at_best_f1 is not None and train_loss_at_best_f1 is not None:
            gap = val_loss_at_best_f1 - train_loss_at_best_f1

        ep_diff: Optional[int] = None
        if best_f1_epoch is not None and best_val_loss_epoch is not None:
            ep_diff = best_f1_epoch - best_val_loss_epoch

        rows.append({
            "lr": run["lr"],
            "wd": run["wd"],
            "val_macro_f1": val_f1,
            "val_loss_at_best_f1": val_loss_at_best_f1,
            "best_val_loss": run["best_val_loss"],
            "best_val_loss_epoch": best_val_loss_epoch,
            "best_epoch": best_f1_epoch,
            "last_epoch": run["last_epoch"],
            "early_stop": run["early_stop"],
            "train_loss_at_best_f1": train_loss_at_best_f1,
            "gap_at_best_f1": gap,
            "f1_window_mean": f1_window_mean,
            "ep_diff_f1_loss": ep_diff,
            "log": run["log"],
        })

    rows.sort(key=lambda r: r["val_macro_f1"], reverse=True)

    def fmt(x: Optional[float], width: int, prec: int = 4) -> str:
        return f"{x:>{width}.{prec}f}" if x is not None else f"{'n/a':>{width}}"

    hdr = (f"{'lr':>8}  {'wd':>6}  {'val_f1':>8}  {'val_loss':>9}  "
           f"{'train_l':>9}  {'gap':>7}  {'f1_win':>8}  "
           f"{'best_ep':>7}  {'last_ep':>7}  {'es':>3}  {'epΔ':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        be = "?" if r["best_epoch"] is None else str(r["best_epoch"])
        le = "?" if r["last_epoch"] is None else str(r["last_epoch"])
        es = ("yes" if r["early_stop"] is True
              else "no" if r["early_stop"] is False
              else "?")
        ed = "?" if r["ep_diff_f1_loss"] is None else f"{r['ep_diff_f1_loss']:+d}"
        print(
            f"{r['lr']:>8}  {r['wd']:>6}  "
            f"{r['val_macro_f1']:>8.4f}  "
            f"{fmt(r['val_loss_at_best_f1'], 9)}  "
            f"{fmt(r['train_loss_at_best_f1'], 9)}  "
            f"{fmt(r['gap_at_best_f1'], 7, prec=3)}  "
            f"{fmt(r['f1_window_mean'], 8)}  "
            f"{be:>7}  {le:>7}  {es:>3}  {ed:>5}"
        )

    out = args.sweep_dir / "summary.json"
    pipeline_tag = (pipelines.pop() if len(pipelines) == 1
                    else "mixed" if pipelines else None)
    out.write_text(json.dumps(
        {"pipeline": pipeline_tag, "backbone": args.backbone, "rows": rows},
        indent=2,
    ))
    print(f"\n[summary] written: {out}")


if __name__ == "__main__":
    main()
