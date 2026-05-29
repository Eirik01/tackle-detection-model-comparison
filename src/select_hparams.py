"""Select the best (lr, weight_decay) combo from an LR/WD sweep.

Scans the per-combo run directories produced by sweep_spatial.sh /
sweep_temporal_dinov3.sh / sweep_temporal_vjepa2.sh and picks the combo with
the highest validation macro-F1. For temporal runs trained with the current
train_temporal.py, `best_val_macro_f1` is the true maximum of val macro-F1 over
the training trajectory (checkpoint and early-stop are also keyed on F1).
Legacy `train.json` files written before that change instead record the F1 at
the best-val-loss epoch; ranking is still by the same field, but the values
are not directly comparable across the two regimes -- check `selection_metric`
in `train.json` to distinguish.

The selected combo is the "base" config frozen for all later experiments
(single-split protocol: train fits the probe, val selects (lr, wd), test is
touched once afterwards).

Run from tackle-detection-model-comparison/ as:
    python -m src.select_hparams --sweep-dir <dir> [--pipeline spatial|temporal]

Reads per combo:
  spatial  -> <combo>/metrics.json  (best_val_macro_f1) + <combo>/config.json
  temporal -> <combo>/train.json    (best_val_macro_f1 + args.learning_rate/...)
Writes:
  <sweep-dir>/selection.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _read_combo(run_dir: Path) -> Optional[Dict]:
    """Return {lr, weight_decay, val_macro_f1, best_val_loss, source} or None."""
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if metrics_path.exists() and config_path.exists():
        metrics = json.loads(metrics_path.read_text())
        cfg = json.loads(config_path.read_text())
        return {
            "lr": cfg.get("lr"),
            "weight_decay": cfg.get("weight_decay"),
            "val_macro_f1": metrics.get("best_val_macro_f1"),
            "best_val_loss": metrics.get("best_val_loss"),
            "best_epoch": metrics.get("best_epoch"),
            "source": str(metrics_path),
        }

    train_path = run_dir / "train.json"
    if train_path.exists():
        train = json.loads(train_path.read_text())
        args = train.get("args", {})
        return {
            "lr": args.get("learning_rate"),
            "weight_decay": args.get("weight_decay"),
            "val_macro_f1": train.get("best_val_macro_f1"),
            "best_val_loss": train.get("best_val_loss"),
            "best_epoch": train.get("best_val_macro_f1_epoch"),
            "selection_metric": train.get("selection_metric"),
            "source": str(train_path),
        }

    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sweep-dir", type=Path, required=True,
                   help="Sweep directory containing per-combo subdirs (e.g. lr1e-3_wd0).")
    p.add_argument("--pipeline", choices=["spatial", "temporal"], default=None,
                   help="Optional label recorded in selection.json. Format is "
                        "auto-detected from the per-combo files regardless.")
    p.add_argument("--metric", default="macro_f1",
                   help="Selection metric. Only 'macro_f1' is supported "
                        "(validation macro-F1 at the best-val-loss epoch).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.metric != "macro_f1":
        raise SystemExit(f"Unsupported --metric {args.metric!r}; only 'macro_f1'.")

    sweep_dir = args.sweep_dir.resolve()
    if not sweep_dir.is_dir():
        raise SystemExit(f"Sweep dir not found: {sweep_dir}")

    combo_dirs = sorted(d for d in sweep_dir.iterdir() if d.is_dir())
    rows: List[Dict] = []
    missing: List[str] = []
    for d in combo_dirs:
        combo = _read_combo(d)
        if combo is None:
            continue
        combo["run_dir"] = str(d)
        if combo["val_macro_f1"] is None:
            missing.append(d.name)
            continue
        rows.append(combo)

    if not rows:
        raise SystemExit(
            f"No combos with a logged val_macro_f1 under {sweep_dir}. "
            f"Did the sweep finish? (dirs without the metric: {missing})"
        )

    rows.sort(key=lambda r: r["val_macro_f1"], reverse=True)
    best = rows[0]

    print(f"[select] sweep dir: {sweep_dir}")
    print(f"[select] combos with a logged val macro-F1: {len(rows)}/{len(combo_dirs)}")
    if missing:
        print(f"[select] WARNING -- {len(missing)} combo dir(s) missing the metric: "
              f"{', '.join(sorted(missing))}")
    print(f"[select] ranking (val macro-F1, best-val-loss epoch):")
    print(f"         {'lr':>10}  {'wd':>8}  {'val_macro_f1':>13}  {'val_loss':>10}")
    for r in rows:
        vl = r["best_val_loss"]
        vl_str = f"{vl:.4f}" if isinstance(vl, (int, float)) else "n/a"
        print(f"         {str(r['lr']):>10}  {str(r['weight_decay']):>8}  "
              f"{r['val_macro_f1']:>13.4f}  {vl_str:>10}")

    print(f"\n[select] BEST: lr={best['lr']}  wd={best['weight_decay']}  "
          f"val_macro_f1={best['val_macro_f1']:.4f}")

    has_legacy = any(r.get("selection_metric") != "val_macro_f1" for r in rows)
    selection = {
        "pipeline": args.pipeline,
        "metric": "val_macro_f1",
        "selection_note": (
            "ranks by best_val_macro_f1 from train.json; for runs written with "
            "selection_metric='val_macro_f1' this is the true max F1 over "
            "training (checkpoint/early-stop also keyed on F1). Older runs "
            "without this tag record F1 at the best-val-loss epoch; do not "
            "mix the two regimes in a single ranking."
            + ("  WARNING: mixed regimes detected in this sweep dir."
               if has_legacy else "")
        ),
        "sweep_dir": str(sweep_dir),
        "n_combos_ranked": len(rows),
        "n_combos_missing_metric": len(missing),
        "best": {
            "lr": best["lr"],
            "weight_decay": best["weight_decay"],
            "val_macro_f1": best["val_macro_f1"],
            "best_val_loss": best["best_val_loss"],
            "best_epoch": best["best_epoch"],
            "run_dir": best["run_dir"],
        },
        "ranking": [
            {
                "lr": r["lr"],
                "weight_decay": r["weight_decay"],
                "val_macro_f1": r["val_macro_f1"],
                "best_val_loss": r["best_val_loss"],
                "run_dir": r["run_dir"],
            }
            for r in rows
        ],
    }
    out_path = sweep_dir / "selection.json"
    out_path.write_text(json.dumps(selection, indent=2))
    print(f"[select] written: {out_path}")


if __name__ == "__main__":
    main()
