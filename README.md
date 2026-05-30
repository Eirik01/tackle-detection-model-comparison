# Foundation Models for Tackle Detection in Football

Experiment code for the master's thesis *"A Comparative Study between Spatial
and Spatio-Temporal Self-Supervised Foundation Models for Tackle Detection in
Football"* (University of Oslo, Spring 2026).

The thesis asks a single question: **as frozen backbones for tackle event
detection in broadcast football, how do spatial and spatio-temporal
self-supervised foundation models compare?** This repository contains the three
paper-faithful pipelines used to answer it, evaluated on the
[TACDEC](https://doi.org/10.1145/3625468.3652166) dataset (Norwegian
Eliteserien, yellow-card-filtered tackle clips).

## The three pipelines

Every pipeline follows the same **two-stage design**: a frozen ViT-Large
backbone extracts features once, then a lightweight head is trained on the
cached features. Backbones are never fine-tuned — only the head is trained.

| # | Backbone (frozen) | Paradigm | Head | Temporal reasoning lives in… | Evaluation |
|---|-------------------|----------|------|------------------------------|------------|
| 1 | DINOv3 ViT-L | spatial (per-frame) | linear probe (~5k params) | nowhere | single 70/15/15 split (seed 42) + 5-fold game-disjoint CV |
| 2 | DINOv3 ViT-L | spatial (per-frame) | attentive probe | the head | single 70/15/15 split (seed 42) |
| 3 | V-JEPA 2 ViT-L | spatio-temporal | attentive probe | the backbone | single 70/15/15 split (seed 42) |

Pipeline 1 is evaluated **two ways**: (1) on the *same* single seed-42 game-disjoint
split as pipelines 2 & 3 — identical held-out test games, so all three pipelines are
compared like-for-like — and (2) with 5-fold game-disjoint cross-validation, for a more
robust estimate of the linear probe that doesn't hinge on one split.

Both backbones are ViT-Large (~300 M params, 1024-d embeddings) so the
comparison isolates *representation type* (image vs. video pre-training) from
model capacity. Heads see a 10-frame window at 5 FPS (2.0 s of footage). DINOv3
features are cached densely at 25 FPS and strided to 5 FPS at load time;
V-JEPA 2 features are cached directly at 5 FPS because its tokens depend on the
input frame rate.

**Headline result:** on TACDEC the spatial DINOv3 backbone beats
spatio-temporal V-JEPA 2 at both frame and event level (~13 mAP gap at event
level), and the linear probe matches the attentive probe within seed noise — so
post-hoc temporal aggregation does not help on this task.

## Repository layout

```
tackle-detection-model-comparison/
├── src/                          # the pipeline package
│   ├── config.py                 # ── single source of truth for all paths & backbone config
│   ├── feature_extractors/       # Stage 1: frozen backbones → cached features
│   │   ├── base_extractor.py
│   │   ├── dinov3_extractor.py
│   │   └── vjepa2_extractor.py
│   ├── models/                   # Stage 2: the trainable heads
│   │   ├── dinov3/               #   linear_probe.py, attentive_probe.py
│   │   └── vjepa2/               #   attentive_pooler.py
│   ├── data/                     # datasets + protocols (game-disjoint splits,
│   │                             #   spatial/temporal windowing, class consolidation)
│   ├── train_spatial.py          # train pipeline 1 (DINOv3 linear)
│   ├── train_temporal.py         # train pipelines 2 & 3 (attentive probes)
│   ├── eval_spatial.py           # frame-level eval, spatial probe
│   ├── eval_spatial_centred.py
│   ├── eval_temporal.py          # window/event-level eval, attentive probes
│   ├── postprocess.py            # peak detection → fired events (feeds Average-mAP)
│   ├── window_protocol.py        # shared windowing definition
│   ├── soccernet_eval.py         # SoccerNet Average-mAP metric (used by eval_*)
│   ├── select_hparams.py         # pick winners from a sweep
│   ├── summarize_sweep.py
│   ├── aggregate_kfold_spatial.py# combine the 5 spatial folds
│   ├── head_efficiency.py        # head latency / throughput / peak memory
│   └── utils.py
│
├── extract_features.py           # top-level feature-extraction entry point
├── dump_kassab_split.py          # reproduce Kassab's exact 70/15/15 partition
├── dump_dinov3_kassab_format.py
├── verify_kassab_test_split.py   # parity check vs. Kassab's published frame counts
│
├── run_*.sh                      # SLURM job scripts (UiO FOX HPC) — see below
├── sweep_*.sh                    # LR/WD hyperparameter sweeps
├── setup.sh                      # module load + uv sync, sourced by every run script
│
├── analysis/                     # dataset & result analysis (class distribution,
│                                 #   event durations, attention windows, clustering)
├── visualization/                # figure generation for the thesis
├── untrimmed_footage_experiment/ # cross-dataset check: best probe on a full
│                                 #   SoccerNet half (predict_soccernet*.py +
│                                 #   runners + manual verification; own README)
├── tests/                        # parity / sanity scripts (split parity, padding checks)
├── results/                      # generated figures
├── pyproject.toml                # dependencies (Python 3.12, torch 2.8, transformers 4.57)
└── slurm_logs/                   # job output
```

`src/config.py` is the place to look first: it defines the TACDEC video/label/
feature/results paths and the backbone selection (`BACKBONE_TYPE`,
`BACKBONE_SIZE`, overridable via environment variables). The paths point at the
UiO **FOX HPC cluster** (`/cluster/work/...` and `/fp/projects01/...`); change
them there to run elsewhere.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate
uv sync
```

On the FOX cluster, `setup.sh` handles the module loads and `uv sync` and is
sourced automatically by the `run_*.sh` scripts. Create a `.env` file at the
repo root for secrets (`DINOv3_key` for the gated HuggingFace weights, and
`SoccerNetv2_password` if running the SoccerNet experiment).

## How to run

The pipeline is two stages — **extract features once, then train/eval heads on
the cache**. On the cluster this is done with `sbatch`; the scripts wrap the
underlying Python entry points and pin paths/seeds.

```bash
# ── Stage 1: extract features (GPU) ────────────────────────────────
sbatch run_extract_dinov3_large_array.sh        # DINOv3 dense @ 25 FPS  (pipelines 1 & 2)
sbatch run_extract_vjepa2_dense_w10_5fps.sh      # V-JEPA 2 dense @ 5 FPS (pipeline 3)

# ── Stage 2: train + evaluate each pipeline (GPU) ──────────────────
sbatch run_train_eval_spatial.sh                 # pipeline 1: DINOv3 linear, single seed-42 split (same test games as 2 & 3)
sbatch run_train_eval_spatial_kfold.sh           # pipeline 1: DINOv3 linear, 5-fold CV (robustness)
sbatch run_train_eval_temporal_dinov3.sh         # pipeline 2: DINOv3 attentive probe
sbatch run_train_eval_temporal_vjepa2.sh         # pipeline 3: V-JEPA 2 attentive probe

# ── Hyperparameter sweeps (LR/WD grids) ────────────────────────────
sbatch sweep_spatial.sh
sbatch sweep_temporal_dinov3.sh
sbatch sweep_temporal_vjepa2.sh

# ── Head efficiency profiling (params / latency / peak VRAM) ───────
# Always runs as part of the eval step above — every train/eval (and the
# standalone eval) appends a row to results/head_efficiency.csv. The scripts
# are pinned to rtx30 so the numbers are comparable across pipelines.
```

Run scripts take positional overrides — e.g.
`sbatch run_train_eval_spatial.sh 42 50 2e-4 tight 0` is `seed epochs lr metric
weight-decay`. The defaults baked into each script are the values selected from
the sweeps and used in the thesis.

Off-cluster, call the Python entry points directly (they read defaults from
`src/config.py`):

```bash
uv run python extract_features.py --model dinov3 --size large
uv run python -m src.train_spatial --epochs 50 --lr 2e-4
uv run python -m src.train_temporal      # attentive probes; --help for backbone/window flags
```

### Reproducing the data split

The split is game-disjoint (no match shared across train/val/test) and fixed by
seed 42 to match the original TACDEC baseline. `dump_kassab_split.py` regenerates
the exact partition and `verify_kassab_test_split.py` checks the per-class frame
counts against Kassab's published numbers.

### SoccerNet qualitative check

`untrimmed_footage_experiment/` runs the best pipeline (DINOv3 attentive probe) on one
half of one SoccerNet game and dumps fired tackle events with timestamps for
manual inspection. See [`untrimmed_footage_experiment/README.md`](untrimmed_footage_experiment/README.md).

## Notes

- The code targets the UiO FOX HPC cluster (SLURM, A100 GPUs). The absolute
  paths in `src/config.py` and the cached-feature workflow assume that
  environment.
- Features are large (~135 GB for DINOv3 dense at 25 FPS) and regenerable, so
  they live on the cluster work area, not in the repo.
- The accompanying thesis (`../thesis_writing/`) documents the methodology,
  protocols, and results in full.
