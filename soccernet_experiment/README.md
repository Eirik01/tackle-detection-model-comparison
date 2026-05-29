# SoccerNet single-half tackle-prediction experiment

Run the **best pipeline (DINOv3 attentive probe)** on one half of one SoccerNet-v2
game and dump every fired tackle event with a timestamp, so the predictions can
be checked by hand against the broadcast.

Target clip: `Manchester City 3 - 1 Barcelona`, UCL 2016-11-01, **half 2**, 720p. Game + half + symlink name are constants at the top of [`download_half.py`](download_half.py); the experiment root path is `SOCCERNET_EXPERIMENT_DIR` in [`src/config.py`](../src/config.py) (mirrored as the `EXP_DIR` default in the two .sh files).

This reuses the exact training/eval preprocessing (reflect padding → 256×256,
25 FPS dense tokens, strided to the W=10 @ 5 FPS window at load time) and the
exact peak-detection postprocessing that feeds the Average-mAP metric. The only
thing it does **not** do is compute mAP — SoccerNet has no tackle ground truth,
so you verify the fired events manually.

## One-time prerequisites

Add the SoccerNet password to `thesis_code/.env` (same file as `DINOv3_key`):

```
SoccerNetv2_password=<the SoccerNet-v2 password>
```

Confirm your best checkpoint exists at
`/cluster/work/projects/ec12/ec-eirikto/TACDEC/models/dinov3_l/best_attn_dinov3_l_<SUFFIX>.pth`
and note its `<SUFFIX>` (passed as `MODEL_SUFFIX`, default `centered_v1`).

## Run order (from `thesis_code/`)

```bash
# 1. Download + symlink the half  (LOGIN NODE — compute nodes have no internet)
uv run python soccernet_experiment/download_half.py

# 2. Extract DINOv3-L dense features  (GPU)
sbatch soccernet_experiment/run_extract.sh

# 3. Predict fired events  (GPU) — set MODEL_SUFFIX to your best run
MODEL_SUFFIX=centered_v1 sbatch soccernet_experiment/run_predict.sh
```

Everything lands under `EXP_DIR` (default
`/cluster/work/projects/ec12/ec-eirikto/soccernet_thesis_experiment`, on the
temporary work area):

```
soccernet_thesis_experiment/
├── europe_uefa-champions-league/2016-2017/.../2_720p.mkv   # SoccerNet's nested layout
├── features/2_720p_dinov3_l_25.0fps_reflect_dense_features.npy
└── predictions/
    ├── 2_720p_events.txt        # human-readable fired events  <-- read this
    ├── 2_720p_events.csv        # fired events (conf >= threshold)
    ├── 2_720p_events_all.csv    # every peak (re-threshold offline)
    └── 2_720p_predict_summary.json
```

## The output

`*_events.txt` lists, sorted by time:

```
timestamp (mm:ss) |   sec   | event          | confidence
12:34.6           |  754.60 | tackle-live    | 0.871
```

Timestamps are seconds **from the start of the half** (= the video clock if you
open the .mkv at 0:00). `tackle-live` = live-action tackle, `tackle-replay` =
replay of a tackle.

## Knobs

- `MIN_CONFIDENCE` (default `0.5`): peak height to count as "fired" in the .txt.
  Lower it to catch more, raise it for precision. `*_events_all.csv` holds every
  peak regardless, so you can re-threshold without re-running inference.
- `SIGMA` (`1.0`), `MIN_DISTANCE_SEC` (`0.5`): smoothing / min spacing between
  peaks — same defaults as `eval_temporal.py`.
- To change the clip: edit `GAME`, `VIDEO_FILE`, `SYMLINK_NAME` at the top of `download_half.py`. To put outputs somewhere else: edit `SOCCERNET_EXPERIMENT_DIR` in `src/config.py` and the mirrored `EXP_DIR` defaults in the two .sh files.

## Notes / gotchas

- **Filenames are generic (`2_720p_*`).** SoccerNet's downloader nests the
  file under `<game path>/2_720p.mkv` and we keep the layout, so the file
  stem used as the feature/prediction prefix is the generic `2_720p`. The
  extractor globs both `*.mp4` and `*.mkv` recursively, so the nesting is
  transparent to the rest of the pipeline.
- **FPS assumption.** SoccerNet-v2 broadcasts are 25 FPS, matching TACDEC, so
  `source_fps=25` / `target_fps=5` / `W=10` carry over unchanged. The extractor
  reads the real FPS from the file; if a clip is not 25 FPS the strides would
  drift — check the "Original: … FPS" line in the extract log.
- **Size/time.** A full 45-min half at 25 FPS dense is ~30–40 GB on disk and a
  few thousand probe forwards; extraction ~15–30 min, prediction a few minutes.
- This experiment writes only under `soccernet_experiment/` and `EXP_DIR`; it
  does not touch TACDEC features, models, or results.
```
