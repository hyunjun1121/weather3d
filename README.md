# weather3d

Code for the paper **"Weather Robustness of Streaming 3D Reconstruction:
Limits of 2D Weather-Restoration Preprocessing under Physics-Based Fog and
Smoke Synthesis"**, and for its follow-up capacity study **"Limited
Adaptation Beats Full Fine-Tuning: Pre-Registered Weather Robustness for
Streaming 3D Reconstruction"**.

The pipeline measures how physics-based synthesized fog and smoke degrade a
streaming 3D reconstruction model (StreamVGGT), and whether 2D video weather
restoration (ViWS-Net) applied as a preprocessing step recovers the loss.
The follow-up adds a pre-registered three-arm fine-tuning study: full
fine-tuning with clean replay, degraded-only training, and LoRA (rank 16,
0.56% of parameters), judged by machine-checked gates with paired Wilcoxon
tests, bootstrap confidence intervals, and Holm correction.

## Pipeline

```text
(1) run_prepare.py           dataset existence/integrity check, manifest.json
(2) run_synthesize.py        clean + GT depth --[physics fog/smoke]--> degraded/
(3) run_infer.py             clean/degraded/restored --[StreamVGGT]--> preds/*.npz
    scripts/restore_viws.py  degraded --[ViWS-Net]--> restored/   (case c2)
(4) run_evaluate.py          preds + GT --> depth / pose / reconstruction metrics
(5) run_report.py            degradation & recovery aggregation --> results.csv, report.md
```

## Cases

| Case | Input | Role |
|---|---|---|
| c0 | clean frames | clean reference |
| c1 | weather-degraded frames | degradation measurement |
| c2 | ViWS-Net-restored frames | 2D restoration preprocessing |
| c3 | degraded (fine-tuned model) | work in progress, beyond the paper |

The c3 fine-tuning code and configs (`configs/ext_v2*.yaml`,
`configs/c3_data.yaml`, `src/weather3d/train/`) are included as work in
progress and are not part of the paper.

## Weather synthesis

- Model: `I = J·t + A·(1−t)` with `t = exp(−β·d)`. Fog uses a homogeneous β;
  smoke adds temporally consistent fBm value noise: `β + σ·noise(x, y, t)`.
- Severity presets (indoor, ~10 m depth range): fog β = 0.04 / 0.08 / 0.16 /
  0.32 / 0.64 m⁻¹ (light → extreme); smoke β = 0.03 / 0.06 / 0.12 with
  σ = 0.05 / 0.10 / 0.20. The paper's evaluation uses the five fog levels and
  the mid/heavy smoke levels.
- Only pixel appearance changes, so the clean GT depth and pose remain valid
  ground truth for every condition (standard practice in the
  weather-robustness literature, cf. Foggy Cityscapes / KITTI-fog).
- Synthesis is deterministic per (global seed, sequence, variant).

## Evaluation protocol

StreamVGGT's own evaluation code is reused or closely followed:

| Metric | Implementation | Reference |
|---|---|---|
| video depth (Abs Rel, δ<1.25, ...) | direct reuse of `eval.video_depth.tools.depth_evaluation`, scale&shift (LAD) alignment | StreamVGGT video depth evaluation |
| pose (ATE, RPE) | own implementation (Umeyama Sim(3) alignment → RMSE, alignment applied to all poses before RPE) | MonST3R / CUT3R convention |
| reconstruction (Acc, Comp, NC) | first-camera frame transform → center 224 crop → scale normalization → Open3D ICP (0.1 m) → KDTree, same formulas as `eval.mv_recon.utils` | StreamVGGT mv_recon evaluation |

Known deviations from the original StreamVGGT evaluation: single-GPU
execution with `max_points` (500k default, seeded sampling) instead of
multi-process distributed evaluation, and LAD scale&shift via Adam-based
approximate optimization (a property of the public implementation).

## Setup

Tested with Python 3.10, PyTorch 2.6.0 + CUDA 12.6 on a single NVIDIA Quadro
RTX 8000 (48 GB).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
pip install numpy opencv-python pillow pyyaml scipy einops \
            transformers huggingface_hub tqdm open3d
```

### StreamVGGT (base model)

```bash
git clone --depth 1 https://github.com/wzzheng/StreamVGGT.git third_party/StreamVGGT
mkdir -p third_party/StreamVGGT/ckpt
curl -L -o third_party/StreamVGGT/ckpt/checkpoints.pth \
  "https://huggingface.co/lch01/StreamVGGT/resolve/main/checkpoints.pth"
# expected size: 5,026,743,569 bytes
```

### ViWS-Net (case c2, optional)

Clone <https://github.com/scott-yjyang/ViWS-Net> into `third_party/ViWS-Net`
and build it in a separate environment (the official environment pins
torch 1.9/cu111, which does not install on Python 3.10; torch 2.6.0 cu126 +
timm 0.9.16 + numpy<2 works). With timm 0.9.16 you must add `pretrained_cfg`
absorbing keyword arguments to the `shunted_t/s/b/weather` factory functions
in `third_party/ViWS-Net/modeling/backbone.py`; without this patch,
restoration fails with `TypeError: unexpected keyword 'pretrained_cfg'`.

Restoration is driven by `experiments/scripts/restore_viws.py`, which restores
all degraded frames (edge-replicated borders to keep the frame count) with a
5-frame sliding window into `restored/<variant>/<seq>/`.

### Data

Both benchmarks are indoor sequences with GT depth and GT pose. Expected
layout at the repository root:

```text
data/
├── 7scenes/<scene>/<seq-XX>/
│   ├── frame-XXXXXX.color.png
│   ├── frame-XXXXXX.depth.proj.png   (16-bit mm; raw .depth.png also recognized)
│   └── frame-XXXXXX.pose.txt         (4x4 c2w)
└── neural_rgbd/<scene>/
    ├── images/img<N>.png
    ├── depth/depth<N>.png            (16-bit mm)
    └── poses.txt                     (per-frame 4x4 c2w, sequential)
```

- **7-Scenes**: download the scene zips from the official Microsoft release
  (e.g. `chess.zip`, `fire.zip`, same URL pattern for the other scenes);
  test-split sequences are used for evaluation. The official zips ship raw
  sensor depth (`frame-XXXXXX.depth.png`); the `.depth.proj.png` pseudo-GT
  used by StreamVGGT evaluation can be generated with the
  [SimpleRecon preprocessing](https://github.com/nianticlabs/simplerecon/blob/main/data_scripts/7scenes_preprocessing.py).
  The default configs use the raw depth.
- **Neural-RGBD**: single zip download (≈7.4 GB):

  ```text
  http://kaldir.vc.in.tum.de/neural_rgbd/neural_rgbd_data.zip
  ```

  Then normalize the layout (frame numbering is preserved; do not renumber —
  poses are matched by line index):

  ```bash
  python -B experiments/scripts/convert_nrgbd.py --root data/neural_rgbd
  ```

## Configurations

| Config | Contents |
|---|---|
| `configs/core_v1.yaml` | core four-scene set: 7-Scenes chess, fire, heads + Neural-RGBD whiteroom; five fog + two smoke severities |
| `configs/ext_v1.yaml` | extension scenes: Neural-RGBD staircase, breakfast_room (same pipeline and severities) |
| `configs/ext_v2.yaml` | benchmark grid used in the follow-up study (evaluation scenes + variants) |
| `configs/ext_v2_tartanair.yaml` | TartanAirV2 generalization data (evaluation only, excluded from training) |
| `configs/c3_data.yaml` | fine-tuning dataset configuration (set `seven_scenes_root` / `neural_rgbd_root` to your local paths) |

## Fine-tuning (C3 capacity study)

The follow-up paper fine-tunes StreamVGGT in three arms from the public
checkpoint — R1 full fine-tuning on a 50/50 clean-replay + degraded
mixture, R2 degraded-only training, R3 LoRA r=16 on attention (merged into
the base weights at save time) — under a fixed recipe: 6,600 steps, peak
lr 1e-5 with 300-step warmup, effective batch 2, fp16 + gradient
checkpointing, and a frozen clean-view teacher providing regression
targets on degraded inputs.

```bash
cd experiments
# one arm (mode: r1 | r2; R3 = the r1 recipe with --lora-r 16).
# Single GPU: plain python; multi-GPU: accelerate launch --multi_gpu.
PYTHONPATH=src python -B src/weather3d/train/trainer.py \
  --mode r1 --config configs/c3_data.yaml \
  --svggt-src <path/to/StreamVGGT>/src \
  --ckpt <path/to/StreamVGGT>/ckpt/checkpoints.pth \
  --out outputs/finetune/r1 \
  --grad-checkpoint --ddp-bucket-view --foreach-optimizer
# re-evaluate a fine-tuned checkpoint on the benchmark grid: copy
# configs/ext_v2.yaml, point `model.weights` at outputs/finetune/r1/
# final_model.pth, set `output_dir` and `cases: [c0, c1]`, then:
python -B run_infer.py --config configs/_finetune_eval_r1.yaml
python -B run_evaluate.py --config configs/_finetune_eval_r1.yaml
python -B run_report.py --config configs/_finetune_eval_r1.yaml
# pre-registered gate statistics (G1/G2/G3):
PYTHONPATH=src python -B src/weather3d/stats.py \
  --base-csv <grid results.csv> --arm r1=outputs/finetune_eval/r1/results.csv \
  --base-ta-csv <tartanair results.csv> \
  --arm-ta r1=outputs/finetune_eval/r1_ta/results.csv \
  --out-dir outputs/stats
```

Success criteria were pre-registered before training: G1 superiority over
the degraded baseline on >= 4/7 key metrics, G2 superiority over the
restoration pipeline on >= 4/7, G3 clean-scene regression within 5% on
>= 6/7. `src/weather3d/stats.py` implements the Wilcoxon signed-rank
test (exact enumeration for n<=16), deterministic paired bootstrap 95%
CIs, Holm correction, and the machine verdict.

## Usage

```bash
cd experiments
python -B run_prepare.py            # data presence/integrity check
python -B run_synthesize.py         # fog x5 + smoke x2 degraded sequences
python -B run_infer.py              # c0 (clean) + c1 (degraded) inference
python -B scripts/restore_viws.py   # c2: ViWS-Net restoration (separate env)
python -B run_infer.py --cases c2   # c2 inference
python -B run_evaluate.py           # metrics
python -B run_report.py             # results.csv + report.md
```

For the extension scenes, add `--config configs/ext_v1.yaml`.

Partial runs: `--cases`, `--variants` (e.g. `fog_mid`), `--sequences`
(e.g. `nrgbd_whiteroom`), `--force`. Outputs accumulate under
`outputs/<experiment>/` (`degraded/`, `restored/`, `preds/`, `eval/`,
`report.md`); existing results are skipped unless `--force`.

## Tests

```bash
cd experiments
python -B tests/run_all.py        # 65 unit tests (synthesis math, noise
                                  # determinism, pose/reconstruction metric
                                  # known-answer, trainer arg/LoRA merge,
                                  # Wilcoxon/Holm/bootstrap statistics)
python -B tests/model_smoke.py    # weight load + dummy inference (needs weights)
```

## License

This repository's own code is released under the MIT License (see `LICENSE`).

Third-party components keep their original licenses: StreamVGGT code and
weights (CC BY-NC-SA 4.0; research use only), ViWS-Net (its repository
license), and the 7-Scenes / Neural-RGBD datasets (their respective terms).

## Changelog

- **v2.0.0** (2026-09-07) — fine-tuning capacity study release: `--cudnn-benchmark`
  flag with default OFF (synced; unconditional `benchmark = True` inflated the
  teacher-stage peak by ~10 GiB on 48 GB GPUs), matching trainer test.
  Path hygiene: `configs/c3_data.yaml` now uses repo-relative dataset roots
  (set them to your local layout); the runtime-resolved
  `configs/_ext_v2_resolved.yaml` was removed from the repository (regenerate
  it by running the pipeline's resolve step, or copy `configs/ext_v2.yaml`
  and fill in your paths). Earlier releases contained absolute paths from the
  development server in these two files; the Zenodo archives of v1.0.x are
  immutable and still contain them.
- **v1.0.1** (2026-08-28) — Zenodo creators metadata.
- **v1.0.0** (2026-08-28) — initial public snapshot (evaluation pipeline,
  weather synthesis, restoration case, fine-tuning trainer, statistics).

## Citation

v2.0.0 is archived on Zenodo:
<https://doi.org/10.5281/zenodo.22586228> (all versions:
<https://doi.org/10.5281/zenodo.22136904>). A BibTeX entry will be added
upon paper publication.
