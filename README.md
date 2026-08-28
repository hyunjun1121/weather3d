# weather3d

Code for the paper **"Weather Robustness of Streaming 3D Reconstruction:
Limits of 2D Weather-Restoration Preprocessing under Physics-Based Fog and
Smoke Synthesis"**.

The pipeline measures how physics-based synthesized fog and smoke degrade a
streaming 3D reconstruction model (StreamVGGT), and whether 2D video weather
restoration (ViWS-Net) applied as a preprocessing step recovers the loss.

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
| `configs/ext_v2.yaml` | fine-tuning data configuration, beyond the paper |
| `configs/ext_v2_tartanair.yaml` | TartanAirV2 generalization data for fine-tuning, beyond the paper |
| `configs/c3_data.yaml` | c3 fine-tuning dataset configuration, beyond the paper |

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
python -B tests/run_all.py        # 20 unit tests (synthesis math, noise
                                  # determinism/time consistency, Umeyama/ATE/RPE
                                  # known-answer, reconstruction metric regression)
python -B tests/model_smoke.py    # weight load + dummy inference (needs weights)
```

## License

This repository's own code is released under the MIT License (see `LICENSE`).

Third-party components keep their original licenses: StreamVGGT code and
weights (CC BY-NC-SA 4.0; research use only), ViWS-Net (its repository
license), and the 7-Scenes / Neural-RGBD datasets (their respective terms).

## Citation

This release (v1.0.1) is archived on Zenodo:
<https://doi.org/10.5281/zenodo.22137888> (all versions:
<https://doi.org/10.5281/zenodo.22136904>). A BibTeX entry will be added
upon paper publication.
