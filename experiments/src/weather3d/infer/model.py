"""StreamVGGT 추론 래퍼.

third_party/StreamVGGT/src를 sys.path에 넣고 공개 모듈을 직접 재사용한다.
추론 절차는 StreamVGGT 자체 평가 코드(src/eval/video_depth/launch.py의
prepare_input -> model.inference)와 동일한 뷰 구성/전처리/autocast를 따른다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch


def _squeeze_np(t) -> np.ndarray:
    """[1,H,W], [1,H,W,1], [1,H,W,3] 등에서 배치/뒤의 크기1 축을 제거한다."""
    arr = t.detach().float().cpu().numpy()
    arr = np.squeeze(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"unexpected prediction shape: {t.shape}")
    return arr


class StreamVGGTRunner:
    def __init__(self, src_dir: str | Path, weights: str | Path, device: str = "cuda"):
        self.src_dir = Path(src_dir).resolve()
        self.weights = Path(weights).resolve()
        self.device = device
        if not self.weights.is_file():
            raise FileNotFoundError(
                f"StreamVGGT checkpoint not found: {self.weights}. "
                "experiments/README.md의 가중치 다운로드 절차 참고."
            )
        if str(self.src_dir) not in sys.path:
            sys.path.insert(0, str(self.src_dir))

        from streamvggt.models.streamvggt import StreamVGGT

        self._StreamVGGT = StreamVGGT
        self._pose_encoding_to_extri_intri = None  # 지연 import

        self.model = StreamVGGT()
        # 신뢰할 수 있는 로컬 checkpoint만 로드한다(AGENTS.md 규칙).
        ckpt = torch.load(self.weights, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt, strict=True)
        del ckpt
        self.model.eval().to(device)

    def _decode_pose(self, pose_enc_list, img_size_hw):
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

        enc = np.stack([p for p in pose_enc_list], axis=0)[None]  # [1, N, 9]
        enc_t = torch.from_numpy(enc.astype(np.float32))
        extri, intri = pose_encoding_to_extri_intri(enc_t, img_size_hw)
        return extri[0].numpy(), intri[0].numpy()  # (N,3,4), (N,3,3)

    def infer(self, image_paths, size: int = 518, crop: bool = False, square_ok: bool = False) -> dict:
        """이미지 경로 리스트 -> 프레임별 depth/point map/pose 예측.

        square_ok: 정사각형 입력(TartanAir V2 640x640)에서만 true로 지정.
            dust3r 계열 load_images_for_eval은 square_ok=False일 때 정사각형
            입력에 4:3 비율 강제(halfh = 3*halfw/4)하는데, 이 식이 true
            division이라 halfh가 float이 되고 crop=False 경로의 resize가
            TypeError로 죽는다(11c 서버 traceback). StreamVGGT 자체 eval은
            비정사각형 벤치마크(ScanNet/7-Scenes 640x480, Sintel 1008x432)
            만 써서 이 가드를 밟지 않았다. square_ok=True는 VGGT demo의
            공식 컨벤션(518x518 = 14x37 patch 정합 유지)이며 GT(640x640,
            1:1)와 aspect가 일치해 eval 정합도 유지된다.

        반환:
            depths       (N, H, W) float32
            depth_confs  (N, H, W)
            pts3d        (N, H, W, 3)  모델 world 좌표계 point map
            pts3d_confs  (N, H, W)
            extri        (N, 3, 4)     w2c extrinsic (pose encoding에서 복원)
            intri        (N, 3, 3)     FoV 기반 근사 intrinsics
            img_size_hw  (H, W)
            frame_names  파일명 리스트
            seconds      추론 소요 시간
        """
        from dust3r.utils.image import load_images_for_eval

        paths = [str(p) for p in image_paths]
        if not paths:
            raise ValueError("empty image list")
        images = load_images_for_eval(paths, size=size, square_ok=square_ok, verbose=False, crop=crop)

        views = []
        for i, im in enumerate(images):
            views.append(
                {
                    "img": im["img"].to(self.device),                      # [1,C,H,W], [-1,1]
                    "ray_map": torch.full(
                        (im["img"].shape[0], 6, im["img"].shape[-2], im["img"].shape[-1]),
                        torch.nan,
                        device=self.device,
                    ),
                    "true_shape": torch.from_numpy(im["true_shape"]).to(self.device),
                    "idx": i,
                    "instance": str(i),
                    "camera_pose": torch.eye(4, dtype=torch.float32, device=self.device)[None],
                    "img_mask": torch.tensor(True, device=self.device)[None],
                    "ray_mask": torch.tensor(False, device=self.device)[None],
                    "update": torch.tensor(True, device=self.device)[None],
                    "reset": torch.tensor(False, device=self.device)[None],
                }
            )
        for v in views:
            v["img"] = (v["img"] + 1.0) / 2.0  # StreamVGGT 입력 규격 [0,1]

        # StreamVGGT 평가 코드와 동일한 autocast 정책(capability >= 8 -> bf16)
        if self.device.startswith("cuda"):
            dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability(self.device)[0] >= 8
                else torch.float16
            )
            ctx = torch.cuda.amp.autocast(dtype=dtype)
        else:
            ctx = torch.autocast(device_type="cpu", enabled=False)

        t0 = time.time()
        with torch.no_grad(), ctx:
            outputs = self.model.inference(views)
        seconds = time.time() - t0

        ress = outputs.ress
        depths, dconfs, pts3d, pconfs, encs = [], [], [], [], []
        for r in ress:
            depths.append(_squeeze_np(r["depth"]))
            dconfs.append(_squeeze_np(r["depth_conf"]))
            pts3d.append(_squeeze_np(r["pts3d_in_other_view"]))
            pconfs.append(_squeeze_np(r["conf"]))
            encs.append(np.squeeze(r["camera_pose"].detach().float().cpu().numpy()))  # (9,)

        img_size_hw = depths[0].shape
        extri, intri = self._decode_pose(encs, img_size_hw)
        frame_names = [Path(p).name for p in paths]
        return {
            "depths": np.stack(depths, 0),
            "depth_confs": np.stack(dconfs, 0),
            "pts3d": np.stack(pts3d, 0),
            "pts3d_confs": np.stack(pconfs, 0),
            "extri": extri,
            "intri": intri,
            "img_size_hw": np.array(img_size_hw, dtype=np.int32),
            "frame_names": np.array(frame_names),
            "seconds": float(seconds),
        }
