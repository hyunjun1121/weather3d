"""weather3d: 3D-weather robustness experiment package.

악조건 날씨 합성 데이터로 3D 복원 모델(StreamVGGT)의 성능 붕괴와 회복을
측정하는 실험 코드. 설계 배경은 workspace readme.md의 실험 설계 v1.

구성:
- sequences/gt: 7-Scenes, Neural-RGBD 시퀀스 발견과 GT(depth/pose/intrinsics) 로딩
- synth: Track A 물리 기반 날씨 합성(atmospheric scattering fog/smoke)
- infer: StreamVGGT 추론 래퍼(third_party/StreamVGGT/src 재사용)
- eval: video depth / camera pose(ATE, RPE) / 재구성(Acc, Comp, NC) 지표
"""

__version__ = "0.1.0"
