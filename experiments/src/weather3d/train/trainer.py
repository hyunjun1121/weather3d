"""C3 증류 트레이너: student StreamVGGT + teacher VGGT(같은 ckpt).

13/13b 서베이로 확정된 upstream 인터페이스를 그대로 미러한다:
- student/teacher 모두 third_party/StreamVGGT/ckpt/checkpoints.pth에서
  strict=True 로드(13b compat 프로브: 양쪽 키 100% 일치).
- 증류 호출 모양은 src/dust3r/inference.py loss_of_one_batch와 동일:
  model(batch, query_pts) -> output.ress / teacher.inference(batch, query_pts)
  -> knowledge.ress / criterion(gts, preds) -> (loss, details).
  차이는 딱 하나 — teacher에 clean 뷰, student에 clean/열화 뷰를 준다
  (upstream은 같은 batch를 양쪽에 준다).
- criterion은 config/train.yaml의 `DistillLoss()` 생성자를 그대로 사용.
- autocast 정책도 loss_of_one_batch와 동일(capability >= 8이 아니면
  fp16). Accelerator mixed_precision은 "fp16"(sm_75; bf16 하드코딩 대체).
- freeze는 src/train.py와 동일: aggregator.patch_embed / camera_token /
  register_token만 고정.
- 14d 서버 실패 대책 2건: (1) losses.py가 정의 없이 호출하는
  check_and_fix_inf_nan을 _patch_missing_helpers로 주입(NameError),
  (2) teacher fp16 상주 + 창 축소(--num-views)로 공유 GPU 메모리 예산
  대응(CUDA OOM — 47GiB 중 타 프로세스 ~12GiB 상시 점유).
- 14e/14f 서버 실패 확정: teacher 가중치 fp16 상주는 upstream에서
  막힌다. VGGT.inference는 head 4개를 autocast(enabled=False)로 실행하며
  입력 캐스팅이 없어(14e: camera_head token_norm에서 fp32 토큰 x fp16
  가중치), 입력 img를 fp16로 통일해도 DPTHead._apply_pos_embed이
  position_grid_to_embed에 dtype을 전달하지 않아 fp32 pos_embed가
  섞이고 x + pos_embed 승격으로 resize_layers에서 다시 죽는다(14f:
  dpt_head.py:234 conv_transpose2d, 서버 traceback + upstream 원문
  확인). trainer 경계에서 고칠 수 없는 third_party 내부 결함이므로
  fp16 teacher 경로를 전면 제거하고 teacher는 항상 순수 fp32
  (autocast 완전 차단)로 실행한다 — 서버 c0~c2 추론과 동일한 구성.
  메모리는 --num-views 4 창 축소로만 확보한다(14d 실측 views=8
  34.98GiB -> views=4 예상 ~28GiB, 공유 GPU 여유 ~36GiB).
- 14g 서버 결과: dtype 경로는 해결 — 스텝 2·4가 loss 유한(peak 27.0GiB,
  예산 내)으로 통과했다. 그러나 스텝 5~6의 teacher DPT refinenet에서
  CUDA OOM(260MiB 할당 실패, 프로세스 34.82GiB). 배치 shape은 고정
  (4뷰 518x392)인데도 할당이 자라는 동적 요인이 겹쳤다: (a) teacher
  forward가 student 활성화 그래프가 살아있는 시점에 실행돼 피크가
  중첩됨, (b) details의 off-path 그래프 조각이 다음 스텝 호출 시점까지
  참조로 살아있음, (c) 게이트 통과 직후 타 프로세스가 1->12.3GiB로
  증가(공유 GPU 변동 — 게이트는 검사 시점만 보장). 대책: teacher-first
  재배열(distill_teacher 분리), 스텝 말미 del loss/details/gts, 로그에
  현재 alloc/res 필드 추가, 서버 스크립트에서
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, teacher 단계 OOM
  스킵(DDP는 allreduce로 전 rank 동시 스킵; student/backward 단계 OOM은
  DDP 대칭 깨짐 위험으로 즉시 중단).
- 14h 서버 결과: 단일 GPU 스모크 12스텝 완주 + reload 통과 — 스텝 4 이후
  현재 alloc이 16.0GiB로 평탄(재배열+참조 해제 효과 실증, oom_skips=0,
  peak 33.55GiB). DDP는 4스텝 정상 학습 뒤 step 5 student 단계에서 OOM
  즉시 중단(설계 동작): DDP 필요치는 33.55 + grad bucket 3.53(952M x 4B)
  + NCCL 버퍼 ~1.5 = 약 38.6GiB/GPU인데, 게이트 기준 32GiB로 골라도
  런 중반 타 프로세스 복귀(gpu0 49.1 -> 36.6GiB 여유)로 물리적으로 부족했다.
  코드 결함이 아니라 DDP 메모리 예산의 재산정이 필요하다 — 14i는 서버
  스크립트에서 DDP 게이트를 40GiB로 올리고, R1 장시간 실행의 안전망인
  --resume(load_resume_state: trainer_state.pth에서 model+optimizer+step
  복원)을 2단계 스모크로 최초 검증한다.
- 14i 서버 결과: smokeA 6스텝 -> smokeB resume 6->12 재기동 완주로
  resume 안전망 서버 실증. ddp는 co-tenant 4장 상시 점유로 [SKIP]
  free-mem(40960 x 2 미충족 — 코드 정상, GPU 여유 확보 후 재실행).
- 14b 준비(2026-08-27): (1) trainer_state.pth가 '항상' optimizer를 담도록
  수정 — 주기 저장이 모멘트 없는 상태로 덮어써 중단-이어학습 시 AdamW
  상태가 유실되던 문제(R1 감독 런처의 resume 정확성에 필수). 저장은
  tmp->rename 원자적 저장으로 바꿔 장시간 런에서 저장 도중 크래시
  손상을 방지한다. (2) R3 대조용 LoRA 경로: inject_lora가 aggregator
  attention qkv/proj를 LoRALinear로 교체(자체 구현 — peft 의존 없음),
  base 전체 동결 + low-rank만 학습, eval 저장은 merge_lora_into로 비래퍼
  키로 병합해 기존 평가(StreamVGGTRunner flat 로드) 호환을 유지한다.
  resume용 trainer_state는 래퍼 원본 구조를 그대로 담는다.
- 14j 최적화(2026-08-27, 성능·메모리 한정 — 학습 수학 불변): §8 사전등록
  "Turing은 fp16 amp + grad checkpointing"의 GC를 이제 이행한다. 실측
  (14h)은 단일 GPU peak 33.55GiB/steady 16.0, DDP 필요치 약 38.6GiB
  (peak + grad bucket 3.53 + NCCL 1.5)로 40960MiB 게이트가 co-tenant
  ~12.3GiB 점유 중엔 영구 미달이었다(14i ddp [SKIP] 원인). 신규 플래그
  (기본 off = 종래 동작, 14i 검증 경로 회귀 없음):
  --grad-checkpoint(aggregator blocks + depth/point/camera head를
  use_reentrant=False checkpoint로 wrap, grad 꺼진 호출은 원본 경로),
  --ddp-bucket-view(DDP grad bucket과 .grad 저장 공유로 전이 피크
  절감), --foreach-optimizer(14d의 foreach=False는 GC 없던 시절 메모리
  대책 — GC로 예산 확보 뒤 속도 회수), --ddp-no-find-unused(track_head
  동결 + find_unused 탐색 제거. query_points=None 학습 경로에서
  track_head는 절대 실행되지 않아 gradient가 항상 None이므로 동결해도
  학습 수학은 불변이다. 로그에 peak_teacher_gib/peak_student_gib를 추가해
  33.55GiB 피크의 귀속(teacher vs student)을 처음 분해한다.
  --num-views 기본값을 8→4로 내린다(8은 14d 실측 34.98GiB OOM).
  14j 스크립트가 A/B 측정으로 recommendation.json을 만들고 14b가 이를
  적용한다(측정 없이는 14b 기본값이 자동으로 바뀌지 않는다).

실행(서버):
  cd experiments && PYTHONPATH=src python -m weather3d.train.trainer \
    --mode r1 --config configs/c3_data.yaml \
    --svggt-src ../third_party/StreamVGGT/src \
    --ckpt ../third_party/StreamVGGT/ckpt/checkpoints.pth \
    --out outputs/c3_train/r1_smoke --max-steps 12
DDP: accelerate launch --multi_gpu --num_processes N (같은 인자).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# 첫 CUDA 할당 전에 반영돼야 하므로 torch import 앞에 둔다(서버 14h+ 실증
# 환경 torch >= 2.1). 서버 스크립트가 이미 설정했다면 setdefault가 값을 존중한다.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="StreamVGGT weather C3 distillation trainer")
    p.add_argument("--mode", choices=["r1", "r2"], default="r1",
                   help="r1: clean 50%% + degraded 50%% replay / r2: degraded only")
    p.add_argument("--config", required=True, help="c3_data.yaml 경로")
    p.add_argument("--svggt-src", required=True, help="third_party/StreamVGGT/src")
    p.add_argument("--ckpt", required=True, help="checkpoints.pth 경로")
    p.add_argument("--out", required=True, help="출력 디렉터리")
    p.add_argument("--num-views", type=int, default=4,
                   help="창 뷰 수. 8은 14d 실측 34.98GiB OOM이라 서버 예산 기준 4")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=6600)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--min-lr", type=float, default=1e-7)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit-windows", type=int, default=0, help="스모크용 창 수 제한")
    p.add_argument("--max-frames-per-seq", type=int, default=0)
    p.add_argument("--mixed-precision", default="fp16", choices=["fp16", "bf16", "no"])
    p.add_argument("--probe-batch", type=int, default=0,
                   help=">0면 학습 없이 N개 배치 구조만 검사하고 종료")
    p.add_argument("--oom-skip-limit", type=int, default=20,
                   help="OOM 스킵 누적 한도(초과 시 emergency 저장 후 중단)")
    p.add_argument("--lora-r", type=int, default=0,
                   help=">0이면 LoRA(R3): rank. 0이면 전체 파인튜닝(R1/R2)")
    p.add_argument("--lora-alpha", type=float, default=16.0,
                   help="LoRA scale = alpha / r")
    p.add_argument("--resume", default="", help="trainer_state.pth 경로(선택)")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="aggregator blocks + depth/point/camera head를 "
                        "use_reentrant=False checkpoint로 wrap(§8 사전등록 이행). "
                        "재계산 비용으로 스텝은 느려지지만 student 활성화 피크가 줄어든다")
    p.add_argument("--ddp-bucket-view", action="store_true",
                   help="DDP gradient_as_bucket_view=True — grad bucket과 .grad "
                        "저장을 공유해 DDP 전이 피크를 줄인다")
    p.add_argument("--foreach-optimizer", action="store_true",
                   help="AdamW foreach=True. 기본(False)은 14d OOM 대책이던 "
                        "transient 버퍼 회피 — GC로 예산 확보 뒤 속도 회수용")
    p.add_argument("--ddp-no-find-unused", action="store_true",
                   help="find_unused_parameters=False + track_head 동결. "
                        "query_points=None 경로에서 track_head는 절대 실행되지 "
                        "않아 학습 수학 불변 — DDP 미사용 param 탐색 비용 제거")
    return p.parse_args()


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    t = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    t = min(1.0, max(0.0, t))
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * t))


def _stub_dead_imports() -> None:
    """dust3r/losses.py 모듈 상단의 미사용 import를 통과시킨다(14 실패 원인).

    upstream(wzzheng/StreamVGGT) losses.py는 gsplat.rasterization을
    18행에서 import만 하고 파일 어디에서도 호출하지 않는다(upstream main
    대조 + DistillLoss 본문 14 survey3 확인: CameraLoss/DepthOrPmapLoss/
    TrackLoss와 헬퍼는 전부 순수 torch). lpips(20행)는 RGBLoss 전용이다.
    서버 .venv에 둘 중 무엇이 없어도 DistillLoss 학습 경로는 영향이 없으므로
    import만 stub으로 통과시킨다. stub이 실제 사용되면 즉시 예외로 실패해
    잘못된 stub 사용이 조용히 넘어가지 않는다.
    """
    try:
        import gsplat  # noqa: F401
    except ImportError:
        import types

        def _unavailable(*args, **kwargs):
            raise RuntimeError(
                "gsplat.rasterization was called but gsplat is not installed "
                "(dead import in dust3r/losses.py)"
            )

        stub = types.ModuleType("gsplat")
        stub.rasterization = _unavailable
        sys.modules["gsplat"] = stub
    try:
        import lpips  # noqa: F401
    except ImportError:
        import types

        sys.modules["lpips"] = types.ModuleType("lpips")


def _patch_missing_helpers() -> None:
    """losses.py가 정의 없이 호출하는 check_and_fix_inf_nan을 주입(14d 실패).

    upstream(wzzheng/StreamVGGT main) losses.py는 CameraLoss.forward에서
    check_and_fix_inf_nan을 3회 호출(loss_T/loss_R/loss_FL)하지만 파일 안에
    정의도 import도 존재하지 않는다(서버 런타임 NameError + upstream raw
    소스 대조 + survey2 def 구조 grep에서 해당 이름 부재). 공개 저장소
    어디에도 정의를 찾지 못해 호출 계약(x = f(x, "tag"), inf/nan을 0으로
    정화)에 맞춰 재구성한다. torch.where는 유한 성분의 gradient를 보존한다.
    """
    import dust3r.losses as losses_mod

    if hasattr(losses_mod, "check_and_fix_inf_nan"):
        return

    def _check_and_fix_inf_nan(x, tag=None):
        if torch.isinf(x).any():
            print(f"[DistillLoss/{tag}] Inf detected -> zeros")
            x = torch.where(torch.isinf(x), torch.zeros_like(x), x)
        if torch.isnan(x).any():
            print(f"[DistillLoss/{tag}] NaN detected -> zeros")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return x

    losses_mod.check_and_fix_inf_nan = _check_and_fix_inf_nan


def move_views(views: list[dict], device: torch.device) -> list[dict]:
    out = []
    for v in views:
        v2 = {}
        for k, x in v.items():
            v2[k] = x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
        out.append(v2)
    return out


def distill_teacher(teacher, batch_teacher):
    """teacher 증류 목표 계산(14g OOM 대책으로 student 앞으로 이동).

    순수 fp32 no_grad + autocast 완전 차단 — 서버 c0~c2 추론과 동일한
    구성이며 upstream head 내부의 fp32 승격과 dtype이 일치한다(14e/14f).
    student 활성화 그래프가 아직 없는 시점에 실행해 teacher DPT 전이
    피크와의 중첩을 원천 제거한다(14g: refinenet에서 27->33GiB 성장
    후 OOM). DDP 통신이 없는 단계라 rank 간 독립 스킵 후 재동기가
    안전하다(main 루프 참조).
    """
    query_pts = None
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        knowledge = teacher.inference(batch_teacher, query_pts)
    # teacher가 순수 fp32면 .float()은 무해한 no-op이고, valid_mask(bool)은
    # is_floating_point 대상이 아니라 그대로 통과한다.
    return [
        {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in r.items()}
        for r in knowledge.ress
    ]


def distill_step(student, criterion, batch_student, gts, device):
    """student forward + criterion(teacher 결과 gts는 선계산).

    student는 autocast(fp16) 안에서 forward(upstream 미러). criterion은
    autocast 밖(fp32)에서 돈다 — gts가 이미 fp32라 승격으로 통일된다.
    """
    query_pts = None
    cap = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 0
    dtype = torch.bfloat16 if cap >= 8 else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=cap > 0):
        output = student(batch_student, query_pts)
    preds = output.ress
    with torch.autocast(device_type="cuda", enabled=False):
        loss, details = criterion(gts, preds)
    return loss, details


def enable_grad_checkpoint(model: nn.Module) -> list[str]:
    """student의 학습 활성화를 use_reentrant=False checkpoint로 감싼다(§8).

    대상: aggregator 직속 nn.ModuleList 블록 전원 + depth/point/camera head.
    third_party 소스를 수정하지 않기 위해 forward를 런타임 교체로 wrap한다.
    torch.is_grad_enabled()가 꺼진 호출(teacher/평가)은 원본 경로를 그대로
    통과해 checkpoint 오버헤드가 없다. 비재진입 checkpoint는 autocast 상태와
    kwargs를 보존해 LoRA·DDP·fp16 autocast와 호환된다. 감싼 모듈 이름
    리스트를 반환(서버 스크립트가 개수를 게이트로 검증).
    """
    import torch.utils.checkpoint as cp

    def wrap(mod: nn.Module, name: str) -> str:
        orig = mod.forward

        def fwd(*args, **kwargs):
            if torch.is_grad_enabled():
                return cp.checkpoint(orig, *args, use_reentrant=False, **kwargs)
            return orig(*args, **kwargs)

        mod.forward = fwd
        return name

    targets: list[tuple[nn.Module, str]] = []
    agg = getattr(model, "aggregator", None)
    if agg is not None:
        for attr, child in list(agg.named_children()):
            if isinstance(child, nn.ModuleList):
                targets.extend(
                    (blk, f"aggregator.{attr}.{i}") for i, blk in enumerate(child)
                )
    for head in ("depth_head", "point_head", "camera_head"):
        mod = getattr(model, head, None)
        if mod is not None:
            targets.append((mod, head))
    return [wrap(mod, name) for mod, name in targets]


def freeze_track_head(model: nn.Module) -> int:
    """track_head 파라미터를 동결한다(--ddp-no-find-unused와 세트).

    학습 경로(query_points=None)에서 track_head는 절대 실행되지 않아
    gradient가 항상 None이다. 동결해도 손실·갱신 수학은 그대로(AdamW는
    grad None 파라미터를 건너뛴다). DDP find_unused_parameters의 스텝마다
    미사용 param 탐색을 없애기 위한 것. 동결한 파라미터 수를 반환.
    """
    head = getattr(model, "track_head", None)
    if head is None:
        return 0
    n = sum(p.numel() for p in head.parameters())
    for p in head.parameters():
        p.requires_grad_(False)
    return n


def load_resume_state(student, optimizer, path: str) -> int:
    """--resume 공통 로딩(prepare 전 raw student/optimizer 대상).

    trainer_state.pth(model + step + [optimizer])에서 이어 학습한다.
    accelerator.prepare로 DDP 래핑되기 '전'에 호출해 module. 접두어 없이
    키가 정합한다. 저장돼 있던 스텝을 반환한다(스케줄은 set_lr(step)이
    전역 스텝으로 이어 계산). 공유 GPU 장시간 실행(14b R1)에서 런 중
    OOM/중단 후 이어가는 안전망 — 14i 스모크로 최초 검증.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    student.load_state_dict(state["model"], strict=True)
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("step", 0))


class LoRALinear(nn.Module):
    """nn.Linear 래퍼 LoRA(R3 대조 실험). base는 동결, low-rank만 학습.

    peft 의존 없는 자체 구현 — 서버 .venv에 새 패키지를 추가하지 않는다.
    forward = base(x) + (alpha/r) * B(A(x)). lora_b를 0으로 초기화해
    학습 시작 시점에는 base와 동치다(즉 R3 시작점 = 사전학습 ckpt).
    eval 저장은 merge_lora_into가 base.weight + (alpha/r) * B@A 로 접어
    비래퍼 키를 유지한다(기존 평가 파이프라인 호환).
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


def inject_lora(root: nn.Module, r: int, alpha: float, attr_names=("qkv", "proj")) -> int:
    """root 아래 attr_names 이름의 nn.Linear를 LoRALinear로 교체.

    대상은 aggregator attention의 qkv/proj(실험맥락 §8 'LoRA r=8~16,
    attention'). named_modules를 미리 스냅샷해 교체 중 이중 래핑을 막는다.
    교체한 Linear 수를 반환. R3 호출부는 이후 전체 동결 + lora만 해제한다.
    """
    count = 0
    for _name, parent in list(root.named_modules()):
        for attr in attr_names:
            child = getattr(parent, attr, None)
            if isinstance(child, nn.Linear):
                setattr(parent, attr, LoRALinear(child, r, alpha))
                count += 1
    return count


def merge_lora_into(sd: dict, model: nn.Module) -> dict:
    """sd(래퍼 포함 state_dict 사본)의 LoRA 키를 병합해 flat 키로 바꾼
    새 dict를 반환한다(입력 sd 불변). 래퍼가 없으면 sd를 그대로 반환.

    ...qkv.base.weight / .lora_a.weight / .lora_b.weight -> ...qkv.weight
    = base.weight + scale * (B @ A). 결과 키 집합이 비래퍼 모델과 동일해
    strict 로드·기존 평가(StreamVGGTRunner)가 그대로 동작한다.
    """
    has_lora = any(isinstance(m, LoRALinear) for _n, m in model.named_modules())
    if not has_lora:
        return sd
    out = dict(sd)
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            delta = (mod.lora_b.weight.detach() @ mod.lora_a.weight.detach()) * mod.scale
            out[f"{name}.weight"] = (mod.base.weight.detach() + delta).cpu().clone()
            if mod.base.bias is not None:
                out[f"{name}.bias"] = mod.base.bias.detach().cpu().clone()
            for junk in (
                f"{name}.base.weight",
                f"{name}.base.bias",
                f"{name}.lora_a.weight",
                f"{name}.lora_b.weight",
            ):
                out.pop(junk, None)
    return out


def _atomic_save(obj, path) -> None:
    """tmp 쓰기 후 rename — 장시간 런에서 저장 도중 크래시 시 파손 방지."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def probe_batch(args, cfg) -> int:
    """데이터셋만 구성해 배치 구조를 JSON로 덤프(모델 불필요, CPU)."""
    from weather3d.train.data import WeatherTrainDataset

    rows = []
    ds = None
    for attempt in range(3):
        ds = WeatherTrainDataset(
            cfg,
            num_views=args.num_views,
            stride=args.stride,
            clean_ratio=0.0 if args.mode == "r2" else 0.5,
            seed=args.seed + attempt,
            limit_windows=max(args.probe_batch, args.limit_windows),
            max_frames_per_seq=args.max_frames_per_seq,
        )
        rows = []
        for i in range(min(args.probe_batch, len(ds))):
            vs, vt, meta = ds[i]
            img_s = vs[0]["img"]
            img_t = vt[0]["img"]
            depth = vt[0]["depthmap"]
            rows.append(
                dict(
                    **meta,
                    keys=sorted(vs[0].keys()),
                    img_shape=list(img_s.shape),
                    img_range=[float(img_s.min()), float(img_s.max())],
                    student_teacher_mad=float((img_s - img_t).abs().mean()),
                    depth_valid_frac=float((depth > 0).float().mean()),
                    pose_finite=bool(torch.isfinite(vt[0]["camera_pose"]).all()),
                    valid_mask_present="valid_mask" in vs[0],
                    valid_mask_shape=list(vs[0]["valid_mask"].shape) if "valid_mask" in vs[0] else [],
                    valid_mask_dtype=str(vs[0]["valid_mask"].dtype) if "valid_mask" in vs[0] else "",
                )
            )
        # clean/degraded가 각 1개 이상 나오도록(모두 한쪽이면 시드 바꿔 재시도).
        if len({r["mode"] for r in rows}) >= 2:
            break
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_batch.json").write_text(
        json.dumps(dict(num_windows=len(ds), num_seqs=len(ds.seqs), batches=rows), indent=2),
        encoding="utf-8",
    )
    print(f"PROBE BATCH OK windows={len(ds)} seqs={len(ds.seqs)} probed={len(rows)}")
    for r in rows:
        print(
            f"  [{r['idx']:>4}] {r['mode']:<8} {r['kind']:<8} sev={r['severity']:.2f} "
            f"shape={r['img_shape']} range=[{r['img_range'][0]:.3f},{r['img_range'][1]:.3f}] "
            f"mad={r['student_teacher_mad']:.4f} depth_valid={r['depth_valid_frac']:.2f}"
        )
    return 0


def main() -> int:
    args = parse_args()

    svggt_src = str(Path(args.svggt_src).resolve())
    if svggt_src not in sys.path:
        sys.path.insert(0, svggt_src)

    from weather3d.train.data import WeatherTrainDataset, load_train_config

    cfg = load_train_config(args.config)
    if args.probe_batch > 0:
        return probe_batch(args, cfg)

    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    _stub_dead_imports()
    from dust3r.losses import DistillLoss
    from streamvggt.models.streamvggt import StreamVGGT
    from vggt.models.vggt import VGGT

    _patch_missing_helpers()

    find_unused = not args.ddp_no_find_unused
    if args.ddp_bucket_view:
        try:
            ddp_kwargs = DistributedDataParallelKwargs(
                find_unused_parameters=find_unused, gradient_as_bucket_view=True
            )
        except TypeError:
            # accelerate 구버전 폴백 — 기능 미적용이지 구동 불능이 아니어야 한다.
            print("[warn] DistributedDataParallelKwargs에 gradient_as_bucket_view 없음 — 기본 kwargs")
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)
    else:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused)
    # accelerate 1.14(서버 .venv)는 Accelerator.__init__에서 even_batches
    # 직접 인자를 제거했다(14c 실패: TypeError). DataLoaderConfiguration으로
    # 전달할 수 있으나 기본값 True가 DDP uneven-epoch 패딩 보호라 그대로 쓴다.
    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device
    # 학습 입력 shape은 고정(views x [1,3,392,518])이라 benchmark가 안전하다.
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed + 1000 * accelerator.process_index)
    np.random.seed(args.seed + 1000 * accelerator.process_index)

    out = Path(args.out)
    if accelerator.is_main_process:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log_path = out / "log.jsonl"

    if accelerator.is_main_process:
        print(f"[init] device={device} ranks={accelerator.num_processes} mp={args.mixed_precision}")

    # --- 모델: student/teacher 같은 ckpt strict(13b compat 확정) ---------
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    student = StreamVGGT()
    student.load_state_dict(ckpt, strict=True)
    teacher = VGGT()
    teacher.load_state_dict(ckpt, strict=True)
    del ckpt

    teacher.to(device)
    # teacher는 항상 순수 fp32(no autocast 블록에서만 호출). fp16 상주는
    # upstream dpt_head 내부 fp32 승격과 충돌해 14e/14f에서 연쇄 실패 —
    # 메모리는 --num-views 창 축소로 확보한다(모듈 docstring 참조).
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- freeze: src/train.py 미러 ------------------------------------
    for p in student.aggregator.patch_embed.parameters():
        p.requires_grad = False
    student.aggregator.camera_token.requires_grad = False
    student.aggregator.register_token.requires_grad = False

    # --- 14j: track_head 동결(학습 수학 불변, find_unused 탐색 제거) ---
    if args.ddp_no_find_unused:
        n_frozen = freeze_track_head(student)
        if accelerator.is_main_process:
            print(f"[model] track_head frozen params={n_frozen:,} (never runs: query_points=None)")

    # --- R3 LoRA: base 전체 동결 + attention qkv/proj low-rank만 학습 --
    if args.lora_r > 0:
        lora_wrapped = inject_lora(student.aggregator, args.lora_r, args.lora_alpha)
        for p in student.parameters():
            p.requires_grad_(False)
        for n_, p_ in student.named_parameters():
            if ".lora_a." in n_ or ".lora_b." in n_:
                p_.requires_grad_(True)
        if accelerator.is_main_process:
            print(f"[model] lora r={args.lora_r} alpha={args.lora_alpha} wrapped={lora_wrapped}")

    # --- 14j: gradient checkpointing(§8 사전등록 이행) ------------------
    if args.grad_checkpoint:
        gc_wrapped = enable_grad_checkpoint(student)
        if accelerator.is_main_process:
            preview = ", ".join(gc_wrapped[:8]) + ("..." if len(gc_wrapped) > 8 else "")
            print(f"[model] grad-checkpoint wrapped={len(gc_wrapped)} ({preview})")

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    if accelerator.is_main_process:
        print(f"[model] trainable={trainable:,}/{total:,} ({trainable / total:.2%}) teacher=float")

    criterion = DistillLoss().to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()  # 모델 로드 직후 파편화 정리(14d OOM 대책)

    # --- 옵티마이저: upstream AdamW(0.9, 0.95) + wd --------------------
    # requires_grad 목록은 optimizer와 clip_grad_norm_이 공유한다(스텝마다
    # 1.2B 전체 파라미터를 다시 필터링하지 않는다). freeze/GC/LoRA는 모두
    # 이 지점 앞에서 끝난다.
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    param_groups = [
        {"params": trainable_params, "weight_decay": args.weight_decay, "lr": args.lr},
    ]
    # foreach=False(기본): trainable 952M의 transient 버퍼를 피해 스텝 피크를
    # 낮춘다(공유 GPU 예산, 14d OOM 대책). --foreach-optimizer는 GC 등으로
    # 예산을 확보한 뒤의 속도 회수용(14j A/B 측정으로 선택).
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.95), foreach=args.foreach_optimizer
    )

    ds = WeatherTrainDataset(
        cfg,
        num_views=args.num_views,
        stride=args.stride,
        clean_ratio=0.0 if args.mode == "r2" else 0.5,
        seed=args.seed,
        limit_windows=args.limit_windows,
        max_frames_per_seq=args.max_frames_per_seq,
    )
    if accelerator.is_main_process:
        print(f"[data] windows={len(ds)} seqs={len(ds.seqs)} mode={args.mode} views={args.num_views}")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda items: items[0],  # (views_student, views_teacher, meta) 통과
        persistent_workers=args.num_workers > 0,
    )

    start_step = 0
    if args.resume:
        start_step = load_resume_state(student, optimizer, args.resume)
        if accelerator.is_main_process:
            print(f"[resume] step={start_step}")

    loader, student, optimizer = accelerator.prepare(loader, student, optimizer)

    def set_lr(step: int) -> float:
        lr = lr_at(step, args)
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr

    def save_ckpt(fname: str, step: int) -> None:
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        model = accelerator.unwrap_model(student)
        # resume용 raw(래퍼 원본 구조)와 eval용 flat(LoRA 병합)을 분리.
        raw = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        flat = merge_lora_into(raw, model)
        _atomic_save(flat, out / fname)
        # trainer_state는 항상 optimizer를 담는다(14b 대책 — 주기 저장이
        # 모멘트 없는 상태로 덮어쓰면 중단-이어학습 시 AdamW 상태 유실).
        payload = {
            "model": raw,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "mode": args.mode,
            "args": vars(args),
        }
        _atomic_save(payload, out / "trainer_state.pth")

    step = start_step
    t0 = time.time()
    running: list[float] = []
    final_loss = float("nan")
    epoch = 0
    oom_skips = 0

    # 14j: 스테이지별 피크 분해. reset_peak_memory_stats는 allocated/reserved
    # 피크를 함께 초기화하므로 전역 피크는 수동 running max로 유지한다.
    stage_peaks = {"teacher": 0.0, "student": 0.0, "reserved": 0.0}

    def _stage_peak(stage: str) -> None:
        if device.type != "cuda":
            return
        stage_peaks[stage] = max(
            stage_peaks[stage], torch.cuda.max_memory_allocated(device) / 2**30
        )
        stage_peaks["reserved"] = max(
            stage_peaks["reserved"], torch.cuda.max_memory_reserved(device) / 2**30
        )
        torch.cuda.reset_peak_memory_stats(device)

    if accelerator.is_main_process:
        print(f"[train] start step={step} max={args.max_steps}")
    done = False
    while not done:
        ds.set_epoch(epoch)
        epoch += 1
        for views_student, views_teacher, meta in loader:
            views_student = move_views(views_student, device)
            views_teacher = move_views(views_teacher, device)

            # --- teacher 단계: DDP 통신이 없어 스킵이 rank 안전 ---------
            oom = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)  # teacher 창 시작
            try:
                gts = distill_teacher(teacher, views_teacher)
            except torch.OutOfMemoryError:
                oom = 1
                torch.cuda.empty_cache()
                print(f"[oom-skip] teacher stage step={step} (freed cache)")
            if accelerator.num_processes > 1:
                # 한 rank라도 teacher OOM이면 전 rank가 이 배치를 건너뛴다
                # (forward/backward 대칭 유지 — 비동기 스킵은 분산 hang).
                # all_reduce는 성공한 rank도 반드시 호출해야 한다.
                flag = torch.tensor([oom], device=device, dtype=torch.int64)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                oom = int(flag.item())
            _stage_peak("teacher")  # student 창 시작을 겸한다(reset)
            if oom:
                oom_skips += 1
                if oom_skips > args.oom_skip_limit:
                    if accelerator.is_main_process:
                        print(f"[abort] oom skips {oom_skips} > limit {args.oom_skip_limit}")
                        save_ckpt("emergency.pth", step)
                    return 1
                continue

            # --- student 단계: DDP에서 OOM 시 즉시 중단(스킵 불가) ------
            try:
                loss, details = distill_step(
                    student, criterion, views_student, gts, device
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                # DDP는 backward 대칭이 깨진 뒤의 스킵이 bucket 손상·hang을
                # 만든다. 단일 프로세스만 스킵 허용.
                if accelerator.num_processes > 1:
                    if accelerator.is_main_process:
                        print(f"[abort] student-stage OOM in DDP step={step}")
                        save_ckpt("emergency.pth", step)
                    return 1
                print(f"[oom-skip] student stage step={step} (single-proc)")
                oom_skips += 1
                if oom_skips > args.oom_skip_limit:
                    print(f"[abort] oom skips {oom_skips} > limit {args.oom_skip_limit}")
                    save_ckpt("emergency.pth", step)
                    return 1
                continue

            loss_value = float(loss)
            if not math.isfinite(loss_value):
                if accelerator.is_main_process:
                    print(f"[abort] non-finite loss {loss_value} at step {step}: {details}")
                    save_ckpt("emergency.pth", step)
                return 1
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _stage_peak("student")  # forward+backward+step 창 종료
            step += 1
            running.append(loss_value)
            final_loss = loss_value

            if step % args.log_every == 0 or step >= args.max_steps:
                lr = set_lr(step)
                elapsed = time.time() - t0
                mean_loss = sum(running) / len(running)
                running.clear()
                if accelerator.is_main_process:
                    # 전역 피크 = 스테이지 피크의 max(reset_peak을 쓰므로 수동).
                    mem = max(stage_peaks["teacher"], stage_peaks["student"])
                    mem_res = stage_peaks["reserved"]
                    alloc = (
                        torch.cuda.memory_allocated(device) / 2**30
                        if device.type == "cuda"
                        else 0.0
                    )
                    res = (
                        torch.cuda.memory_reserved(device) / 2**30
                        if device.type == "cuda"
                        else 0.0
                    )
                    line = dict(
                        step=step,
                        loss=loss_value,
                        loss_mean=mean_loss,
                        lr=lr,
                        mode=meta["mode"],
                        kind=meta["kind"],
                        steps_per_sec=step / max(elapsed, 1e-9),
                        peak_mem_gib=round(mem, 2),
                        peak_mem_reserved_gib=round(mem_res, 2),
                        peak_teacher_gib=round(stage_peaks["teacher"], 2),
                        peak_student_gib=round(stage_peaks["student"], 2),
                        alloc_gib=round(alloc, 2),
                        reserved_gib=round(res, 2),
                        oom_skips=oom_skips,
                        details={k: float(v) for k, v in list(details.items())[:12]},
                    )
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(line) + "\n")
                    print(
                        f"[step {step:>6}] loss={loss_value:.4f} mean={mean_loss:.4f} "
                        f"lr={lr:.2e} {meta['mode']}/{meta['kind']} "
                        f"peak={mem:.1f}GiB alloc={alloc:.1f}GiB "
                        f"{step / max(elapsed, 1e-9):.2f} it/s"
                    )
            else:
                set_lr(step)

            if step % args.save_every == 0:
                save_ckpt(f"step{step:07d}.pth", step)
            # 14g OOM 대책: 이번 스텝의 참조를 여기서 끊는다. details의
            # off-path 그래프 조각이 다음 스텝 teacher forward까지 살아
            # 있던 것을 차단(로그 블록은 위에서 이미 details를 소비).
            del loss, details, gts
            if step >= args.max_steps:
                done = True
                break

    accelerator.wait_for_everyone()
    save_ckpt("final_model.pth", step)
    if accelerator.is_main_process:
        print(f"TRAIN SMOKE OK steps={step} final_loss={final_loss:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
