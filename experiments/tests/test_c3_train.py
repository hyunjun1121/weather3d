"""C3 학습 데이터 파이프라인 단위 테스트(numpy 로직 위주, 모델 불필요).

대상: experiments/src/weather3d/train/data.py의 순수 로직
(severity 보간, 날씨 종류 분포, 노이즈 볼륨, TrainSplit 파싱).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from weather3d.train.data import (  # noqa: E402
    SEVERITY_RANGES,
    WEATHER_MIX,
    lerp,
    noise_volume,
    pick_kind,
    sample_weather_params,
)


def test_lerp_endpoints():
    assert lerp(1.0, 3.0, 0.0) == 1.0
    assert lerp(1.0, 3.0, 1.0) == 3.0
    assert abs(lerp(1.0, 3.0, 0.5) - 2.0) < 1e-12


def test_severity_ranges_match_eval_presets():
    # 끝점은 ext_v2.yaml eval 프리셋과 일치해야 한다.
    assert SEVERITY_RANGES["fog"]["beta"] == (0.04, 0.64)
    assert SEVERITY_RANGES["smoke"]["beta"] == (0.03, 0.12)
    assert SEVERITY_RANGES["smoke"]["sigma"] == (0.05, 0.20)
    assert SEVERITY_RANGES["rain"]["density"] == (150.0, 900.0)
    assert SEVERITY_RANGES["rain"]["length"] == (12.0, 26.0)
    assert SEVERITY_RANGES["lowlight"]["gamma"] == (1.6, 3.0)
    assert SEVERITY_RANGES["lowlight"]["gain"] == (0.55, 0.16)


def test_sample_weather_params_endpoints_and_gain_direction():
    p_lo = sample_weather_params("fog", 0.0)
    p_hi = sample_weather_params("fog", 1.0)
    assert abs(p_lo["beta"] - 0.04) < 1e-12
    assert abs(p_hi["beta"] - 0.64) < 1e-12
    # gain은 severity가 오르면 줄어든다(끝점 순서 반대).
    g_lo = sample_weather_params("lowlight", 0.1)["gain"]
    g_hi = sample_weather_params("lowlight", 0.9)["gain"]
    assert g_lo > g_hi
    mid = sample_weather_params("rain", 0.5)
    assert 150 < mid["density"] < 900
    with np.testing.assert_raises(ValueError):
        sample_weather_params("snow", 0.5)


def test_pick_kind_distribution():
    rng = np.random.default_rng(0)
    n = 20000
    counts = {}
    for _ in range(n):
        k = pick_kind(rng)
        counts[k] = counts.get(k, 0) + 1
    assert set(counts) == set(WEATHER_MIX)
    for k, p in WEATHER_MIX.items():
        assert abs(counts[k] / n - p) < 0.02, (k, counts[k] / n, p)


def test_noise_volume_shape_range_determinism():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = noise_volume(96, 128, 8, rng1, cell=24, tcell=2)
    b = noise_volume(96, 128, 8, rng2, cell=24, tcell=2)
    assert a.shape == (8, 96, 128)
    assert a.dtype == np.float32
    assert a.min() >= 0.0 and a.max() <= 1.0
    np.testing.assert_allclose(a, b)


def test_noise_volume_temporal_correlation():
    # 인접 프레임은 멀리 떨어진 프레임보다 유사해야 한다(시간 상관).
    rng = np.random.default_rng(7)
    v = noise_volume(96, 128, 16, rng, cell=24, tcell=4)
    near = float(np.abs(v[8] - v[9]).mean())
    far = float(np.abs(v[8] - v[15]).mean())
    assert near < far, (near, far)


def test_train_split_parsing(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        scene = root / "office"
        scene.mkdir()
        (scene / "TrainSplit.txt").write_text(
            "office/seq-01/\noffice/seq-03/\noffice/seq-06/\n", encoding="utf-8"
        )
        for seq in ("seq-01", "seq-02", "seq-03", "seq-06"):
            d = scene / seq
            d.mkdir()
            (d / "frame-000000.color.png").write_bytes(b"")
        # _seven_scene_seqs 경로: TrainSplit에 없는 seq-02는 제외되고
        # 프레임이 있는 seq만 남는다.
        from weather3d.train.data import _seven_scene_seqs

        seqs = _seven_scene_seqs(root, ["office"])
        assert [s.key for s in seqs] == ["7scenes/office/seq-01", "7scenes/office/seq-03", "7scenes/office/seq-06"]
        # TrainSplit 없음 -> 전체 seq.
        (scene / "TrainSplit.txt").unlink()
        seqs = _seven_scene_seqs(root, ["office"])
        assert len(seqs) == 4


def test_window_index_and_limits():
    """WeatherTrainDataset 창 인덱스 로직(디스크 없이 시퀀 목록만 구성)."""
    from weather3d.train.data import TrainSeq, WeatherTrainDataset

    seq = TrainSeq(
        key="7scenes/x/seq-01",
        dataset="seven_scenes",
        seq_dir=Path("/nonexistent"),
        frame_names=[f"frame-{i:06d}.color" for i in range(100)],
    )
    ds = WeatherTrainDataset.__new__(WeatherTrainDataset)
    ds.seqs = [seq]
    ds.num_views = 8
    ds.stride = 4
    ds.clean_ratio = 0.5
    ds.seed = 0
    ds.epoch = 0
    span = (8 - 1) * 4
    ds.hop = max(1, span // 2)
    ds.index = []
    n = len(seq.frame_names)
    for start in range(0, n - span, ds.hop):
        ds.index.append((0, start))
    # 100프레임, span=28, hop=14 -> start 0,14,...,70 -> 6개.
    assert len(ds.index) == 6
    si, start = ds.index[-1]
    assert start + span <= n - 1
    # 프레임 부족 시퀀스는 창이 없다.
    short = TrainSeq(key="s", dataset="seven_scenes", seq_dir=Path("/x"), frame_names=["a", "b"])
    span_ok = (8 - 1) * 4 < len(short.frame_names)
    assert not span_ok


def test_patch_missing_helpers():
    """trainer._patch_missing_helpers(14d NameError 수정).

    losses.py에 check_and_fix_inf_nan이 없으면 주입하고, 있으면 건드리지
    않는다. 주입된 함수는 inf/nan을 0으로 정화하되 유한 성분의 gradient를
    보존해야 한다(가짜 dust3r.losses 모듈로 sys.modules을 잠시 교체).
    """
    import sys
    import types

    import torch

    from weather3d.train import trainer as trainer_mod

    fake = types.ModuleType("dust3r.losses")
    pkg = types.ModuleType("dust3r")
    pkg.__path__ = []
    pkg.losses = fake
    saved = {k: sys.modules.get(k) for k in ("dust3r", "dust3r.losses")}
    sys.modules["dust3r"] = pkg
    sys.modules["dust3r.losses"] = fake
    try:
        assert not hasattr(fake, "check_and_fix_inf_nan")
        trainer_mod._patch_missing_helpers()
        f = fake.check_and_fix_inf_nan
        x = torch.tensor([1.0, float("inf"), float("nan")], requires_grad=True)
        y = f(x, "unit")
        assert torch.isfinite(y).all() and y[0] == 1.0
        y[0].backward()
        assert x.grad is not None and x.grad[0] == 1.0  # 유한 성분 grad 보존
        finite = f(torch.tensor([2.0, 3.0]), "unit")
        assert torch.equal(finite, torch.tensor([2.0, 3.0]))  # 정상 통과
        trainer_mod._patch_missing_helpers()  # 이미 있으면 미변경
        assert fake.check_and_fix_inf_nan is f
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_distill_teacher_detached():
    """distill_teacher(14g OOM 재배열로 분리).

    gts는 순수 fp32·no-grad(증류 목표는 상수), bool(valid_mask)은 그대로
    통과, float64 등 다른 부동 dtype은 fp32로 승격을 검증한다.
    teacher.inference는 query_pts=None 호출 계약도 확인한다.
    """
    import torch

    from weather3d.train.trainer import distill_teacher

    class _Preds:
        def __init__(self, ress):
            self.ress = ress

    class _Teacher:
        def inference(self, views, query_pts):
            assert query_pts is None
            return _Preds(
                [
                    dict(
                        img=torch.rand(1, 3, 4, 4),
                        depth=torch.rand(1, 4, 4),
                        conf=torch.rand(1, 4, 4, dtype=torch.float64),
                        valid_mask=torch.rand(1, 4, 4) > 0.5,
                        pose=torch.rand(1, 9),
                    )
                ]
            )

    gts = distill_teacher(_Teacher(), [dict(img=torch.rand(1, 3, 4, 4))])
    assert len(gts) == 1
    r = gts[0]
    for k in ("img", "depth", "conf", "pose"):
        assert r[k].dtype == torch.float32, k
        assert not r[k].requires_grad, k
    assert r["valid_mask"].dtype == torch.bool  # bool은 승격 대상 아님


def test_resume_roundtrip():
    """load_resume_state(14i): model+optimizer 상태·step·lr 복원 검증(CPU).

    서로 다른 초기값의 student/optimizer가 trainer_state.pth 로드 뒤 저장
    시점의 가중치·AdamW 모멘트·lr·step으로 정확히 복원되는지 확인한다.
    R1 장시간 실행의 중단-이어가기 안전망 검증.
    """
    import tempfile

    import torch

    from weather3d.train.trainer import load_resume_state

    torch.manual_seed(3)
    ref = torch.nn.Linear(4, 3)
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-3, foreach=False)
    for _ in range(2):  # exp_avg/exp_avg_sq 상태 생성
        loss = (ref(torch.randn(2, 4)) ** 2).sum()
        opt_ref.zero_grad()
        loss.backward()
        opt_ref.step()

    student = torch.nn.Linear(4, 3)  # 다른 초기값 — 이어받을 대상
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4, foreach=False)
    assert not torch.allclose(student.weight, ref.weight)

    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "trainer_state.pth")
        torch.save(
            {"model": ref.state_dict(), "optimizer": opt_ref.state_dict(), "step": 137},
            p,
        )
        step = load_resume_state(student, opt, p)
    assert step == 137
    assert torch.allclose(student.weight, ref.weight)
    assert torch.allclose(student.bias, ref.bias)
    # optimizer: lr(스케줄 연속성)과 1차 모멘트가 저장 상태에서 복원된다.
    assert abs(opt.param_groups[0]["lr"] - opt_ref.param_groups[0]["lr"]) < 1e-12
    key = next(iter(opt.state))
    a = opt.state[key].get("exp_avg")
    b = opt_ref.state[next(iter(opt_ref.state))].get("exp_avg")
    assert a is not None and b is not None and torch.allclose(a, b)
    # optimizer 키가 없는 저장물은 model+step만 복원(에러 없이).
    with tempfile.TemporaryDirectory() as td:
        p2 = str(Path(td) / "trainer_state.pth")
        torch.save({"model": ref.state_dict(), "step": 5}, p2)
        step2 = load_resume_state(student, opt, p2)
    assert step2 == 5
    assert torch.allclose(student.weight, ref.weight)


def test_lora_inject_freeze_and_merge():
    """R3 LoRA: 주입·동결·병합 평가 호환 검증(CPU).

    (1) inject_lora가 qkv/proj Linear만 교체, (2) base 동결 + low-rank만
    학습, (3) 초기 델타 0(base 동치), (4) merge_lora_into 결과 키가 비래퍼
    원본과 동일 + 수치 일치 + 병합 모델 forward 동치.
    """
    import torch
    import torch.nn as nn

    from weather3d.train.trainer import inject_lora, merge_lora_into

    class Attn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)

        def forward(self, x):
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            a = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
            return self.proj(a @ v)

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = nn.Sequential(Attn(8), Attn(8))

        def forward(self, x):
            return self.aggregator(x)

    torch.manual_seed(0)
    root = Root()
    sd0 = {k: v.clone() for k, v in root.state_dict().items()}  # 비래퍼 원본 키
    n = inject_lora(root.aggregator, r=4, alpha=8.0)
    assert n == 4, n  # Attn 2개 x qkv/proj
    # base 동결 + lora만 학습(LoRALinear.__init__가 base를 동결한다)
    for name, p in root.named_parameters():
        want = ".lora_a." in name or ".lora_b." in name
        assert p.requires_grad == want, (name, p.requires_grad)
    # 학습 시퀀스 재현: lora_a/b를 무작위로 흔들어 0이 아닌 델타 생성
    attn0 = root.aggregator[0]
    z = torch.randn(2, 8)
    with torch.no_grad():
        x = torch.randn(2, 8)
        # 초기(lora_b=0)에는 각 래퍼가 base와 동치다
        assert torch.allclose(attn0.qkv(z), attn0.qkv.base(z)), "qkv 초기 델타가 0이 아님"
        assert torch.allclose(attn0.proj(z), attn0.proj.base(z)), "proj 초기 델타가 0이 아님"
        for m in (attn0.qkv, attn0.proj, root.aggregator[1].qkv, root.aggregator[1].proj):
            m.lora_a.weight.normal_(0.0, 0.2)
            m.lora_b.weight.normal_(0.0, 0.2)
    raw = {k: v.detach().cpu() for k, v in root.state_dict().items()}
    flat = merge_lora_into(raw, root)
    # (1) 키 집합이 비래퍼 원본과 동일(eval strict 로드 호환)
    assert sorted(flat) == sorted(sd0), (sorted(flat), sorted(sd0))
    # (2) 병합 수치: base + scale*(B@A), bias 불변
    scale = 8.0 / 4.0
    assert torch.allclose(
        flat["aggregator.0.qkv.weight"],
        sd0["aggregator.0.qkv.weight"] + scale * (attn0.qkv.lora_b.weight @ attn0.qkv.lora_a.weight),
    )
    assert torch.allclose(flat["aggregator.0.qkv.bias"], sd0["aggregator.0.qkv.bias"])
    assert torch.allclose(flat["aggregator.1.proj.weight"],
                          sd0["aggregator.1.proj.weight"]
                          + scale * (root.aggregator[1].proj.lora_b.weight @ root.aggregator[1].proj.lora_a.weight))
    # (3) 병합 모델 forward == 래퍼 모델 forward
    ref_root = Root()
    ref_root.load_state_dict(flat)
    with torch.no_grad():
        assert torch.allclose(root(x), ref_root(x), atol=1e-5)
    # (4) 원본 sd는 불변(래퍼 키 그대로)
    assert any(".lora_a." in k for k in raw) and not any(".lora_a." in k for k in flat)


def test_merge_lora_passthrough_no_wrapper():
    """래퍼가 없으면 merge_lora_into는 sd를 그대로 반환(R1/R2 경로)."""
    import torch
    import torch.nn as nn

    from weather3d.train.trainer import merge_lora_into

    m = nn.Linear(4, 4)
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    out = merge_lora_into(sd, m)
    assert out is sd


def test_enable_grad_checkpoint():
    """14j GC wrap: 대상 선정·gradient 등가성·no_grad 우회·kwargs 통과(CPU).

    use_reentrant=False checkpoint는 재계산으로 동일 gradient를 만들어야
    한다(학습 수학 불변). grad가 꺼진 호출은 원본 경로를 그대로 통과해
    teacher/평가에 checkpoint 오버헤드가 없어야 한다.
    """
    import torch
    import torch.nn as nn

    from weather3d.train.trainer import enable_grad_checkpoint

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc = nn.Linear(d, d)

        def forward(self, x, mask=None):  # kwargs 통로 검증용
            h = self.fc(x)
            return h if mask is None else h * mask

    class Head(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc = nn.Linear(d, d)

        def forward(self, x, patch_start_idx=None):
            h = self.fc(x)
            return h if patch_start_idx is None else h + patch_start_idx

    class Agg(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([Block(6), Block(6)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = Agg()
            self.depth_head = Head(6)
            self.point_head = Head(6)
            self.camera_head = Head(6)
            self.track_head = Head(6)  # GC 대상 아님(학습 경로 미실행)

        def forward(self, x):
            for blk in self.aggregator.blocks:
                x = blk(x)
            return self.depth_head(x) + self.point_head(x) + self.camera_head(x)

    torch.manual_seed(0)
    model = Model()
    names = enable_grad_checkpoint(model)
    assert names == [
        "aggregator.blocks.0", "aggregator.blocks.1",
        "depth_head", "point_head", "camera_head",
    ], names
    assert "track_head" not in names

    # gradient 등가성: 같은 가중치로 GC on/off 각각 backward
    x = torch.randn(3, 6)
    model.zero_grad(set_to_none=True)
    model(x).pow(2).sum().backward()
    g_gc = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    ref = Model()
    ref.load_state_dict(model.state_dict())
    ref(x).pow(2).sum().backward()
    assert g_gc, "GC 경로에서 gradient가 하나도 나오지 않았다"
    for n, p in ref.named_parameters():
        if p.grad is None:  # track_head처럼 미실행 모듈은 양쪽 다 None
            assert n not in g_gc, n
            continue
        assert torch.allclose(g_gc[n], p.grad, atol=1e-6), n

    # no_grad 호출은 checkpoint 우회 — bitwise 동일
    with torch.no_grad():
        assert torch.equal(model(x), ref(x))

    # kwargs는 checkpoint를 그대로 통과한다(블록 mask, head patch_start_idx)
    x2 = torch.randn(2, 6, requires_grad=True)
    model.aggregator.blocks[0](x2, mask=torch.tensor(2.0)).sum().backward()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()
    x3 = torch.randn(2, 6, requires_grad=True)
    model.depth_head(x3, patch_start_idx=torch.tensor(1.0)).sum().backward()
    assert x3.grad is not None and torch.isfinite(x3.grad).all()


def test_freeze_track_head():
    """14j: track_head 동결 — 해당 모듈만 frozen, 학습 수학 불변 근거."""
    import torch
    import torch.nn as nn

    from weather3d.train.trainer import freeze_track_head

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.track_head = nn.Linear(4, 4)
            self.depth_head = nn.Linear(4, 4)

    m = Model()
    n = freeze_track_head(m)
    assert n == m.track_head.weight.numel() + m.track_head.bias.numel()
    assert not m.track_head.weight.requires_grad
    assert not m.track_head.bias.requires_grad
    assert m.depth_head.weight.requires_grad  # 다른 모듈은 불변
    assert freeze_track_head(nn.Linear(2, 2)) == 0  # track_head 없는 모델


def test_parse_defaults():
    """14j: CLI 기본값 — num_views=4(서버 예산 실측), 최적화 플래그 기본 off.

    플래그 기본 off는 14i가 검증한 종래 코드 경로의 회귀가 없음을 보장한다.
    """
    from weather3d.train import trainer as trainer_mod

    saved = sys.argv
    sys.argv = ["trainer", "--config", "c.yaml", "--svggt-src", "s",
                "--ckpt", "c.pth", "--out", "o"]
    try:
        args = trainer_mod.parse_args()
    finally:
        sys.argv = saved
    assert args.num_views == 4
    assert args.grad_checkpoint is False
    assert args.ddp_bucket_view is False
    assert args.foreach_optimizer is False
    assert args.ddp_no_find_unused is False
    assert args.mixed_precision == "fp16"


def main():  # run_all.py 호환(개별 실행용)
    for name in sorted(dir(sys.modules[__name__])):
        if name.startswith("test_"):
            getattr(sys.modules[__name__], name)()
            print(f"[PASS] {name}")


if __name__ == "__main__":
    main()
