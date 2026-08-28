"""시퀀스 단위 예측 결과 npz 저장/로딩."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_predictions(path: str | Path, preds: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in preds.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays)
    return path


def load_predictions(path: str | Path) -> dict:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
