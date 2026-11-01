"""テストの共通の道具。

- テストの間は ``MIHARI_FORBID_IR=1`` を立てる: 実機の送信口（``check_ir_allowed``）が必ず例外を上げるので、
  テストから赤外線が出ることは無い
- 合成の部屋（なめらかな模様の横長の画像）と、マゼンタの「人」を描く道具
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def forbid_ir(monkeypatch: pytest.MonkeyPatch) -> None:
    """どのテストでも赤外線の実送信を禁止する。"""
    monkeypatch.setenv("MIHARI_FORBID_IR", "1")


def make_room(width: int = 1536, height: int = 1024, seed: int = 7) -> np.ndarray:
    """位相限定相関が効く、なめらかで繰り返しの無い模様の横長の画像（RGB・uint8）。

    マゼンタ（R・B が高く G が低い）にならないよう G を高めに保つ。
    """
    rng = np.random.default_rng(seed)
    small = rng.random((height // 16 + 2, width // 16 + 2, 3))
    from PIL import Image

    img = Image.fromarray((small * 255).astype(np.uint8)).resize((width, height), Image.BICUBIC)
    arr = np.asarray(img).astype(np.int32)
    arr[..., 1] = np.clip(arr[..., 1] // 2 + 110, 0, 255)  # G を 110 以上に
    return arr.astype(np.uint8)


def add_person(room: np.ndarray, center_x: int, *, width: int = 70, top: float = 0.35, bottom: float = 0.8) -> np.ndarray:
    """横長の画像の center_x に、マゼンタの縦長の「人」を描いた写しを返す。"""
    out = room.copy()
    h = out.shape[0]
    x1 = max(0, center_x - width // 2)
    x2 = min(out.shape[1], center_x + width // 2)
    out[int(h * top): int(h * bottom), x1:x2] = (255, 0, 255)
    return out


async def no_sleep(_seconds: float) -> None:
    """待たない sleep（うなずきの間合いを飛ばす）。"""
    return None
