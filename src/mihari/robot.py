"""ロボットとのやり取りの約束（実機・シミュレーション・録画再生で共通）。

見張り番の判断の流れ（``patrol.Sentry``）は、この約束だけを使う。実機（``robot_ws.WsRobot``）、
横長の写真から首の向きごとの景色を切り出すシミュレーション（``robot_sim.PanoramaRobot``）、
実機で撮っておいた写真を流し込む録画再生（``robot_sim.ReplayRobot``）を同じ流れで回せる。
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .rules import EnvReading


@dataclass
class Frame:
    """本体カメラの静止画 1 枚。

    Attributes:
        jpeg: JPEG のバイト列。
        pan: 撮った時に指示していた pan。
        tilt: 同じく tilt。
        yaw_deg: 本体が測った首の向き（度・本体の座標）。分からなければ None。
        pitch_deg: 同じく上下。
        captured_at: 受け取った時刻（単調時計の秒）。
    """

    jpeg: bytes
    pan: int
    tilt: int
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    captured_at: float = 0.0
    _image: np.ndarray | None = field(default=None, repr=False, compare=False)

    def image(self) -> np.ndarray:
        """RGB の配列（H, W, 3・uint8）。初回に JPEG を展開して覚えておく。"""
        if self._image is None:
            with Image.open(io.BytesIO(self.jpeg)) as im:
                self._image = np.asarray(im.convert("RGB"))
        return self._image


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """RGB の配列を JPEG にする。"""
    buf = io.BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class IrNotArmedError(RuntimeError):
    """赤外線の実送信が許されていない（``--enable-ir`` が無い）のに送ろうとした。"""


class IrForbiddenError(RuntimeError):
    """環境変数 ``MIHARI_FORBID_IR=1``（テスト中）なのに送ろうとした。"""


def check_ir_allowed(armed: bool) -> None:
    """赤外線を送ってよいかの最後の関所。実機の送信口は必ずこれを通る。

    Args:
        armed: ``--enable-ir`` が付いているか。

    Raises:
        IrForbiddenError: ``MIHARI_FORBID_IR=1`` の時（テストはこれを立てて走る）。
        IrNotArmedError: armed が False の時。
    """
    if os.environ.get("MIHARI_FORBID_IR") == "1":
        raise IrForbiddenError("MIHARI_FORBID_IR=1: infrared sending is forbidden here (tests)")
    if not armed:
        raise IrNotArmedError("infrared sending needs --enable-ir")


class Robot(Protocol):
    """見張り番が使うロボットの約束。"""

    name: str
    #: 実機か（記録に残す・実機の写真はリポジトリの外に置く判断に使う）。
    is_real: bool

    async def connect(self) -> None:
        """繋ぐ（実機は母艦のサーバーへ。シミュレーションは何もしない）。"""

    async def close(self) -> None:
        """切る。"""

    async def look(self, pan: int, tilt: int) -> None:
        """首を ``(pan, tilt)`` へ向ける。"""

    async def capture(self) -> Frame:
        """いまの向きの静止画を 1 枚撮る（首を動かした後は落ち着くのを待ってから）。"""

    def latest_env(self) -> EnvReading | None:
        """最新の温湿度（まだ無ければ None）。"""

    async def show_face(self, expressions: Sequence[str], hold_s: float) -> bool:
        """顔の表情で知らせる。できなければ False。"""

    async def say(self, text: str) -> bool:
        """声で知らせる。できなければ False。"""

    async def send_ir(self, payload: dict, *, armed: bool) -> dict:
        """赤外線の指令を送る（``check_ir_allowed`` を通る）。本体の返事を返す。"""
