"""実機の無い時のロボット: シミュレーションと録画再生。

- ``PanoramaRobot``: 横長の写真（部屋のパノラマ）から、pan に応じた位置を切り出して「首の向きごとの
  カメラ画像」にする。振り向いた後の写真もその角度で切り出すので、「人が正面に来たか」まで確かめられる
- ``ReplayRobot``: 実機で ``mihari record`` が保存した写真と温湿度を流し込む。指示された pan に一番近い
  写真を返す

どちらも赤外線は送らない（送ろうとした中身を ``ir_log`` に残すだけ）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from .config import PAN_MAX, PAN_MIN, TILT_CENTER
from .geometry import clamp_pan, clamp_tilt
from .robot import Frame, check_ir_allowed, encode_jpeg
from .rules import EnvReading


def load_image(path: Path) -> np.ndarray:
    """画像ファイルを RGB の配列で読む。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_env_json(path: Path) -> dict:
    """温湿度の JSON（``{"temp_c":..,"humidity":..}``）を読む。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class SimCall:
    """シミュレーションのロボットが受けた指示の記録（テストで確かめる）。"""

    kind: str
    args: tuple


class _SimBase:
    """シミュレーションと録画再生に共通の部分。"""

    is_real = False

    def __init__(self, env: dict | None, clock: Callable[[], float]) -> None:
        self._env = env
        self.clock = clock
        self.pan = 90
        self.tilt = TILT_CENTER
        self.calls: list[SimCall] = []
        self.ir_log: list[dict] = []

    def set_env(self, env: dict | None) -> None:
        """流し込む温湿度を差し替える（何周かで温度を変える試験用）。"""
        self._env = env

    async def connect(self) -> None:
        self.calls.append(SimCall("connect", ()))

    async def close(self) -> None:
        self.calls.append(SimCall("close", ()))

    async def look(self, pan: int, tilt: int) -> None:
        self.pan = clamp_pan(pan)
        self.tilt = clamp_tilt(tilt)
        self.calls.append(SimCall("look", (self.pan, self.tilt)))

    def latest_env(self) -> EnvReading | None:
        if self._env is None:
            return None
        return EnvReading(
            temp_c=self._env.get("temp_c"),
            humidity=self._env.get("humidity"),
            pressure_hpa=self._env.get("pressure_hpa"),
            unit=str(self._env.get("unit", "sim")),
            received_at=self.clock(),
        )

    async def show_face(self, expressions: Sequence[str], hold_s: float) -> bool:
        self.calls.append(SimCall("face", (tuple(expressions), hold_s)))
        return True

    async def say(self, text: str) -> bool:
        self.calls.append(SimCall("say", (text,)))
        return True

    async def send_ir(self, payload: dict, *, armed: bool) -> dict:
        check_ir_allowed(armed)
        self.ir_log.append(dict(payload))
        self.calls.append(SimCall("ir", (dict(payload),)))
        return {"type": "ir_ack", "ok": True, "name": payload.get("name") or payload.get("protocol"), "sim": True}


class PanoramaRobot(_SimBase):
    """横長の写真から首の向きごとの景色を切り出すシミュレーション。

    Args:
        panorama: 部屋の横長の写真（RGB の配列）。
        env: 流し込む温湿度（``{"temp_c":..,"humidity":..}``）。None なら温湿度なし。
        view_frac: 1 枚の写真が横長の写真の幅の何割に当たるか（画角を決める）。
        out_size: 出力する写真の大きさ（幅, 高さ）。本体カメラと同じ 320x240 が既定。
        pan_right_sign: pan を増やすと右を向くなら +1、左なら -1（取り付けの向きの再現）。
        view_center_y: 切り出しの縦の中心（横長の写真の高さに対する割合。立った人の頭が入るよう少し上）。
        clock: 時計（テストで差し替える）。
    """

    name = "sim-panorama"

    def __init__(
        self,
        panorama: np.ndarray,
        *,
        env: dict | None = None,
        view_frac: float = 0.42,
        out_size: tuple[int, int] = (320, 240),
        pan_right_sign: int = 1,
        view_center_y: float = 0.42,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(env, clock)
        if pan_right_sign not in (-1, 1):
            raise ValueError("pan_right_sign must be -1 or 1")
        self.panorama = panorama
        self.out_size = out_size
        self.pan_right_sign = pan_right_sign
        self.view_center_y = view_center_y
        h, w = panorama.shape[:2]
        self.view_w = view_frac * w
        self.view_h = min(h, self.view_w * out_size[1] / out_size[0])
        self._px_per_deg = (w - self.view_w) / (PAN_MAX - PAN_MIN)

    @property
    def hfov_deg(self) -> float:
        """このシミュレーションの画角（度）。"""
        return self.view_w / self._px_per_deg

    def view_center_x(self, pan: float) -> float:
        """pan の時の切り出しの中心（横長の写真の画素）。"""
        w = self.panorama.shape[1]
        t = (pan - PAN_MIN) * self._px_per_deg
        left_end = self.view_w / 2
        return left_end + t if self.pan_right_sign > 0 else (w - left_end) - t

    def pan_of_x(self, x: float) -> float:
        """横長の写真の横位置 x を正面に見る pan（テストの正解に使う）。"""
        w = self.panorama.shape[1]
        left_end = self.view_w / 2
        t = (x - left_end) if self.pan_right_sign > 0 else ((w - left_end) - x)
        return PAN_MIN + t / self._px_per_deg

    async def capture(self) -> Frame:
        h, w = self.panorama.shape[:2]
        cx = self.view_center_x(self.pan)
        x1 = int(round(cx - self.view_w / 2))
        x1 = max(0, min(w - int(round(self.view_w)), x1))
        x2 = x1 + int(round(self.view_w))
        y1 = int(round(self.view_center_y * h - self.view_h / 2))
        y1 = max(0, min(h - int(round(self.view_h)), y1))
        y2 = y1 + int(round(self.view_h))
        crop = self.panorama[y1:y2, x1:x2]
        resized = np.asarray(Image.fromarray(crop).resize(self.out_size, Image.BILINEAR))
        self.calls.append(SimCall("capture", (self.pan, self.tilt)))
        return Frame(jpeg=encode_jpeg(resized), pan=self.pan, tilt=self.tilt, captured_at=self.clock())


class ReplayRobot(_SimBase):
    """実機で保存した写真と温湿度を流し込む録画再生。

    録画のディレクトリには ``session.json``（``{"frames":[{"pan":50,"tilt":82,"file":"view_050.jpg"},...],
    "env":{...}}``）と写真がある（``mihari record`` が作る）。

    Args:
        directory: 録画のディレクトリ。
        env_override: 温湿度を差し替える時に渡す。
        clock: 時計。
    """

    name = "replay"

    def __init__(
        self,
        directory: Path,
        *,
        env_override: dict | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        directory = Path(directory)
        session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
        super().__init__(env_override if env_override is not None else session.get("env"), clock)
        self.directory = directory
        self.frames = [
            (int(item["pan"]), int(item.get("tilt", TILT_CENTER)), directory / item["file"],
             item.get("yaw_deg"), item.get("pitch_deg"))
            for item in session["frames"]
        ]
        if not self.frames:
            raise ValueError(f"no frames in {directory}")

    async def capture(self) -> Frame:
        pan, tilt, path, yaw, pitch = min(self.frames, key=lambda f: (abs(f[0] - self.pan), abs(f[1] - self.tilt)))
        self.calls.append(SimCall("capture", (self.pan, self.tilt)))
        return Frame(jpeg=Path(path).read_bytes(), pan=pan, tilt=tilt, yaw_deg=yaw, pitch_deg=pitch,
                     captured_at=self.clock())
