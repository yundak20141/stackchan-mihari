"""実機: 母艦の中継サーバーにスマホ役として繋ぎ、スタックチャン本体を動かす。

スタックチャン本体（M5Stack CoreS3）は母艦 PC の WebSocket サーバーに常時繋がっている。
このプログラムはそのサーバーへ「スマホ（role=phone）」として繋ぎ、次のメッセージでやり取りする
（サーバーとファームは既存のもの。このリポジトリはそれを使う側だけを持つ）:

- 送る: ``hello``（role=phone）／``monitor_start``・``monitor_stop``（監視モードの静止画の配信）／
  ``pose``（首の pan・tilt）／``set_mode``・``expr``（顔で知らせる時）／``speak``（声で知らせる時）／
  ``ir_send``（赤外線。``--enable-ir`` の時だけ）
- 受ける: ``accepted``／``fw_status``（本体が繋がっているか）／``env_status``（温湿度・10 秒ごと）／
  ``monitor_frame``（静止画の JPEG を base64 で）／``ir_ack``／``speak_done``

サーバーは首が動いた後 1.5 秒待ってから撮り直すので、``look`` の後は ``settle_s`` 以上たってから
届いた写真だけを「その向きの写真」として受け取る。
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, Callable, Sequence

from .config import PAN_CENTER, TILT_CENTER
from .geometry import clamp_pan, clamp_tilt
from .robot import Frame, check_ir_allowed
from .rules import EnvReading

logger = logging.getLogger(__name__)


class DeviceOfflineError(RuntimeError):
    """母艦のサーバーには繋がったが、スタックチャン本体がサーバーに繋がっていない。"""


class CaptureTimeoutError(TimeoutError):
    """決めた時間内に静止画が届かなかった。"""


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class WsRobot:
    """母艦の中継サーバー経由で本体を動かす実機のロボット。

    Args:
        url: サーバーの URL（例 ``ws://<母艦の IP>:8080``。.env の MIHARI_SERVER_URL）。
        settle_s: 首を動かしてから、この秒より後に届いた写真だけを受け取る。
        interval_ms: 静止画の間隔（サーバーの最小は 1000）。
        frame_timeout_s: 写真を待つ上限の秒。
        connect_timeout_s: 繋ぐ時に待つ上限の秒。
        require_device: True なら本体がサーバーにいない時に ``DeviceOfflineError``。
        clock: 単調時計（秒）。
    """

    name = "stackchan"
    is_real = True

    def __init__(
        self,
        url: str,
        *,
        settle_s: float = 1.2,
        interval_ms: int = 1000,
        frame_timeout_s: float = 10.0,
        connect_timeout_s: float = 6.0,
        require_device: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url
        self.settle_s = settle_s
        self.interval_ms = interval_ms
        self.frame_timeout_s = frame_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.require_device = require_device
        self.clock = clock
        self.pan = PAN_CENTER
        self.tilt = TILT_CENTER
        self.device_connected: bool | None = None
        self._ws: Any = None
        self._reader: asyncio.Task | None = None
        self._accepted = asyncio.Event()
        self._fw_status_seen = asyncio.Event()
        self._frames: asyncio.Queue[tuple[float, dict]] = asyncio.Queue()
        self._env: EnvReading | None = None
        self._pose_at: float | None = None
        self._streaming = False
        self._waiters: dict[str, list[asyncio.Future]] = {}
        self._closed = asyncio.Event()
        #: 送ったメッセージの記録（type だけ・テストと記録用）。
        self.sent_types: list[str] = []

    # ───────── 接続 ─────────

    async def connect(self) -> None:
        """サーバーへ繋ぎ、hello を送って accepted と fw_status を待ち、静止画の配信を始める。

        Raises:
            DeviceOfflineError: ``require_device`` で、本体がサーバーにいない時。
            TimeoutError: サーバーが応えない時。
        """
        from websockets.asyncio.client import connect

        self._ws = await asyncio.wait_for(
            connect(self.url, max_size=8 * 1024 * 1024, open_timeout=self.connect_timeout_s),
            self.connect_timeout_s + 1,
        )
        self._reader = asyncio.create_task(self._read_loop())
        await self._send({"type": "hello", "role": "phone", "device": "mihari-sentry", "protocol_version": 2})
        await asyncio.wait_for(self._accepted.wait(), self.connect_timeout_s)
        try:
            await asyncio.wait_for(self._fw_status_seen.wait(), 3.0)
        except asyncio.TimeoutError:
            pass
        if self.require_device and not self.device_connected:
            await self.close()
            raise DeviceOfflineError(
                "母艦のサーバーには繋がりましたが、スタックチャン本体が繋がっていません（電源と Wi-Fi を確かめてください）"
            )
        await self.start_stream()

    async def start_stream(self) -> None:
        """監視モードの静止画の配信を頼む（サーバーが本体を monitor モードにする）。"""
        await self._send({"type": "monitor_start", "interval_ms": int(self.interval_ms)})
        self._streaming = True

    async def stop_stream(self) -> None:
        """静止画の配信を止める。"""
        if self._ws is not None and self._streaming:
            await self._send({"type": "monitor_stop"})
        self._streaming = False

    async def close(self) -> None:
        """配信を止めて切る。"""
        try:
            await self.stop_stream()
        except Exception:  # noqa: BLE001
            pass
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._ws = None

    async def _send(self, payload: dict) -> None:
        if self._ws is None:
            raise ConnectionError("not connected")
        self.sent_types.append(str(payload.get("type")))
        await self._ws.send(json.dumps(payload, ensure_ascii=False))

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue  # スマホ役には binary は来ない（来ても使わない）
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict):
                    self._on_message(msg)
        except Exception as exc:  # noqa: BLE001
            logger.info("[ws] reader stopped: %s", exc)
        finally:
            self._closed.set()
            for futures in self._waiters.values():
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(ConnectionError("connection closed"))

    def _on_message(self, msg: dict) -> None:
        kind = msg.get("type")
        now = self.clock()
        if kind == "accepted":
            self._accepted.set()
        elif kind == "fw_status":
            self.device_connected = bool(msg.get("connected"))
            self._fw_status_seen.set()
        elif kind == "env_status":
            self._env = EnvReading(
                temp_c=_num(msg.get("temp_c")),
                humidity=_num(msg.get("humidity")),
                pressure_hpa=_num(msg.get("pressure_hpa")),
                unit=str(msg.get("unit", "unknown")),
                received_at=now,
            )
        elif kind == "monitor_frame":
            # 古い写真は 3 枚まで（首を回している間に溜まった分を捨てる）。
            while self._frames.qsize() >= 3:
                self._frames.get_nowait()
            self._frames.put_nowait((now, msg))
        elif kind in ("ir_ack", "speak_done", "mode_change_result"):
            waiters = self._waiters.get(kind) or []
            while waiters:
                fut = waiters.pop(0)
                if not fut.done():
                    fut.set_result(msg)
                    break

    async def _wait_for(self, kind: str, timeout: float) -> dict:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(kind, []).append(fut)
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            if fut in self._waiters.get(kind, []):
                self._waiters[kind].remove(fut)

    # ───────── 首とカメラ ─────────

    async def look(self, pan: int, tilt: int) -> None:
        self.pan = clamp_pan(pan)
        self.tilt = clamp_tilt(tilt)
        await self._send({"type": "pose", "pan": self.pan, "tilt": self.tilt})
        self._pose_at = self.clock()

    async def capture(self) -> Frame:
        """首を動かしてから ``settle_s`` 以上たって届いた静止画を 1 枚返す。

        Raises:
            CaptureTimeoutError: ``frame_timeout_s`` 以内に届かない時。
            ConnectionError: 接続が切れた時。
        """
        threshold = (self._pose_at or 0.0) + self.settle_s
        deadline = self.clock() + self.frame_timeout_s
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise CaptureTimeoutError(
                    f"no monitor_frame within {self.frame_timeout_s:.0f} s (device connected={self.device_connected})"
                )
            if self._closed.is_set() and self._frames.empty():
                raise ConnectionError("connection closed")
            try:
                received_at, msg = await asyncio.wait_for(self._frames.get(), min(remaining, 0.5))
            except asyncio.TimeoutError:
                continue
            if received_at < threshold:
                continue
            try:
                jpeg = base64.b64decode(msg.get("jpeg_b64") or "", validate=True)
            except (ValueError, TypeError):
                continue
            if not jpeg.startswith(b"\xff\xd8"):
                continue
            return Frame(
                jpeg=jpeg, pan=self.pan, tilt=self.tilt,
                yaw_deg=_num(msg.get("yaw_deg")), pitch_deg=_num(msg.get("pitch_deg")),
                captured_at=received_at,
            )

    def latest_env(self) -> EnvReading | None:
        return self._env

    # ───────── 知らせる ─────────

    async def show_face(self, expressions: Sequence[str], hold_s: float) -> bool:
        """顔で知らせる: 配信を止め、見た目が Home のまま首を明け渡す follow モードにして表情を出す。

        監視画面（monitor）では表情の指示が受け付けられないため、いったんモードを替える。
        次の見回しで配信を再開すると、サーバーが本体を monitor モードに戻す。
        """
        await self.stop_stream()
        await self._send({"type": "set_mode", "mode": "follow"})
        await asyncio.sleep(1.5)
        await self._send({"type": "pose", "pan": self.pan, "tilt": self.tilt})
        for name in expressions:
            await self._send({"type": "expr", "name": name})
            await asyncio.sleep(hold_s)
        await self.start_stream()
        return True

    async def say(self, text: str) -> bool:
        """声で知らせる（サーバーに声の合成がつながっている時だけ鳴る。返事の tts_ms > 0 で判定）。"""
        request_id = uuid.uuid4().hex[:12]
        await self._send({"type": "speak", "voiceText": text, "requestId": request_id})
        try:
            done = await self._wait_for("speak_done", 20.0)
        except asyncio.TimeoutError:
            return False
        return bool(_num(done.get("tts_ms")) or 0)

    # ───────── 赤外線 ─────────

    async def send_ir(self, payload: dict, *, armed: bool) -> dict:
        """赤外線の指令を本体へ送り、``ir_ack`` を待つ（``check_ir_allowed`` を必ず通る）。"""
        check_ir_allowed(armed)
        await self._send(dict(payload))
        try:
            return await self._wait_for("ir_ack", 5.0)
        except asyncio.TimeoutError:
            return {"type": "ir_ack", "ok": False, "reason": "timeout"}
