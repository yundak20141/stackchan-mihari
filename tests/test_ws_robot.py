"""実機の口（WsRobot）のテスト: 母艦の中継サーバーの「スマホ向けの約束」を真似た偽のサーバーで確かめる。

偽のサーバーは hello に accepted と fw_status と env_status を返し、monitor_start の間は
「いまの首の向き」の景色（合成の部屋から切り出す）を monitor_frame で送り続ける。
"""
from __future__ import annotations

import asyncio
import base64
import json
import time

import pytest
from websockets.asyncio.server import serve

from conftest import add_person, make_room, no_sleep
from mihari.config import Settings
from mihari.detection import MarkerDetector
from mihari.patrol import Sentry
from mihari.robot import IrForbiddenError
from mihari.robot_sim import PanoramaRobot
from mihari.robot_ws import DeviceOfflineError, WsRobot


class FakeBridge:
    """juiz-app の ws_server のスマホ向けの振る舞いの最小の写し。"""

    def __init__(self, room, *, device_connected: bool = True, env: dict | None = None,
                 frame_interval_s: float = 0.05) -> None:
        self.device = PanoramaRobot(room, pan_right_sign=1)
        self.device_connected = device_connected
        self.env = env if env is not None else {"temp_c": 29.2, "humidity": 48.0, "pressure_hpa": 1012.3,
                                                "unit": "env3", "comfort": "hot"}
        self.frame_interval_s = frame_interval_s
        self.received: list[dict] = []
        self.port = 0
        self._streaming = False

    def types(self) -> list[str]:
        return [m.get("type") for m in self.received]

    async def handler(self, ws) -> None:
        stream_task = None
        try:
            async for raw in ws:
                msg = json.loads(raw)
                self.received.append(msg)
                kind = msg.get("type")
                if kind == "hello":
                    await ws.send(json.dumps({"type": "accepted", "protocol_version": 2, "server": "juiz-app"}))
                    await ws.send(json.dumps({"type": "fw_status", "connected": self.device_connected}))
                    if self.device_connected:
                        await ws.send(json.dumps({"type": "env_status", **self.env}))
                elif kind == "monitor_start":
                    self._streaming = True
                    if stream_task is None:
                        stream_task = asyncio.create_task(self._stream(ws))
                elif kind == "monitor_stop":
                    self._streaming = False
                elif kind == "pose":
                    await self.device.look(int(msg["pan"]), int(msg["tilt"]))
        finally:
            if stream_task is not None:
                stream_task.cancel()

    async def _stream(self, ws) -> None:
        request_id = 0
        while True:
            await asyncio.sleep(self.frame_interval_s)
            if not self._streaming:
                continue
            frame = await self.device.capture()
            request_id += 1
            await ws.send(json.dumps({
                "type": "monitor_frame", "request_id": request_id,
                "jpeg_b64": base64.b64encode(frame.jpeg).decode("ascii"),
                "yaw_deg": 90 - self.device.pan, "pitch_deg": 33, "frame_acquired_ms": request_id * 50,
                "protocol_version": 2,
            }))


async def _with_bridge(bridge: FakeBridge, body):
    async with serve(bridge.handler, "127.0.0.1", 0) as server:
        bridge.port = server.sockets[0].getsockname()[1]
        return await body(f"ws://127.0.0.1:{bridge.port}")


def test_connect_hello_env_and_capture_after_settle() -> None:
    bridge = FakeBridge(make_room())

    async def body(url):
        robot = WsRobot(url, settle_s=0.2, frame_timeout_s=3.0)
        await robot.connect()
        try:
            assert robot.device_connected is True
            env = robot.latest_env()
            assert (env.temp_c, env.humidity, env.unit) == (29.2, 48.0, "env3")
            await robot.look(110, 82)
            sent_at = time.monotonic()
            frame = await robot.capture()
            assert frame.captured_at - sent_at >= 0.2 - 0.02   # 落ち着くのを待ってから受け取った
            assert frame.pan == 110
            assert frame.yaw_deg == -20.0
            assert frame.image().shape == (240, 320, 3)
        finally:
            await robot.close()

    asyncio.run(_with_bridge(bridge, body))
    first = bridge.received[0]
    assert first["type"] == "hello" and first["role"] == "phone"
    assert bridge.types()[:2] == ["hello", "monitor_start"]
    assert {"type": "pose", "pan": 110, "tilt": 82} in bridge.received
    assert bridge.types()[-1] == "monitor_stop"


def test_device_offline_is_reported() -> None:
    bridge = FakeBridge(make_room(), device_connected=False)

    async def body(url):
        with pytest.raises(DeviceOfflineError):
            await WsRobot(url, settle_s=0.1).connect()

    asyncio.run(_with_bridge(bridge, body))
    assert "monitor_start" not in bridge.types()


def test_full_patrol_over_websocket_never_sends_ir() -> None:
    room = add_person(make_room(), 1150)
    bridge = FakeBridge(room)
    truth = bridge.device.pan_of_x(1150)

    async def body(url):
        robot = WsRobot(url, settle_s=0.12, frame_timeout_s=3.0)
        sentry = Sentry(robot, MarkerDetector(), Settings(), sleep=no_sleep)
        await robot.connect()
        try:
            return await sentry.patrol_once()
        finally:
            await robot.close()

    report = asyncio.run(_with_bridge(bridge, body))
    assert report.person_present
    assert report.target_pan == pytest.approx(truth, abs=2.5)
    assert report.ac_decision.action == "cool"
    assert report.ir_outcome.status == "dry_run"
    assert "ir_send" not in bridge.types()      # 偽のサーバーに赤外線の指令は 1 度も届いていない
    pans = [m["pan"] for m in bridge.received if m.get("type") == "pose"]
    assert pans[:5] == [50, 70, 90, 110, 130]


def test_ws_send_ir_is_blocked_in_tests() -> None:
    bridge = FakeBridge(make_room())

    async def body(url):
        robot = WsRobot(url, settle_s=0.1)
        await robot.connect()
        try:
            with pytest.raises(IrForbiddenError):
                await robot.send_ir({"type": "ir_send", "kind": "raw", "name": "ac_cool"}, armed=True)
        finally:
            await robot.close()

    asyncio.run(_with_bridge(bridge, body))
    assert "ir_send" not in bridge.types()
