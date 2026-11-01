"""偽の本体: 実機の代わりに母艦の中継サーバーへ「本体（device）」として繋ぐ試験の道具。

本物のスタックチャンが手元に無い時に、本物の中継サーバー（juiz-app の ws_server）と見張り番を
まるごと繋いで試すためのもの。本体ファームの次の振る舞いだけを真似る:

- 接続したら role を付けない ``hello`` を送る（サーバーは本体と見なす）
- ``mode_change_prepare`` に ``mode_change_ack``（ready=true）を返す
- ``visual_begin`` / ``visual_end`` に ``visual_ack`` を返す
- ``visual_capture`` に ``visual_image`` の見出しと JPEG の binary を返す（景色は横長の写真から
  いまの首の向きで切り出す＝``PanoramaRobot``）
- ``pose`` で首の向きを変える
- ``env_status`` を決まった間隔で送る
- ``ir_send`` は記録して ``ir_ack`` を返すだけ（偽物なので赤外線は出ない）

使い方::

    python -m mihari.fake_device --server ws://127.0.0.1:18080 \
        --panorama sample_data/room_person_left.jpg --env sample_data/env_hot.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from .robot_sim import PanoramaRobot, load_env_json, load_image

logger = logging.getLogger("mihari.fake_device")

#: ファームが visual_image に付ける校正の欄（juiz-app の検査が通る値）。
CALIBRATION = {"yaw_center_deg": 0, "pitch_center_deg": 33, "yaw_min_deg": -45, "yaw_max_deg": 45,
               "pitch_min_deg": 18, "pitch_max_deg": 58}


def comfort_of(temp: float, hum: float) -> str:
    """本体ファームと同じ順と境目の快適度。"""
    if temp < 17:
        return "cold"
    if temp >= 28:
        return "hot"
    if hum >= 70:
        return "humid"
    if hum < 35:
        return "dry"
    return "ok"


class FakeDevice:
    """偽の本体。

    Args:
        robot: 景色を作るシミュレーション。
        env: 送る温湿度。
        env_interval_s: env_status を送る間隔。
    """

    def __init__(self, robot: PanoramaRobot, env: dict, *, env_interval_s: float = 2.0) -> None:
        self.robot = robot
        self.env = env
        self.env_interval_s = env_interval_s
        self.t0 = time.monotonic()
        self.log: list[dict] = []

    def _ms(self) -> int:
        return int((time.monotonic() - self.t0) * 1000) % 2**32

    async def _env_loop(self, ws) -> None:
        while True:
            temp, hum = float(self.env["temp_c"]), float(self.env["humidity"])
            await ws.send(json.dumps({
                "type": "env_status", "unit": self.env.get("unit", "sim"), "temp_c": temp, "humidity": hum,
                "pressure_hpa": self.env.get("pressure_hpa"), "comfort": comfort_of(temp, hum), "age_ms": 0,
            }))
            await asyncio.sleep(self.env_interval_s)

    async def handle(self, ws, msg: dict) -> None:
        """サーバーからの 1 メッセージに本体のように応える。"""
        kind = msg.get("type")
        self.log.append({k: v for k, v in msg.items() if k != "jpeg_b64"})
        if kind == "mode_change_prepare":
            await ws.send(json.dumps({"type": "mode_change_ack", "protocol_version": 2, "ready": True}))
            logger.info("[fake] mode -> %s", msg.get("mode"))
        elif kind in ("visual_begin", "visual_end"):
            await ws.send(json.dumps({"type": "visual_ack", "sequence": msg.get("sequence"), "command": kind,
                                      "accepted": True, "reason": ""}))
        elif kind == "visual_capture":
            frame = await self.robot.capture()
            acquired = self._ms()
            header = {
                "type": "visual_image", "request_id": msg.get("request_id"), "format": "jpeg",
                "len": len(frame.jpeg), "width": 320, "height": 240, "frame_acquired_ms": acquired,
                "pose_sample_ms": (acquired + 5) % 2**32, "pose_skew_ms": 5, "pose_valid": True,
                "yaw_deg": 90 - self.robot.pan, "pitch_deg": 33, "calibration": CALIBRATION,
            }
            await ws.send(json.dumps(header))
            await ws.send(frame.jpeg)
        elif kind == "pose":
            await self.robot.look(int(msg.get("pan", 90)), int(msg.get("tilt", 82)))
            logger.info("[fake] pose pan=%s tilt=%s", self.robot.pan, self.robot.tilt)
        elif kind in ("expr", "motion"):
            logger.info("[fake] %s %s", kind, msg.get("name"))
        elif kind == "ir_send":
            logger.info("[fake] ir_send received (fake device: no infrared): %s", msg)
            await ws.send(json.dumps({"type": "ir_ack", "ok": True, "name": msg.get("name") or msg.get("protocol")}))

    async def run(self, url: str, seconds: float) -> None:
        """``seconds`` 秒だけ繋いで応え続ける（必ず止まる）。"""
        from websockets.asyncio.client import connect

        async with connect(url, max_size=8 * 1024 * 1024) as ws:
            await ws.send(json.dumps({"type": "hello", "device": "fake-stackchan", "fw_version": "fake"}))
            env_task = asyncio.create_task(self._env_loop(ws))
            try:
                async def reader() -> None:
                    async for raw in ws:
                        if isinstance(raw, (bytes, bytearray)):
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(msg, dict):
                            await self.handle(ws, msg)

                try:
                    await asyncio.wait_for(reader(), seconds)
                except asyncio.TimeoutError:
                    pass
            finally:
                env_task.cancel()


def main(argv: list[str] | None = None) -> int:
    """入り口。"""
    p = argparse.ArgumentParser(prog="mihari.fake_device", description="偽の本体（試験の道具）")
    p.add_argument("--server", required=True, help="中継サーバーの URL（例 ws://127.0.0.1:18080）")
    p.add_argument("--panorama", type=Path, required=True)
    p.add_argument("--env", type=Path, required=True)
    p.add_argument("--pan-right-sign", type=int, choices=(-1, 1), default=-1)
    p.add_argument("--seconds", type=float, default=120.0, help="この秒で止まる")
    p.add_argument("--log", type=Path, default=None, help="受けたメッセージの記録（JSONL）")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    robot = PanoramaRobot(load_image(args.panorama), pan_right_sign=args.pan_right_sign)
    device = FakeDevice(robot, load_env_json(args.env))
    from websockets.exceptions import ConnectionClosed

    try:
        asyncio.run(device.run(args.server, args.seconds))
    except ConnectionClosed as exc:
        logger.info("[fake] server closed the connection: %s", exc)
    finally:
        if args.log is not None:
            args.log.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in device.log) + "\n",
                                encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
