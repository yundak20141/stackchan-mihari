"""コマンドの入り口: ``python -m mihari <サブコマンド>``。

- ``check-gpu``: GPU（CUDA）で人物検出のモデルが動くかを確かめる
- ``sim``: 横長の写真から首の向きごとの景色を切り出すシミュレーションで回す（実機なし）
- ``replay``: 実機で保存した写真と温湿度を流し込んで回す（実機なし）
- ``probe``: 実機につないで、温湿度と静止画 1 枚を受け取るだけ（首は正面へ）
- ``record``: 実機で見回して写真と温湿度を保存する（あとで replay に使う）
- ``run``: 実機で見張る
- ``learn-ir``: 本体にリモコンの信号を覚えさせる（受けるだけ・送らない。オーナーの立会いで）

赤外線を実際に送るのは ``--enable-ir`` を付けた時だけ。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import Settings, load_env_file, parse_notify, parse_pans, settings_from_env

logger = logging.getLogger("mihari")


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--env-file", type=Path, default=None, help="読む設定ファイル（既定は作業ディレクトリの .env）")
    p.add_argument("--out", type=Path, default=None, help="結果の置き場（既定は MIHARI_OUTPUT_DIR/<種類>-<時刻>）")
    p.add_argument("--log-level", default="INFO")


def _add_patrol(p: argparse.ArgumentParser) -> None:
    p.add_argument("--detector", choices=("torch", "marker"), default="torch",
                   help="torch = AI の人物検出（既定）/ marker = 試験用のマゼンタ検出")
    p.add_argument("--model", choices=("resnet50", "mobilenet"), default=None)
    p.add_argument("--device", choices=("cuda", "cpu"), default=None,
                   help="推論の装置（既定 cuda。cuda が使えなければ止まる。cpu は明示した時だけ）")
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--pans", default=None, help="見回す pan（例 50,70,90,110,130）")
    p.add_argument("--cycles", type=int, default=1, help="何周回すか（1 以上。必ず止まる）")
    p.add_argument("--interval", type=float, default=10.0, help="周の間に休む秒")
    p.add_argument("--notify", default=None, help="知らせ方（nod,face,voice の組み合わせ。既定 nod）")
    p.add_argument("--enable-ir", action="store_true",
                   help="赤外線を実際に送る（既定は送らずに『送るつもりだった』と記録するだけ）")
    p.add_argument("--show", action="store_true", help="結果の絵を窓に出す")


def build_parser() -> argparse.ArgumentParser:
    """引数の読み手を作る。"""
    parser = argparse.ArgumentParser(prog="mihari", description="スタックチャンの見張り番")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-gpu", help="GPU で人物検出のモデルが動くか確かめる")
    _add_common(p)
    p.add_argument("--model", choices=("resnet50", "mobilenet"), default=None)
    p.add_argument("--image", type=Path, default=None, help="試しに検出する画像")

    p = sub.add_parser("sim", help="シミュレーションで回す（実機なし）")
    _add_common(p)
    _add_patrol(p)
    p.add_argument("--panorama", type=Path, required=True, help="部屋の横長の写真")
    p.add_argument("--env", type=Path, default=None, help="温湿度の JSON（{\"temp_c\":..,\"humidity\":..}）")
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--hum", type=float, default=None)
    p.add_argument("--pan-right-sign", type=int, choices=(-1, 1), default=-1,
                   help="シミュレーションの取り付けの向き（pan を増やすと右なら 1）")

    p = sub.add_parser("replay", help="保存した写真と温湿度で回す（実機なし）")
    _add_common(p)
    _add_patrol(p)
    p.add_argument("--dir", type=Path, required=True, help="mihari record が作ったディレクトリ")
    p.add_argument("--env", type=Path, default=None, help="温湿度を差し替える JSON")

    for name, text in (("probe", "実機: 温湿度と静止画 1 枚を受け取るだけ"),
                       ("record", "実機: 見回して写真と温湿度を保存する"),
                       ("run", "実機: 見張る")):
        p = sub.add_parser(name, help=text)
        _add_common(p)
        p.add_argument("--server", default=None, help="母艦の中継サーバーの URL（既定は .env の MIHARI_SERVER_URL）")
        if name == "record":
            p.add_argument("--pans", default=None)
        if name == "run":
            _add_patrol(p)

    p = sub.add_parser("learn-ir", help="本体にリモコンの信号を覚えさせる（受けるだけ）")
    _add_common(p)
    p.add_argument("--server", default=None)
    p.add_argument("--name", required=True, help="覚えさせる名前（例 ac_cool）")
    p.add_argument("--timeout", type=int, default=10)
    return parser


def resolve_settings(args: argparse.Namespace) -> Settings:
    """環境変数と引数から設定を作る。``enable_ir`` は ``--enable-ir`` の時だけ True。"""
    load_env_file(args.env_file)
    s = settings_from_env()
    over: dict = {}
    if getattr(args, "server", None):
        over["server_url"] = args.server
    if getattr(args, "model", None):
        over["model"] = args.model
    if getattr(args, "device", None):
        over["device"] = args.device
    if getattr(args, "min_score", None) is not None:
        over["min_score"] = args.min_score
    if getattr(args, "pans", None):
        over["scan_pans"] = parse_pans(args.pans)
    if getattr(args, "notify", None):
        over["notify"] = parse_notify(args.notify)
    over["enable_ir"] = bool(getattr(args, "enable_ir", False))
    return replace(s, **over)


def _out_dir(args: argparse.Namespace, s: Settings, kind: str) -> Path:
    return args.out if args.out is not None else s.output_dir / f"{kind}-{_stamp()}"


def _detector(args: argparse.Namespace, s: Settings):
    from .detection import build_detector

    return build_detector(args.detector, model=s.model, device=s.device, min_score=s.min_score,
                          model_dir=s.model_dir)


def _viewer(enabled: bool):
    if not enabled:
        return None
    import cv2

    from .annotate import render_report

    def show(report) -> None:
        cv2.imshow("mihari", render_report(report))
        cv2.waitKey(1)

    return show


def _print_report(report) -> None:
    from .patrol import describe

    print(describe(report), flush=True)


async def _run_patrol(robot, args: argparse.Namespace, s: Settings, kind: str) -> int:
    from .patrol import EventLog, Sentry

    detector = _detector(args, s)
    out = _out_dir(args, s, kind)
    log = EventLog(out)
    log.write({"type": "start", "kind": kind, "robot": robot.name, "detector": detector.name,
               "enable_ir": s.enable_ir, "scan_pans": list(s.scan_pans), "notify": list(s.notify),
               "time": datetime.now().isoformat(timespec="seconds")})
    print(f"[mihari] {kind}: robot={robot.name} detector={detector.name} enable_ir={s.enable_ir} out={out}",
          flush=True)
    if not s.enable_ir:
        print("[mihari] 赤外線は送りません（--enable-ir 無し）。送る判断は『送るつもりだった』と記録します", flush=True)
    viewer = _viewer(args.show)

    def on_report(report) -> None:
        _print_report(report)
        if viewer is not None:
            viewer(report)

    sentry = Sentry(robot, detector, s, log=log, on_report=on_report)
    pump = asyncio.create_task(_pump_window()) if viewer is not None else None
    await robot.connect()
    try:
        await sentry.run(args.cycles, args.interval)
    finally:
        await robot.close()
        log.write({"type": "end", "patrols": sentry.patrols})
        if pump is not None:
            pump.cancel()
    print(f"[mihari] done: {sentry.patrols} patrol(s). events={out / 'events.jsonl'} image={out / 'latest.jpg'}",
          flush=True)
    if viewer is not None:
        import cv2

        print("[mihari] 窓で何かキーを押すと閉じます（10 秒で自動で閉じる）", flush=True)
        cv2.waitKey(10_000)
        cv2.destroyAllWindows()
    return 0


async def _pump_window() -> None:
    """結果の窓が固まらないよう、撮影や待ちの間も OpenCV の窓の処理を回し続ける。"""
    import cv2

    while True:
        cv2.waitKey(1)
        await asyncio.sleep(0.05)


def cmd_check_gpu(args: argparse.Namespace) -> int:
    """GPU の確認: torch の版・CUDA・装置名を出し、モデルを GPU に載せて推論の時間を測る。"""
    import numpy as np
    import torch

    from .detection import TorchvisionPersonDetector
    from .robot_sim import load_image

    s = resolve_settings(args)
    print(f"torch {torch.__version__} (CUDA build {torch.version.cuda}) cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
    det = TorchvisionPersonDetector(s.model, device="cuda", min_score=s.min_score, model_dir=s.model_dir)
    print(f"model loaded on {next(det.net.parameters()).device}: {det.name}")
    image = load_image(args.image) if args.image else np.zeros((240, 320, 3), np.uint8)
    times = []
    found = []
    for _ in range(5):
        found = det.detect(image)
        times.append(det.last_inference_ms)
    print(f"inference ms (5 runs): {', '.join(f'{t:.1f}' for t in times)}")
    print(f"persons: {[(round(d.score, 3), [round(v) for v in d.box]) for d in found]}")
    return 0


def cmd_sim(args: argparse.Namespace) -> int:
    """シミュレーションで回す。"""
    from .robot_sim import PanoramaRobot, load_env_json, load_image

    s = resolve_settings(args)
    env = load_env_json(args.env) if args.env else None
    if args.temp is not None or args.hum is not None:
        env = {"temp_c": args.temp, "humidity": args.hum, "unit": "sim"}
    robot = PanoramaRobot(load_image(args.panorama), env=env, pan_right_sign=args.pan_right_sign)
    return asyncio.run(_run_patrol(robot, args, s, "sim"))


def cmd_replay(args: argparse.Namespace) -> int:
    """録画再生で回す。"""
    from .robot_sim import ReplayRobot, load_env_json

    s = resolve_settings(args)
    robot = ReplayRobot(args.dir, env_override=load_env_json(args.env) if args.env else None)
    return asyncio.run(_run_patrol(robot, args, s, "replay"))


def _ws_robot(s: Settings):
    from .robot_ws import WsRobot

    return WsRobot(s.server_url, settle_s=s.settle_s)


def cmd_run(args: argparse.Namespace) -> int:
    """実機で見張る。"""
    from .robot_ws import DeviceOfflineError

    s = resolve_settings(args)
    try:
        return asyncio.run(_run_patrol(_ws_robot(s), args, s, "real"))
    except DeviceOfflineError as exc:
        print(f"[mihari] {exc}", file=sys.stderr)
        return 2
    except (OSError, TimeoutError) as exc:
        print(f"[mihari] サーバー {s.server_url} に繋がりません: {exc}", file=sys.stderr)
        return 2


async def _probe(s: Settings, out: Path) -> int:
    from .config import PAN_CENTER

    robot = _ws_robot(s)
    t0 = time.monotonic()
    await robot.connect()
    try:
        print(f"[probe] connected to {s.server_url}: device_connected={robot.device_connected}", flush=True)
        await robot.look(PAN_CENTER, s.scan_tilt)
        frame = await robot.capture()
        out.mkdir(parents=True, exist_ok=True)
        path = out / "probe_center.jpg"
        path.write_bytes(frame.jpeg)
        h, w = frame.image().shape[:2]
        print(f"[probe] frame {w}x{h} {len(frame.jpeg)} bytes yaw={frame.yaw_deg} pitch={frame.pitch_deg} -> {path}",
              flush=True)
        for _ in range(24):
            if robot.latest_env() is not None:
                break
            await asyncio.sleep(0.5)
        env = robot.latest_env()
        print(f"[probe] env: {None if env is None else (env.temp_c, env.humidity, env.pressure_hpa, env.unit)}",
              flush=True)
        (out / "probe.json").write_text(json.dumps({
            "frame": {"file": path.name, "width": w, "height": h, "bytes": len(frame.jpeg),
                      "yaw_deg": frame.yaw_deg, "pitch_deg": frame.pitch_deg},
            "env": None if env is None else {"temp_c": env.temp_c, "humidity": env.humidity,
                                             "pressure_hpa": env.pressure_hpa, "unit": env.unit},
            "elapsed_s": round(time.monotonic() - t0, 2),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        await robot.close()
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """実機で温湿度と静止画 1 枚を受け取る。"""
    from .robot_ws import DeviceOfflineError

    s = resolve_settings(args)
    try:
        return asyncio.run(_probe(s, _out_dir(args, s, "probe")))
    except DeviceOfflineError as exc:
        print(f"[probe] {exc}", file=sys.stderr)
        return 2
    except (OSError, TimeoutError) as exc:
        print(f"[probe] サーバー {s.server_url} に繋がりません: {exc}", file=sys.stderr)
        return 2


async def _record(s: Settings, out: Path) -> int:
    robot = _ws_robot(s)
    await robot.connect()
    frames = []
    try:
        out.mkdir(parents=True, exist_ok=True)
        for pan in s.scan_pans:
            await robot.look(pan, s.scan_tilt)
            frame = await robot.capture()
            name = f"view_{pan:03d}.jpg"
            (out / name).write_bytes(frame.jpeg)
            frames.append({"pan": pan, "tilt": s.scan_tilt, "file": name, "yaw_deg": frame.yaw_deg,
                           "pitch_deg": frame.pitch_deg})
            print(f"[record] pan {pan}: {len(frame.jpeg)} bytes yaw={frame.yaw_deg}", flush=True)
        from .config import PAN_CENTER

        await robot.look(PAN_CENTER, s.scan_tilt)
        env = robot.latest_env()
        session = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "frames": frames,
            "env": None if env is None else {"temp_c": env.temp_c, "humidity": env.humidity,
                                             "pressure_hpa": env.pressure_hpa, "unit": env.unit},
        }
        (out / "session.json").write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[record] saved {len(frames)} frames and env={session['env']} -> {out}", flush=True)
    finally:
        await robot.close()
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """実機で見回して保存する。"""
    from .robot_ws import DeviceOfflineError

    s = resolve_settings(args)
    try:
        return asyncio.run(_record(s, _out_dir(args, s, "recording")))
    except DeviceOfflineError as exc:
        print(f"[record] {exc}", file=sys.stderr)
        return 2
    except (OSError, TimeoutError) as exc:
        print(f"[record] サーバー {s.server_url} に繋がりません: {exc}", file=sys.stderr)
        return 2


async def _learn(s: Settings, name: str, timeout: int) -> int:
    from websockets.asyncio.client import connect

    async with connect(s.server_url, open_timeout=6) as ws:
        await ws.send(json.dumps({"type": "hello", "role": "phone", "device": "mihari-sentry"}))
        await ws.send(json.dumps({"type": "ir_learn", "name": name, "timeout_s": timeout}))
        print(f"[learn-ir] {timeout} 秒以内に、本体の正面でリモコンのボタンを 1 回押してください", flush=True)
        deadline = time.monotonic() + timeout + 5
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), deadline - time.monotonic())
            except asyncio.TimeoutError:
                break
            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "ir_learned":
                print(f"[learn-ir] {msg}", flush=True)
                return 0 if msg.get("ok") else 1
    print("[learn-ir] 返事がありませんでした", file=sys.stderr)
    return 1


def cmd_learn_ir(args: argparse.Namespace) -> int:
    """リモコンの信号を覚えさせる（赤外線を受けるだけ）。"""
    s = resolve_settings(args)
    return asyncio.run(_learn(s, args.name, args.timeout))


COMMANDS = {
    "check-gpu": cmd_check_gpu,
    "sim": cmd_sim,
    "replay": cmd_replay,
    "probe": cmd_probe,
    "record": cmd_record,
    "run": cmd_run,
    "learn-ir": cmd_learn_ir,
}


def main(argv: list[str] | None = None) -> int:
    """入り口。"""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if hasattr(args, "cycles") and args.cycles < 1:
        print("--cycles must be >= 1", file=sys.stderr)
        return 2
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
