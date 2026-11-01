"""見張り番の 1 周の流れ（実機・シミュレーション・録画再生で共通）。

1 周 = 見回す → 人を探す → 見つけたら振り向いて確かめて知らせる → 温湿度でエアコンを判断する。

1. ``scan_pans`` の角度へ順に首を向け、それぞれ静止画を 1 枚撮って人物検出にかける
2. 隣どうしの写真の景色のずれから画角と向きを測る（測れなければ設定値）
3. 写真ごとの検出を「その人の方向の pan」に直し、隣の写真の重なりを 1 人にまとめる
4. 一番確かな人の方へ首を向け、もう 1 枚撮って正面に来たかを確かめる（ずれていれば 1 回だけ合わせ直す）
5. 知らせる（うなずき・顔・声のうち設定したもの）
6. 最新の温湿度と「人がいるか」からエアコンへの判断を出し、赤外線の関所に渡す（既定は送らない）

結果は ``PatrolReport`` にまとめ、記録（JSONL）と絵（JPEG）に残す。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Awaitable, Callable

import cv2

from .config import PAN_CENTER, Settings
from .detection import Detection, Detector
from .geometry import PanCalibration, Sighting, calibrate_from_views, clamp_pan, merge_sightings
from .ir import IrGate, IrOutcome
from .robot import Frame, Robot
from .rules import AcDecision, AcState, EnvReading, decide_ac, next_state

logger = logging.getLogger(__name__)


@dataclass
class ViewResult:
    """見回しの 1 枚。"""

    pan: int
    tilt: int
    frame: Frame
    detections: list[Detection]
    infer_ms: float


@dataclass
class PatrolReport:
    """1 周の結果。"""

    index: int
    started_at: float
    views: list[ViewResult]
    calibration: PanCalibration
    sightings: list[Sighting]
    target_pan: int | None
    after_frame: Frame | None
    after_detections: list[Detection]
    residual_deg: float | None
    corrections: int
    notified: list[str]
    env: EnvReading | None
    ac_decision: AcDecision
    ir_outcome: IrOutcome
    detector: str
    robot: str
    elapsed_s: float = 0.0
    infer_ms_mean: float = 0.0
    #: 振り向いた後の写真にも人が写っていたか（見回しで何も見なかった周は None）。
    confirmed: bool | None = None

    @property
    def person_present(self) -> bool:
        """人を見つけたか（振り向いた後の写真で確かめられなかったものは数えない）。"""
        return bool(self.sightings) and self.confirmed is not False

    def to_record(self) -> dict:
        """記録（JSONL の 1 行）にする。写真の中身は入れない。"""
        return {
            "type": "patrol",
            "index": self.index,
            "robot": self.robot,
            "detector": self.detector,
            "elapsed_s": round(self.elapsed_s, 2),
            "infer_ms_mean": round(self.infer_ms_mean, 1),
            "views": [
                {"pan": v.pan, "tilt": v.tilt, "yaw_deg": v.frame.yaw_deg, "infer_ms": round(v.infer_ms, 1),
                 "detections": [{"box": [round(x, 1) for x in d.box], "score": round(d.score, 3)}
                                for d in v.detections]}
                for v in self.views
            ],
            "calibration": asdict(self.calibration),
            "sightings": [
                {"world_pan": round(s.world_pan, 1), "score": round(s.score, 3), "view_pan": s.view_pan,
                 "views": s.views}
                for s in self.sightings
            ],
            "person_present": self.person_present,
            "confirmed": self.confirmed,
            "target_pan": self.target_pan,
            "residual_deg": None if self.residual_deg is None else round(self.residual_deg, 2),
            "corrections": self.corrections,
            "notified": self.notified,
            "env": None if self.env is None else {
                "temp_c": self.env.temp_c, "humidity": self.env.humidity,
                "pressure_hpa": self.env.pressure_hpa, "unit": self.env.unit,
            },
            "comfort": self.ac_decision.comfort.value,
            "ac": {"action": self.ac_decision.action, "reason": self.ac_decision.reason,
                   "setpoint_c": self.ac_decision.setpoint_c},
            "ir": {"status": self.ir_outcome.status, "payload": self.ir_outcome.payload,
                   "ack": self.ir_outcome.ack, "note": self.ir_outcome.note},
        }


class EventLog:
    """記録（JSONL）と絵の書き出し先。``directory`` が None なら何も書かない。"""

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            self.path: Path | None = directory / "events.jsonl"
        else:
            self.path = None

    def write(self, record: dict) -> None:
        """1 行足す。"""
        if self.path is None:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_image(self, name: str, bgr) -> Path | None:
        """絵を JPEG で保存する（``latest.jpg`` にも上書きで写す）。"""
        if self.directory is None:
            return None
        path = self.directory / name
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        cv2.imwrite(str(self.directory / "latest.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return path


#: 見つけた時に声で言う文。
VOICE_TEXT = "お帰りなさいませ。お部屋の様子を見ておりました"


class Sentry:
    """見張り番。

    Args:
        robot: ロボット（実機・シミュレーション・録画再生）。
        detector: 人物検出器。
        settings: 設定。
        log: 記録の書き出し先。
        clock: 単調時計（秒）。
        on_report: 1 周ごとに結果を受ける関数（画面に出す時など）。
        sleep: 待つ関数（うなずきの間合いと周の間。テストで差し替える）。
    """

    def __init__(
        self,
        robot: Robot,
        detector: Detector,
        settings: Settings,
        *,
        log: EventLog | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_report: Callable[[PatrolReport], None] | None = None,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        self.robot = robot
        self.detector = detector
        self.settings = settings
        self.log = log or EventLog(None)
        self.clock = clock
        self.on_report = on_report
        self.sleep = sleep
        self.gate = IrGate(settings.ir, armed=settings.enable_ir)
        self.state = AcState(started_at=clock())
        self.fallback = PanCalibration(pan_sign=settings.pan_sign, hfov_deg=settings.hfov_deg, source="config")
        #: 最後に測れた校正（次の周で測れなかった時に使う）。
        self.calibration = self.fallback
        self.patrols = 0

    async def _shoot(self, pan: int, tilt: int) -> ViewResult:
        await self.robot.look(pan, tilt)
        try:
            frame = await self.robot.capture()
        except TimeoutError as exc:
            # 揺らぎ（本体の撮影の取りこぼし・通信の遅れ）は 1 回だけ撮り直す。2 回目も駄目なら止める。
            logger.warning("[patrol] capture at pan %s timed out (%s); retrying once", pan, exc)
            await self.robot.look(pan, tilt)
            frame = await self.robot.capture()
        start = time.perf_counter()
        detections = self.detector.detect(frame.image())
        infer_ms = (time.perf_counter() - start) * 1000
        return ViewResult(pan=frame.pan, tilt=frame.tilt, frame=frame, detections=detections, infer_ms=infer_ms)

    def _sightings(self, views: list[ViewResult], cal: PanCalibration) -> list[Sighting]:
        cands = []
        for view in views:
            width = view.frame.image().shape[1]
            for det in view.detections:
                cands.append(Sighting(
                    world_pan=cal.target_pan(view.pan, det.cx, width),
                    score=det.score, view_pan=view.pan, box=det.box,
                ))
        # 同じ人と見なす幅は画角の 1/5（隣の写真で同じ人が少しずれて写る分）。
        return merge_sightings(cands, same_person_deg=max(6.0, cal.hfov_deg / 5))

    async def _notify(self, pan: int) -> list[str]:
        done: list[str] = []
        tilt = self.settings.scan_tilt
        for how in self.settings.notify:
            if how == "nod":
                for _ in range(2):
                    await self.robot.look(pan, tilt + 12)
                    await self.sleep(0.35)
                    await self.robot.look(pan, tilt)
                    await self.sleep(0.35)
                done.append("nod")
            elif how == "face":
                if await self.robot.show_face(("surprise", "smile"), 1.2):
                    done.append("face")
            elif how == "voice":
                if await self.robot.say(VOICE_TEXT):
                    done.append("voice")
                else:
                    done.append("voice(silent)")
        return done

    async def patrol_once(self) -> PatrolReport:
        """1 周回して結果を返す（記録と絵も残す）。"""
        s = self.settings
        self.patrols += 1
        t0 = self.clock()
        wall0 = time.perf_counter()

        views = [await self._shoot(pan, s.scan_tilt) for pan in s.scan_pans]
        cal = calibrate_from_views([(v.pan, v.frame.image()) for v in views], self.calibration)
        if cal.source == "scan":
            self.calibration = cal
        sightings = self._sightings(views, cal)

        target_pan: int | None = None
        after: ViewResult | None = None
        residual: float | None = None
        corrections = 0
        notified: list[str] = []
        confirmed: bool | None = None
        if sightings:
            target_pan = clamp_pan(sightings[0].world_pan)
            after = await self._shoot(target_pan, s.scan_tilt)
            while True:
                if not after.detections:
                    break
                width = after.frame.image().shape[1]
                # 振り向いた後の写真の人を pan に直し、いま向いている pan との差を「ずれ」とする。
                # 実機では写真の pan ＝ 指示した pan。録画再生では一番近い写真の pan になるので、
                # 写真の pan を起点に計算する。
                worlds = [cal.target_pan(after.pan, d.cx, width) for d in after.detections]
                nearest_world = min(worlds, key=lambda w: abs(w - target_pan))
                residual = nearest_world - target_pan
                if abs(residual) <= s.center_tolerance_deg or corrections >= s.max_corrections:
                    break
                corrected = clamp_pan(nearest_world)
                if corrected == target_pan:
                    break  # 可動範囲の端で、これ以上は向けない
                target_pan = corrected
                corrections += 1
                after = await self._shoot(target_pan, s.scan_tilt)
            # 見回しの 1 枚だけの検出は、ハンガーの服などを人と見た誤りのことがある（2026-10-29 の実機）。
            # 振り向いた後の写真にも人が写っている時だけ「いる」と見なして知らせる。
            confirmed = bool(after.detections)
            if confirmed:
                notified = await self._notify(target_pan)
            else:
                await self.robot.look(PAN_CENTER, s.scan_tilt)
        else:
            await self.robot.look(PAN_CENTER, s.scan_tilt)
        present = confirmed is True

        now = self.clock()
        env = self.robot.latest_env()
        decision = decide_ac(person_present=present, env=env, now=now, state=self.state,
                             policy=s.ac, thresholds=s.thresholds)
        outcome = await self.gate.handle(decision, self.robot)
        self.state = next_state(self.state, decision=decision, person_present=present, now=now)

        infer = [v.infer_ms for v in views] + ([after.infer_ms] if after else [])
        report = PatrolReport(
            index=self.patrols, started_at=t0, views=views, calibration=cal, sightings=sightings,
            target_pan=target_pan, after_frame=after.frame if after else None,
            after_detections=after.detections if after else [], residual_deg=residual, corrections=corrections,
            notified=notified, env=env, ac_decision=decision, ir_outcome=outcome,
            detector=getattr(self.detector, "name", type(self.detector).__name__),
            robot=getattr(self.robot, "name", type(self.robot).__name__),
            elapsed_s=time.perf_counter() - wall0,
            infer_ms_mean=sum(infer) / len(infer) if infer else 0.0,
            confirmed=confirmed,
        )
        self.log.write(report.to_record())
        if self.log.directory is not None:
            from .annotate import render_report

            self.log.save_image(f"patrol_{self.patrols:04d}.jpg", render_report(report))
        if self.on_report is not None:
            self.on_report(report)
        return report

    async def run(self, cycles: int, interval_s: float) -> list[PatrolReport]:
        """``cycles`` 周回す（周の間は ``interval_s`` 秒休む）。``cycles`` は 1 以上で必ず止まる。"""
        if cycles < 1:
            raise ValueError("cycles must be >= 1")
        reports = []
        for i in range(cycles):
            reports.append(await self.patrol_once())
            if i + 1 < cycles:
                await self.sleep(interval_s)
        return reports


def describe(report: PatrolReport) -> str:
    """1 周の結果を日本語の 1〜3 行にする（端末に出す）。"""
    lines = []
    if report.person_present:
        best = report.sightings[0]
        lines.append(f"[{report.index}] 人を {len(report.sightings)} 人見つけた → pan {report.target_pan} へ振り向いた"
                     f"（確からしさ {best.score:.2f}"
                     + (f"・振り向いた後の中心からのずれ {report.residual_deg:+.1f} 度" if report.residual_deg is not None else "")
                     + f"）。知らせ: {', '.join(report.notified) or 'なし'}")
    elif report.sightings:
        best = report.sightings[0]
        lines.append(f"[{report.index}] 人らしきものを見た（確からしさ {best.score:.2f}）が、pan {report.target_pan} へ"
                     "振り向いた後の写真に人がいないので、人はいないと見なした")
    else:
        lines.append(f"[{report.index}] 人はいない（{len(report.views)} 方向を見た）")
    env = report.env
    env_text = f"{env.temp_c:.1f}℃ / {env.humidity:.0f}%" if env is not None and env.valid else "温湿度なし"
    lines.append(f"    部屋: {env_text} → {report.ac_decision.comfort.value}。エアコン: {report.ac_decision.action}"
                 f"（{report.ac_decision.reason}）")
    if report.ir_outcome.status != "none":
        lines.append(f"    赤外線: {report.ir_outcome.status} {report.ir_outcome.payload}（{report.ir_outcome.note}）")
    return "\n".join(lines)


def with_overrides(settings: Settings, **kwargs) -> Settings:
    """設定の一部を差し替えた写しを返す。"""
    return replace(settings, **kwargs)


__all__ = ["Sentry", "PatrolReport", "ViewResult", "EventLog", "describe", "with_overrides"]
