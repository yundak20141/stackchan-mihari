"""見張りの 1 周（シミュレーション・録画再生）のテスト。

AI の重みを使わず、マゼンタの「人」を描いた合成の部屋と ``MarkerDetector`` で流れ全体を回す。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from conftest import add_person, make_room, no_sleep
from mihari.config import Settings
from mihari.detection import MarkerDetector
from mihari.patrol import EventLog, Sentry
from mihari.robot import encode_jpeg
from mihari.robot_sim import PanoramaRobot, ReplayRobot

HOT = {"temp_c": 29.5, "humidity": 50.0, "unit": "sim"}
OK = {"temp_c": 24.0, "humidity": 50.0, "unit": "sim"}


def patrol(robot, tmp_path: Path | None = None, settings: Settings | None = None):
    sentry = Sentry(robot, MarkerDetector(), settings or Settings(),
                    log=EventLog(tmp_path) if tmp_path else None, sleep=no_sleep)
    return sentry, asyncio.run(sentry.patrol_once())


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("person_x", [450, 760, 1100])
def test_finds_person_and_turns_to_face_them(sign: int, person_x: int, tmp_path: Path) -> None:
    robot = PanoramaRobot(add_person(make_room(), person_x), env=HOT, pan_right_sign=sign)
    truth = robot.pan_of_x(person_x)
    _, report = patrol(robot, tmp_path)

    assert report.person_present
    assert report.calibration.source == "scan"
    assert report.target_pan == pytest.approx(truth, abs=2.0)
    # 振り向いた後の写真で、人が正面の近くにいる（閉じたループで確かめた）
    assert report.residual_deg is not None
    assert abs(report.residual_deg) <= Settings().center_tolerance_deg
    looks = [c.args for c in robot.calls if c.kind == "look"]
    assert [p for p, _ in looks[:5]] == [50, 70, 90, 110, 130]   # まず見回す
    assert looks[5][0] == report.target_pan                         # 次に人の方を向く
    assert report.notified == ["nod"]


@pytest.mark.parametrize(("sign", "person_x", "limit"), [(1, 250, 45), (-1, 250, 135), (1, 1300, 135)])
def test_person_beyond_reach_turns_to_the_limit(sign: int, person_x: int, limit: int) -> None:
    robot = PanoramaRobot(add_person(make_room(), person_x), env=HOT, pan_right_sign=sign)
    assert not 45 <= robot.pan_of_x(person_x) <= 135   # 首の可動範囲の外にいる
    _, report = patrol(robot)
    assert report.person_present
    assert report.target_pan == limit                    # 端まで向いて止まる（範囲の外へは指示しない）
    assert all(45 <= c.args[0] <= 135 for c in robot.calls if c.kind == "look")


def test_hot_room_with_person_decides_cool_but_does_not_send(tmp_path: Path) -> None:
    robot = PanoramaRobot(add_person(make_room(), 1150), env=HOT, pan_right_sign=-1)
    _, report = patrol(robot, tmp_path)
    assert report.ac_decision.action == "cool"
    assert report.ir_outcome.status == "dry_run"
    assert report.ir_outcome.payload == {"type": "ir_send", "kind": "raw", "name": "ac_cool"}
    assert robot.ir_log == []


def test_empty_room_no_person_no_ac(tmp_path: Path) -> None:
    robot = PanoramaRobot(make_room(), env=HOT, pan_right_sign=-1)
    _, report = patrol(robot, tmp_path)
    assert not report.person_present
    assert report.target_pan is None
    assert report.notified == []
    assert report.ac_decision.action == "none"
    assert report.ir_outcome.status == "none"
    # 誰もいなければ正面に戻る
    assert [c.args for c in robot.calls if c.kind == "look"][-1] == (90, 82)


def test_comfortable_room_with_person_no_ac(tmp_path: Path) -> None:
    robot = PanoramaRobot(add_person(make_room(), 760), env=OK, pan_right_sign=1)
    _, report = patrol(robot, tmp_path)
    assert report.person_present
    assert report.ac_decision.action == "none"


def test_event_log_and_picture_are_written(tmp_path: Path) -> None:
    robot = PanoramaRobot(add_person(make_room(), 760), env=HOT, pan_right_sign=1)
    patrol(robot, tmp_path)
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[-1])
    assert record["type"] == "patrol"
    assert record["person_present"] is True
    assert record["ac"]["action"] == "cool"
    assert record["ir"]["status"] == "dry_run"
    assert len(record["views"]) == 5
    assert (tmp_path / "patrol_0001.jpg").stat().st_size > 10_000
    assert (tmp_path / "latest.jpg").exists()


def test_second_patrol_does_not_repeat_the_same_command() -> None:
    robot = PanoramaRobot(add_person(make_room(), 760), env=HOT, pan_right_sign=1)
    sentry = Sentry(robot, MarkerDetector(), Settings(), sleep=no_sleep)
    first = asyncio.run(sentry.patrol_once())
    second = asyncio.run(sentry.patrol_once())
    assert first.ac_decision.action == "cool"
    assert second.ac_decision.action == "none"
    assert "送り直さない" in second.ac_decision.reason


def test_run_stops_after_cycles() -> None:
    robot = PanoramaRobot(make_room(), env=OK)
    sentry = Sentry(robot, MarkerDetector(), Settings(), sleep=no_sleep)
    reports = asyncio.run(sentry.run(3, 0.0))
    assert [r.index for r in reports] == [1, 2, 3]
    with pytest.raises(ValueError):
        asyncio.run(sentry.run(0, 0.0))


def test_replay_runs_the_same_flow(tmp_path: Path) -> None:
    # 録画のディレクトリを、シミュレーションで撮った写真から作る（実機の record と同じ形）
    room = add_person(make_room(), 1150)
    sim = PanoramaRobot(room, pan_right_sign=-1)
    frames = []

    async def shoot():
        for pan in (50, 70, 90, 110, 130):
            await sim.look(pan, 82)
            frame = await sim.capture()
            name = f"view_{pan:03d}.jpg"
            (tmp_path / name).write_bytes(frame.jpeg)
            frames.append({"pan": pan, "tilt": 82, "file": name})

    asyncio.run(shoot())
    (tmp_path / "session.json").write_text(json.dumps({"frames": frames, "env": HOT}), encoding="utf-8")

    robot = ReplayRobot(tmp_path)
    _, report = patrol(robot)
    assert report.person_present
    assert report.target_pan == pytest.approx(sim.pan_of_x(1150), abs=3.0)
    assert report.ac_decision.action == "cool"
    assert report.ir_outcome.status == "dry_run"


def test_encode_jpeg_roundtrip() -> None:
    import numpy as np

    from mihari.robot import Frame

    img = make_room(320, 240)
    frame = Frame(jpeg=encode_jpeg(img), pan=90, tilt=82)
    assert frame.image().shape == (240, 320, 3)
    assert np.abs(frame.image().astype(int) - img.astype(int)).mean() < 8


class PhantomDetector:
    """最初の 1 枚（見回しの pan 50）にだけ人を見る検出器（ハンガーの服を人と見た誤りの再現）。"""

    name = "phantom"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image):
        from mihari.detection import Detection

        self.calls += 1
        if self.calls == 1:
            return [Detection(box=(250.0, 120.0, 310.0, 240.0), score=0.65)]
        return []


def test_sighting_not_seen_after_turning_is_dropped(tmp_path: Path) -> None:
    robot = PanoramaRobot(make_room(), env=HOT, pan_right_sign=1)
    sentry = Sentry(robot, PhantomDetector(), Settings(), log=EventLog(tmp_path), sleep=no_sleep)
    report = asyncio.run(sentry.patrol_once())

    assert report.sightings                     # 見回しでは人らしきものを見た
    assert report.target_pan is not None        # 確かめるために振り向いた
    assert report.confirmed is False            # 振り向いた後の写真には人がいない
    assert not report.person_present
    assert report.notified == []                # うなずかない
    assert report.ac_decision.action == "none"  # 暑くても、人がいないので操作しない
    assert [c.args for c in robot.calls if c.kind == "look"][-1] == (90, 82)  # 正面に戻る
    record = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert record["confirmed"] is False
    assert record["person_present"] is False


def test_real_person_is_confirmed_after_turning() -> None:
    robot = PanoramaRobot(add_person(make_room(), 760), env=HOT, pan_right_sign=1)
    _, report = patrol(robot)
    assert report.confirmed is True
    assert report.person_present


def test_default_min_score_is_strict_enough_for_hanging_clothes() -> None:
    # 2026-10-29 の実機で、ハンガーの上着が person 0.65 と出た。試験用の部屋の写真の人は 0.98 以上だった
    assert Settings().min_score >= 0.8


class FlakyRobot(PanoramaRobot):
    """最初の撮影だけ時間切れになるロボット（揺らぎの再現）。"""

    def __init__(self, *args, fail_times: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_times = fail_times

    async def capture(self):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise TimeoutError("simulated capture timeout")
        return await super().capture()


def test_capture_timeout_is_retried_once() -> None:
    robot = FlakyRobot(add_person(make_room(), 760), env=HOT, pan_right_sign=1, fail_times=1)
    _, report = patrol(robot)
    assert report.person_present
    assert len(report.views) == 5


def test_capture_timeout_twice_stops_the_patrol() -> None:
    robot = FlakyRobot(make_room(), env=HOT, pan_right_sign=1, fail_times=2)
    with pytest.raises(TimeoutError):
        patrol(robot)
