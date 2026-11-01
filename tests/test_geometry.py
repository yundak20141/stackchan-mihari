"""首の角度の計算のテスト。"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from conftest import make_room
from mihari.geometry import (
    PanCalibration,
    Sighting,
    calibrate_from_views,
    clamp_pan,
    clamp_tilt,
    estimate_shift,
    merge_sightings,
)
from mihari.robot_sim import PanoramaRobot

FALLBACK = PanCalibration(pan_sign=-1, hfov_deg=60.0, source="config")


@pytest.mark.parametrize(("pan", "expected"), [(10, 45), (45, 45), (90, 90), (134.6, 135), (200, 135), (89.5, 90)])
def test_clamp_pan(pan: float, expected: int) -> None:
    assert clamp_pan(pan) == expected


@pytest.mark.parametrize(("tilt", "expected"), [(0, 60), (82, 82), (130, 120)])
def test_clamp_tilt(tilt: float, expected: int) -> None:
    assert clamp_tilt(tilt) == expected


def test_offset_is_zero_at_center_and_signed() -> None:
    cal = PanCalibration(pan_sign=1, hfov_deg=60.0, source="config")
    assert cal.offset_deg(160, 320) == 0.0
    assert cal.offset_deg(320, 320) == pytest.approx(30.0)   # 右端は半分の画角
    assert cal.offset_deg(0, 320) == pytest.approx(-30.0)


@pytest.mark.parametrize("sign", [1, -1])
def test_target_pan_follows_sign(sign: int) -> None:
    cal = PanCalibration(pan_sign=sign, hfov_deg=60.0, source="config")
    # 右端の 3/4 の所（中心から +15 度）にいる人。
    assert cal.target_pan(90, 240, 320) == pytest.approx(90 + sign * 15.0)


def test_estimate_shift_finds_known_translation() -> None:
    room = make_room(640, 480)
    a = room[100:340, 100:420]
    b = room[100 + 5:340 + 5, 100 - 37:420 - 37]   # b は a から景色が右へ 37・上へ 5 ずれた
    dx, dy, strength = estimate_shift(a, b)
    assert dx == pytest.approx(37, abs=1.0)
    assert dy == pytest.approx(-5, abs=1.0)
    assert strength > 10


def test_estimate_shift_rejects_mismatched_sizes() -> None:
    with pytest.raises(ValueError):
        estimate_shift(np.zeros((10, 10)), np.zeros((10, 12)))


def _views(robot: PanoramaRobot, pans: tuple[int, ...]) -> list[tuple[float, np.ndarray]]:
    async def go():
        out = []
        for pan in pans:
            await robot.look(pan, 82)
            out.append((pan, (await robot.capture()).image()))
        return out

    return asyncio.run(go())


@pytest.mark.parametrize("sign", [1, -1])
def test_calibration_from_scan_measures_fov_and_direction(sign: int) -> None:
    robot = PanoramaRobot(make_room(), pan_right_sign=sign)
    cal = calibrate_from_views(_views(robot, (50, 70, 90, 110, 130)), FALLBACK)
    assert cal.source == "scan"
    assert cal.pan_sign == sign
    assert cal.hfov_deg == pytest.approx(robot.hfov_deg, rel=0.05)
    assert cal.pairs_used >= 3


def test_calibration_falls_back_on_featureless_views() -> None:
    flat = np.full((1024, 1536, 3), 128, np.uint8)
    robot = PanoramaRobot(flat, pan_right_sign=1)
    assert calibrate_from_views(_views(robot, (50, 70, 90, 110, 130)), FALLBACK) == FALLBACK


def test_calibration_falls_back_with_single_view() -> None:
    robot = PanoramaRobot(make_room(), pan_right_sign=1)
    assert calibrate_from_views(_views(robot, (90,)), FALLBACK) == FALLBACK


def test_merge_sightings_joins_same_person_in_adjacent_views() -> None:
    a = Sighting(world_pan=112.0, score=0.95, view_pan=110, box=(0, 0, 1, 1))
    b = Sighting(world_pan=114.5, score=0.80, view_pan=130, box=(0, 0, 1, 1))
    c = Sighting(world_pan=60.0, score=0.70, view_pan=50, box=(0, 0, 1, 1))
    merged = merge_sightings([b, c, a], same_person_deg=8.0)
    assert [m.world_pan for m in merged] == [112.0, 60.0]   # 確かな順・代表は高い方
    assert merged[0].views == 2
    assert merged[1].views == 1
