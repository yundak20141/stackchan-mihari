"""判断の規則（快適度とエアコン）のテスト。"""
from __future__ import annotations

import math

import pytest

from mihari.config import AcPolicy, ClimateThresholds
from mihari.rules import AcDecision, AcState, Comfort, EnvReading, classify_comfort, decide_ac, next_state

TH = ClimateThresholds()
POLICY = AcPolicy()


def env(temp: float | None, hum: float | None, at: float = 100.0) -> EnvReading:
    return EnvReading(temp_c=temp, humidity=hum, unit="sim", received_at=at)


@pytest.mark.parametrize(
    ("temp", "hum", "expected"),
    [
        (16.9, 50, Comfort.COLD),
        (17.0, 50, Comfort.OK),        # 17 ちょうどは寒くない（未満が寒い）
        (27.9, 50, Comfort.OK),
        (28.0, 50, Comfort.HOT),       # 28 ちょうどから暑い
        (25.0, 70.0, Comfort.HUMID),   # 70 ちょうどから蒸し暑い
        (25.0, 69.9, Comfort.OK),
        (25.0, 34.9, Comfort.DRY),
        (25.0, 35.0, Comfort.OK),
        (16.0, 80.0, Comfort.COLD),    # 寒いが蒸し暑いより先（本体ファームと同じ順）
        (30.0, 80.0, Comfort.HOT),     # 暑いが蒸し暑いより先
    ],
)
def test_classify_comfort_boundaries(temp: float, hum: float, expected: Comfort) -> None:
    assert classify_comfort(env(temp, hum), TH) is expected


@pytest.mark.parametrize("reading", [None, env(None, 50), env(25, None), env(math.nan, 50), env(25, math.inf)])
def test_classify_unknown_when_unreadable(reading) -> None:
    assert classify_comfort(reading, TH) is Comfort.UNKNOWN


def decide(person: bool, reading: EnvReading | None, now: float = 100.0, state: AcState | None = None) -> AcDecision:
    return decide_ac(person_present=person, env=reading, now=now, state=state or AcState(), policy=POLICY,
                     thresholds=TH)


def test_person_and_hot_means_cool_at_setpoint() -> None:
    d = decide(True, env(29.0, 50))
    assert (d.action, d.setpoint_c, d.comfort) == ("cool", 26, Comfort.HOT)
    assert d.wants_ir


def test_person_and_cold_means_heat_at_setpoint() -> None:
    d = decide(True, env(15.0, 50))
    assert (d.action, d.setpoint_c) == ("heat", 22)


def test_person_and_humid_means_dry() -> None:
    d = decide(True, env(25.0, 75))
    assert (d.action, d.setpoint_c) == ("dry", None)


@pytest.mark.parametrize(("temp", "hum"), [(24.0, 50.0), (24.0, 30.0)])
def test_person_and_comfortable_means_nothing(temp: float, hum: float) -> None:
    d = decide(True, env(temp, hum))
    assert d.action == "none"
    assert not d.wants_ir


def test_nobody_hot_room_is_left_alone() -> None:
    d = decide(False, env(31.0, 50))
    assert d.action == "none"
    assert "人がいない" in d.reason


def test_no_reading_means_nothing_even_with_person() -> None:
    assert decide(True, None).action == "none"


def test_stale_reading_is_not_used() -> None:
    d = decide(True, env(31.0, 50, at=0.0), now=POLICY.env_max_age_s + 1)
    assert d.action == "none"
    assert "古い" in d.reason


def test_same_command_is_not_resent_within_min_resend() -> None:
    state = AcState(last_action="cool", last_sent_at=100.0)
    assert decide(True, env(29.0, 50, at=1000.0), now=1000.0, state=state).action == "none"
    later = 100.0 + POLICY.min_resend_s + 1
    assert decide(True, env(29.0, 50, at=later), now=later, state=state).action == "cool"


def test_different_command_is_sent_even_soon_after() -> None:
    state = AcState(last_action="cool", last_sent_at=100.0)
    assert decide(True, env(15.0, 50, at=200.0), now=200.0, state=state).action == "heat"


def test_turns_off_own_ac_after_long_absence() -> None:
    state = AcState(last_action="cool", last_sent_at=0.0, person_last_seen_at=0.0)
    soon = POLICY.off_after_absent_s - 1
    assert decide(False, env(29.0, 50, at=soon), now=soon, state=state).action == "none"
    late = POLICY.off_after_absent_s
    d = decide(False, env(29.0, 50, at=late), now=late, state=state)
    assert d.action == "off"
    assert d.wants_ir


def test_never_turns_off_an_ac_it_did_not_start() -> None:
    state = AcState(last_action=None, person_last_seen_at=0.0)
    now = POLICY.off_after_absent_s * 10
    assert decide(False, env(29.0, 50, at=now), now=now, state=state).action == "none"


def test_next_state_remembers_action_and_sighting() -> None:
    s1 = next_state(AcState(), decision=AcDecision("cool", "x", 26), person_present=True, now=10.0)
    assert (s1.last_action, s1.last_sent_at, s1.person_last_seen_at) == ("cool", 10.0, 10.0)
    s2 = next_state(s1, decision=AcDecision("none", "y"), person_present=False, now=20.0)
    assert (s2.last_action, s2.last_sent_at, s2.person_last_seen_at) == ("cool", 10.0, 10.0)
    s3 = next_state(s2, decision=AcDecision("off", "z"), person_present=False, now=30.0)
    assert (s3.last_action, s3.last_sent_at) == (None, 30.0)
