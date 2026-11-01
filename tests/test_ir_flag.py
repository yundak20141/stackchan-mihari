"""赤外線の実送信が既定で OFF であることのテスト。"""
from __future__ import annotations

import asyncio

import pytest

from mihari.cli import build_parser, resolve_settings
from mihari.config import IrSettings, Settings, settings_from_env
from mihari.ir import IrGate, build_ir_payload
from mihari.robot import IrForbiddenError, IrNotArmedError, check_ir_allowed
from mihari.robot_sim import PanoramaRobot
from mihari.rules import AcDecision

from conftest import make_room


def test_settings_default_is_ir_off() -> None:
    assert Settings().enable_ir is False


def test_env_cannot_turn_ir_on() -> None:
    s = settings_from_env({"MIHARI_ENABLE_IR": "1", "ENABLE_IR": "true", "MIHARI_IR_KIND": "raw"})
    assert s.enable_ir is False


@pytest.mark.parametrize("command", [["sim", "--panorama", "x.jpg"], ["replay", "--dir", "d"], ["run"]])
def test_cli_default_is_ir_off(command: list[str], tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # 手元の .env を読まない
    args = build_parser().parse_args(command)
    assert resolve_settings(args).enable_ir is False


def test_cli_flag_turns_ir_on(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["run", "--enable-ir"])
    assert resolve_settings(args).enable_ir is True


def test_last_gate_refuses_when_not_armed(monkeypatch) -> None:
    monkeypatch.delenv("MIHARI_FORBID_IR", raising=False)
    with pytest.raises(IrNotArmedError):
        check_ir_allowed(False)
    check_ir_allowed(True)  # armed なら通る（テストの外の実機だけ）


def test_tests_can_never_send_even_when_armed() -> None:
    # conftest が MIHARI_FORBID_IR=1 を立てている
    with pytest.raises(IrForbiddenError):
        check_ir_allowed(True)


def test_gate_dry_run_records_but_does_not_call_robot() -> None:
    robot = PanoramaRobot(make_room(320, 240))
    gate = IrGate(IrSettings(), armed=False)
    outcome = asyncio.run(gate.handle(AcDecision("cool", "人がいて暑い", 26), robot))
    assert outcome.status == "dry_run"
    assert outcome.payload == {"type": "ir_send", "kind": "raw", "name": "ac_cool"}
    assert robot.ir_log == []
    assert not [c for c in robot.calls if c.kind == "ir"]


def test_gate_does_nothing_for_none_decision() -> None:
    robot = PanoramaRobot(make_room(320, 240))
    outcome = asyncio.run(IrGate(IrSettings(), armed=False).handle(AcDecision("none", "快適"), robot))
    assert outcome.status == "none"
    assert outcome.payload is None


def test_armed_gate_in_tests_is_stopped_by_forbid_flag() -> None:
    robot = PanoramaRobot(make_room(320, 240))
    with pytest.raises(IrForbiddenError):
        asyncio.run(IrGate(IrSettings(), armed=True).handle(AcDecision("heat", "寒い", 22), robot))
    assert robot.ir_log == []


def test_ac_payloads() -> None:
    ir = IrSettings(kind="ac", protocol="DAIKIN")
    assert build_ir_payload(AcDecision("cool", "x", 26), ir) == {
        "type": "ir_send", "kind": "ac", "protocol": "DAIKIN", "power": True, "mode": "cool", "fan": "auto",
        "temp_c": 26,
    }
    assert build_ir_payload(AcDecision("off", "x"), ir) == {
        "type": "ir_send", "kind": "ac", "protocol": "DAIKIN", "power": False,
    }
    assert build_ir_payload(AcDecision("none", "x"), ir) is None
    with pytest.raises(ValueError):
        build_ir_payload(AcDecision("cool", "x", 26), IrSettings(kind="ac", protocol=None))
