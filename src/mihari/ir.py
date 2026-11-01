"""赤外線（エアコン）の指令の組み立てと、実送信の関所。

既定は「送るつもりだった」と記録するだけで、本体へは何も送らない。実際に送るのは
CLI で ``--enable-ir`` を付けた時だけ（``Settings.enable_ir`` は環境変数からは立たない）。
本体側の指令の形はファームの既存の口（``ir_send`` の ``kind:"raw"`` / ``kind:"ac"``）に合わせる。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import IrSettings
from .robot import Robot
from .rules import AcDecision

logger = logging.getLogger(__name__)

#: kind=ac の時の運転モードの名前（本体ファームの stdAc の写しが受ける名前）。
AC_MODE_NAMES = {"cool": "cool", "heat": "heat", "dry": "dry"}


def build_ir_payload(decision: AcDecision, ir: IrSettings) -> dict | None:
    """判断から本体へ送る ``ir_send`` の中身を作る。何もしない判断なら None。

    Args:
        decision: エアコンへの判断。
        ir: 指令の作り方。

    Returns:
        ``{"type":"ir_send","kind":"raw","name":...}`` か
        ``{"type":"ir_send","kind":"ac","protocol":...,"power":...,"mode":...,"temp_c":...,"fan":"auto"}``。

    Raises:
        ValueError: kind=ac なのにメーカー名が無い時・覚えさせた信号の名前が無い時。
    """
    if not decision.wants_ir:
        return None
    if ir.kind == "raw":
        name = ir.raw_names.get(decision.action)
        if not name:
            raise ValueError(f"no learned signal name for '{decision.action}' (MIHARI_IR_RAW_{decision.action.upper()})")
        return {"type": "ir_send", "kind": "raw", "name": name}
    if not ir.protocol:
        raise ValueError("MIHARI_IR_PROTOCOL is required when MIHARI_IR_KIND=ac")
    if decision.action == "off":
        return {"type": "ir_send", "kind": "ac", "protocol": ir.protocol, "power": False}
    payload: dict = {
        "type": "ir_send", "kind": "ac", "protocol": ir.protocol, "power": True,
        "mode": AC_MODE_NAMES[decision.action], "fan": "auto",
    }
    if decision.setpoint_c is not None:
        payload["temp_c"] = int(decision.setpoint_c)
    return payload


@dataclass(frozen=True)
class IrOutcome:
    """赤外線の関所を通った結果。

    Attributes:
        status: ``"none"``（送る判断ではない）／``"dry_run"``（送るつもりだった・--enable-ir 無し）／
            ``"sent"``（送って本体が受けた）／``"failed"``（送ったが本体が断った・返事が無い・組み立てられない）。
        payload: 送った（送るつもりだった）中身。
        ack: 本体の返事（送った時だけ）。
        note: 説明（日本語）。
    """

    status: str
    payload: dict | None = None
    ack: dict | None = None
    note: str = ""


class IrGate:
    """赤外線の実送信の関所。

    Args:
        ir: 指令の作り方。
        armed: 実際に送るか（CLI の ``--enable-ir``）。既定 False。
    """

    def __init__(self, ir: IrSettings, *, armed: bool = False) -> None:
        self.ir = ir
        self.armed = armed

    async def handle(self, decision: AcDecision, robot: Robot) -> IrOutcome:
        """判断を受けて、送る・送るつもりだったと記録する・何もしない、のどれかをする。

        armed で ``ac_pan`` があれば、送る前に首をエアコンの方へ向ける（赤外線は見通しが要る）。
        """
        if not decision.wants_ir:
            return IrOutcome("none", note=decision.reason)
        try:
            payload = build_ir_payload(decision, self.ir)
        except ValueError as exc:
            return IrOutcome("failed", note=f"指令を組み立てられない: {exc}")
        if not self.armed:
            logger.info("[ir] would send (no --enable-ir): %s", payload)
            return IrOutcome("dry_run", payload=payload, note="--enable-ir が無いので送らずに記録だけした")
        if self.ir.ac_pan is not None:
            await robot.look(self.ir.ac_pan, getattr(robot, "tilt", 82))
        ack = await robot.send_ir(payload, armed=True)
        ok = bool(ack.get("ok"))
        return IrOutcome("sent" if ok else "failed", payload=payload, ack=ack,
                         note="本体が受けた" if ok else f"本体が断った: {ack.get('reason')}")
