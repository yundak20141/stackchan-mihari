"""判断の規則（純粋な関数だけ。ロボットにも AI にも触らない）。

- 温湿度から快適度 5 段（寒い・暑い・蒸し暑い・乾燥・快適）を決める（本体ファームと同じ順と境目）
- 「人がいるか」と快適度から、エアコンに何をさせるか（冷房・暖房・除湿・停止・何もしない）を決める

エアコンの判断は「人がいる時だけ動かす」「同じ指令を短い間に送り直さない」「自分が入れた
エアコンは、人がいない状態がしばらく続いたら切る」の 3 つを守る。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

from .config import AcPolicy, ClimateThresholds


class Comfort(str, Enum):
    """快適度 5 段と「読めない」。"""

    COLD = "cold"
    HOT = "hot"
    HUMID = "humid"
    DRY = "dry"
    OK = "ok"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EnvReading:
    """本体の温湿度ユニットの 1 回分の値。

    Attributes:
        temp_c: 温度（℃）。読めなければ None。
        humidity: 湿度（%）。読めなければ None。
        pressure_hpa: 気圧（hPa）。無ければ None。
        unit: ユニットの種類（``env3`` / ``env4`` / ``none`` / ``sim`` など）。
        received_at: 受け取った時刻（単調時計の秒）。
    """

    temp_c: float | None
    humidity: float | None
    pressure_hpa: float | None = None
    unit: str = "unknown"
    received_at: float = 0.0

    @property
    def valid(self) -> bool:
        """温度と湿度の両方が有限の数か。"""
        return (
            isinstance(self.temp_c, (int, float)) and math.isfinite(self.temp_c)
            and isinstance(self.humidity, (int, float)) and math.isfinite(self.humidity)
        )


def classify_comfort(env: EnvReading | None, th: ClimateThresholds) -> Comfort:
    """快適度を決める。順は 寒い → 暑い → 蒸し暑い → 乾燥 → 快適（本体ファームの classify と同じ）。

    Args:
        env: 温湿度。None か値が無ければ UNKNOWN。
        th: 境目。

    Returns:
        快適度。
    """
    if env is None or not env.valid:
        return Comfort.UNKNOWN
    assert env.temp_c is not None and env.humidity is not None
    if env.temp_c < th.cold_below:
        return Comfort.COLD
    if env.temp_c >= th.hot_from:
        return Comfort.HOT
    if env.humidity >= th.humid_from:
        return Comfort.HUMID
    if env.humidity < th.dry_below:
        return Comfort.DRY
    return Comfort.OK


#: エアコンの操作の名前。
AC_ACTIONS = ("none", "cool", "heat", "dry", "off")


@dataclass(frozen=True)
class AcDecision:
    """エアコンへの判断。

    Attributes:
        action: ``none`` / ``cool`` / ``heat`` / ``dry`` / ``off``。
        reason: 判断の理由（日本語・記録と画面に出す）。
        setpoint_c: 冷房・暖房の設定温度（それ以外は None）。
        comfort: 判断に使った快適度。
    """

    action: str
    reason: str
    setpoint_c: int | None = None
    comfort: Comfort = Comfort.UNKNOWN

    @property
    def wants_ir(self) -> bool:
        """赤外線を送る判断か。"""
        return self.action != "none"


@dataclass(frozen=True)
class AcState:
    """エアコンについて覚えておくこと（判断をまたいで持ち越す）。

    Attributes:
        last_action: 最後に送った（送るつもりだった）操作。まだ無ければ None。
        last_sent_at: その時刻（単調時計の秒）。
        person_last_seen_at: 最後に人を見た時刻。まだ見ていなければ None。
        started_at: 見張りを始めた時刻（人を一度も見ていない時の「いない時間」の起点）。
    """

    last_action: str | None = None
    last_sent_at: float | None = None
    person_last_seen_at: float | None = None
    started_at: float = 0.0


def decide_ac(
    *,
    person_present: bool,
    env: EnvReading | None,
    now: float,
    state: AcState,
    policy: AcPolicy,
    thresholds: ClimateThresholds,
) -> AcDecision:
    """人の有無と温湿度から、エアコンに何をさせるかを決める。

    規則（上から順に見る）:
      1. 温湿度が読めない・古い（``env_max_age_s`` 超）→ 何もしない
      2. 人がいる:
         暑い → 冷房 / 寒い → 暖房 / 蒸し暑い → 除湿 / 快適・乾燥 → 何もしない。
         ただし同じ操作を ``min_resend_s`` 以内に送っていれば送り直さない
      3. 人がいない:
         自分が冷房・暖房・除湿を入れていて、いない状態が ``off_after_absent_s`` 以上続いたら停止。
         それ以外は何もしない（人のいない部屋のために電気を使わない）

    Args:
        person_present: いまの見回しで人を見つけたか。
        env: 最新の温湿度。
        now: いまの時刻（単調時計の秒）。
        state: 持ち越している状態。
        policy: 方針。
        thresholds: 暑い・寒いの境目。

    Returns:
        判断。
    """
    comfort = classify_comfort(env, thresholds)
    if comfort is Comfort.UNKNOWN:
        return AcDecision("none", "温湿度が読めないので判断しない", comfort=comfort)
    assert env is not None
    age = now - env.received_at
    if age > policy.env_max_age_s:
        return AcDecision("none", f"温湿度が {age:.0f} 秒前の値で古いので判断しない", comfort=comfort)

    if person_present:
        if comfort is Comfort.HOT:
            action, setpoint, why = "cool", policy.cool_setpoint_c, "人がいて暑い"
        elif comfort is Comfort.COLD:
            action, setpoint, why = "heat", policy.heat_setpoint_c, "人がいて寒い"
        elif comfort is Comfort.HUMID:
            action, setpoint, why = "dry", None, "人がいて蒸し暑い"
        else:
            return AcDecision("none", "人がいるが快適なので操作しない", comfort=comfort)
        if (
            state.last_action == action
            and state.last_sent_at is not None
            and now - state.last_sent_at < policy.min_resend_s
        ):
            ago = now - state.last_sent_at
            ago_text = f"{ago:.0f} 秒前" if ago < 60 else f"{ago / 60:.0f} 分前"
            return AcDecision(
                "none", f"{why}が、同じ指令を {ago_text}に出したので送り直さない", comfort=comfort,
            )
        return AcDecision(action, why, setpoint_c=setpoint, comfort=comfort)

    if state.last_action in ("cool", "heat", "dry"):
        since = state.person_last_seen_at if state.person_last_seen_at is not None else state.started_at
        absent = now - since
        if absent >= policy.off_after_absent_s:
            return AcDecision("off", f"人がいない状態が {absent / 60:.0f} 分続いたので止める", comfort=comfort)
        return AcDecision("none", f"人がいない（{absent / 60:.0f} 分）。もう少し待ってから止める", comfort=comfort)
    return AcDecision("none", "人がいないので操作しない", comfort=comfort)


def next_state(
    state: AcState,
    *,
    decision: AcDecision,
    person_present: bool,
    now: float,
) -> AcState:
    """判断の後の状態を返す（送った・送るつもりだった操作と、人を見た時刻を覚える）。

    赤外線を実際に送らない設定でも「送るつもりだった」操作として覚える。そうしないと、
    試運転の記録で毎回同じ「冷房」が並び、本番の動き（送り直さない）と食い違うため。
    """
    updated = state
    if person_present:
        updated = replace(updated, person_last_seen_at=now)
    if decision.wants_ir:
        updated = replace(updated, last_action=decision.action, last_sent_at=now)
        if decision.action == "off":
            updated = replace(updated, last_action=None)
    return updated
