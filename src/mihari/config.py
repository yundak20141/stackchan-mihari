"""設定の読み込み。

値は環境変数（``.env`` があれば先に読み込む）から取り、CLI の引数で上書きする。
赤外線の実送信（``enable_ir``）だけは環境変数からは立てられず、CLI の ``--enable-ir`` でしか
True にならない（うっかり .env に書いて送ってしまう事故を防ぐため）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

#: サーボの可動範囲（本体ファームの config.h と同じ値）。
PAN_MIN = 45
PAN_MAX = 135
PAN_CENTER = 90
TILT_MIN = 60
TILT_MAX = 120
TILT_CENTER = 82


@dataclass(frozen=True)
class ClimateThresholds:
    """暑い・寒いなどの境目（本体ファームの快適度 5 段と同じ既定値）。

    Attributes:
        cold_below: この温度（℃）未満は寒い。
        hot_from: この温度（℃）以上は暑い。
        humid_from: この湿度（%）以上は蒸し暑い。
        dry_below: この湿度（%）未満は乾燥。
    """

    cold_below: float = 17.0
    hot_from: float = 28.0
    humid_from: float = 70.0
    dry_below: float = 35.0


@dataclass(frozen=True)
class AcPolicy:
    """エアコン操作の方針。

    Attributes:
        cool_setpoint_c: 冷房の設定温度。
        heat_setpoint_c: 暖房の設定温度。
        min_resend_s: 同じ指令を送り直さない秒（連打で赤外線を潰さない・リモコンの代わりに暴れない）。
        off_after_absent_s: 人がいない状態がこの秒続いたら、自分が入れたエアコンを切る。
        env_max_age_s: 温湿度の値がこれより古ければ判断に使わない。
    """

    cool_setpoint_c: int = 26
    heat_setpoint_c: int = 22
    min_resend_s: float = 1800.0
    off_after_absent_s: float = 1800.0
    env_max_age_s: float = 120.0


@dataclass(frozen=True)
class IrSettings:
    """赤外線の指令の作り方。

    Attributes:
        kind: ``"raw"``（本体に覚えさせた信号を名前で送る）か ``"ac"``（メーカーの方式で組み立てる）。
        raw_names: 操作ごとの覚えさせた信号の名前。
        protocol: kind=ac の時のメーカー名（IRremoteESP8266 の名前・例 ``"DAIKIN"``）。
        ac_pan: 送る前に首を向ける pan（エアコンの方向）。None なら向きを変えない。
    """

    kind: str = "raw"
    raw_names: Mapping[str, str] = field(default_factory=lambda: {
        "cool": "ac_cool", "heat": "ac_heat", "dry": "ac_dry", "off": "ac_off",
    })
    protocol: str | None = None
    ac_pan: int | None = None


@dataclass(frozen=True)
class Settings:
    """見張り番の設定一式。

    Attributes:
        server_url: 母艦の中継サーバーの WebSocket URL。
        scan_pans: 見回す pan の並び。
        scan_tilt: 見回す時の tilt。
        settle_s: 首を動かしてから写真を採るまで待つ秒。
        model: 人物検出のモデル名（``resnet50`` / ``mobilenet``）。
        min_score: 人と見なす確からしさの下限。
        device: 推論の装置（``cuda`` / ``cpu``）。cuda が使えない時は止まる。
        model_dir: モデルの重みの置き場。
        hfov_deg: 画角の初期値（見回しの写真から自動で測れない時に使う）。
        pan_sign: pan を増やすと右を向くなら +1、左なら -1（同上）。
        center_tolerance_deg: 振り向いた後、人が正面からこの角度以内なら合わせ直さない。
        max_corrections: 振り向いた後に合わせ直す回数の上限。
        notify: 見つけた時の知らせ方（``nod`` / ``face`` / ``voice`` の組み合わせ）。
        thresholds: 暑い・寒いの境目。
        ac: エアコン操作の方針。
        ir: 赤外線の指令の作り方。
        enable_ir: 赤外線を実際に送るか。既定 False。CLI の ``--enable-ir`` でだけ True になる。
        output_dir: 結果（写真と記録）の置き場。
    """

    server_url: str = "ws://127.0.0.1:8080"
    scan_pans: tuple[int, ...] = (50, 70, 90, 110, 130)
    scan_tilt: int = TILT_CENTER
    settle_s: float = 1.2
    model: str = "resnet50"
    min_score: float = 0.8
    device: str = "cuda"
    model_dir: Path = Path(".models")
    hfov_deg: float = 60.0
    pan_sign: int = -1
    center_tolerance_deg: float = 4.0
    max_corrections: int = 1
    notify: tuple[str, ...] = ("nod",)
    thresholds: ClimateThresholds = field(default_factory=ClimateThresholds)
    ac: AcPolicy = field(default_factory=AcPolicy)
    ir: IrSettings = field(default_factory=IrSettings)
    enable_ir: bool = False
    output_dir: Path = Path("runs")


NOTIFY_CHOICES = ("nod", "face", "voice")


def _get(env: Mapping[str, str], key: str) -> str | None:
    """空文字を None として環境変数を読む。"""
    value = env.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    value = _get(env, key)
    return default if value is None else float(value)


def _int(env: Mapping[str, str], key: str, default: int | None) -> int | None:
    value = _get(env, key)
    return default if value is None else int(value)


def parse_pans(text: str) -> tuple[int, ...]:
    """``"50,70,90"`` を pan の並びにする（範囲外は可動範囲に収める・重複は除く）。

    Args:
        text: カンマ区切りの pan。

    Returns:
        pan の並び（書いた順）。

    Raises:
        ValueError: 1 つも読めない時。
    """
    pans: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        pan = max(PAN_MIN, min(PAN_MAX, int(part)))
        if pan not in pans:
            pans.append(pan)
    if not pans:
        raise ValueError("scan pans are empty")
    return tuple(pans)


def parse_notify(text: str) -> tuple[str, ...]:
    """``"nod,face"`` を知らせ方の並びにする。

    Raises:
        ValueError: 知らない知らせ方がある時。
    """
    items = tuple(p.strip() for p in text.split(",") if p.strip())
    for item in items:
        if item not in NOTIFY_CHOICES:
            raise ValueError(f"unknown notify '{item}' (choose from {', '.join(NOTIFY_CHOICES)})")
    return items


def load_env_file(path: Path | None) -> None:
    """``.env`` を環境変数へ読み込む（既にある環境変数は上書きしない）。

    Args:
        path: 読むファイル。None なら作業ディレクトリの ``.env`` があれば読む。
    """
    from dotenv import load_dotenv

    target = path if path is not None else Path(".env")
    if target.exists():
        load_dotenv(target, override=False)


def settings_from_env(env: Mapping[str, str] | None = None) -> Settings:
    """環境変数から設定を作る。``MIHARI_ENABLE_IR`` のような変数は読まない（enable_ir は常に False）。

    Args:
        env: 読む環境変数（None なら ``os.environ``）。

    Returns:
        設定。
    """
    env = os.environ if env is None else env
    base = Settings()
    thresholds = ClimateThresholds(
        cold_below=_float(env, "MIHARI_COLD_BELOW_C", base.thresholds.cold_below),
        hot_from=_float(env, "MIHARI_HOT_FROM_C", base.thresholds.hot_from),
        humid_from=_float(env, "MIHARI_HUMID_FROM", base.thresholds.humid_from),
        dry_below=_float(env, "MIHARI_DRY_BELOW", base.thresholds.dry_below),
    )
    ac = AcPolicy(
        cool_setpoint_c=_int(env, "MIHARI_COOL_SETPOINT_C", base.ac.cool_setpoint_c) or base.ac.cool_setpoint_c,
        heat_setpoint_c=_int(env, "MIHARI_HEAT_SETPOINT_C", base.ac.heat_setpoint_c) or base.ac.heat_setpoint_c,
        min_resend_s=_float(env, "MIHARI_AC_MIN_RESEND_S", base.ac.min_resend_s),
        off_after_absent_s=_float(env, "MIHARI_AC_OFF_AFTER_ABSENT_S", base.ac.off_after_absent_s),
        env_max_age_s=_float(env, "MIHARI_ENV_MAX_AGE_S", base.ac.env_max_age_s),
    )
    raw_names = dict(IrSettings().raw_names)
    for action in ("cool", "heat", "dry", "off"):
        name = _get(env, f"MIHARI_IR_RAW_{action.upper()}")
        if name is not None:
            raw_names[action] = name
    ac_pan = _int(env, "MIHARI_AC_PAN", None)
    ir = IrSettings(
        kind=_get(env, "MIHARI_IR_KIND") or "raw",
        raw_names=raw_names,
        protocol=_get(env, "MIHARI_IR_PROTOCOL"),
        ac_pan=None if ac_pan is None else max(PAN_MIN, min(PAN_MAX, ac_pan)),
    )
    if ir.kind not in ("raw", "ac"):
        raise ValueError("MIHARI_IR_KIND must be raw or ac")
    pan_sign = _int(env, "MIHARI_PAN_SIGN", base.pan_sign)
    if pan_sign not in (-1, 1):
        raise ValueError("MIHARI_PAN_SIGN must be -1 or 1")
    pans = _get(env, "MIHARI_SCAN_PANS")
    notify = _get(env, "MIHARI_NOTIFY")
    return replace(
        base,
        server_url=_get(env, "MIHARI_SERVER_URL") or base.server_url,
        scan_pans=parse_pans(pans) if pans else base.scan_pans,
        scan_tilt=max(TILT_MIN, min(TILT_MAX, _int(env, "MIHARI_SCAN_TILT", base.scan_tilt) or base.scan_tilt)),
        settle_s=_float(env, "MIHARI_SETTLE_S", base.settle_s),
        model=_get(env, "MIHARI_MODEL") or base.model,
        min_score=_float(env, "MIHARI_MIN_SCORE", base.min_score),
        device=_get(env, "MIHARI_DEVICE") or base.device,
        model_dir=Path(_get(env, "MIHARI_MODEL_DIR") or base.model_dir),
        hfov_deg=_float(env, "MIHARI_HFOV_DEG", base.hfov_deg),
        pan_sign=pan_sign,
        notify=parse_notify(notify) if notify else base.notify,
        thresholds=thresholds,
        ac=ac,
        ir=ir,
        enable_ir=False,
        output_dir=Path(_get(env, "MIHARI_OUTPUT_DIR") or base.output_dir),
    )
