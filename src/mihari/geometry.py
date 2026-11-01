"""首の角度の計算。

- 写真の中の人の位置（横の画素）から、その人の方向の pan を求める
- 見回しで撮った隣どうしの写真のずれ（位相限定相関）から、画角と「pan を増やすとどちらを向くか」を測る
- 隣の写真に同じ人が写った時に 1 人にまとめる

サーボの pan は 45..135（90 が正面）。カメラは頭に付いているので、首を回すと写る景色が横にずれる。
そのずれの量と向きを見回しのたびに測れば、画角や取り付けの向きを決め打ちしなくてよい。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .config import PAN_MAX, PAN_MIN, TILT_MAX, TILT_MIN


def clamp_pan(pan: float) -> int:
    """pan を可動範囲 45..135 の整数に収める。"""
    return int(round(max(PAN_MIN, min(PAN_MAX, pan))))


def clamp_tilt(tilt: float) -> int:
    """tilt を可動範囲 60..120 の整数に収める。"""
    return int(round(max(TILT_MIN, min(TILT_MAX, tilt))))


@dataclass(frozen=True)
class PanCalibration:
    """写真の横位置と pan の関係。

    Attributes:
        pan_sign: 写真の右にある物へ向くには pan を増やすなら +1、減らすなら -1。
        hfov_deg: 写真の横幅が何度に当たるか（画角）。
        source: どこから決めたか（``"scan"`` = 見回しの写真から測った / ``"config"`` = 設定値）。
        pairs_used: 測るのに使えた隣どうしの組の数。
    """

    pan_sign: int
    hfov_deg: float
    source: str
    pairs_used: int = 0

    def offset_deg(self, cx_px: float, width_px: int) -> float:
        """写真の中心から右へ何度ずれているか（左は負）。

        Args:
            cx_px: 人の箱の中心の横位置（画素）。
            width_px: 写真の幅（画素）。
        """
        return (cx_px / width_px - 0.5) * self.hfov_deg

    def target_pan(self, view_pan: float, cx_px: float, width_px: int) -> float:
        """その写真を撮った pan と人の横位置から、人の方向の pan を求める（可動範囲には収めない）。"""
        return view_pan + self.pan_sign * self.offset_deg(cx_px, width_px)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """RGB（H, W, 3）か灰色（H, W）を float の灰色にする。"""
    if image.ndim == 3:
        image = image[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    return image.astype(np.float64)


def _tukey(n: int, alpha: float = 0.3) -> np.ndarray:
    """端だけをなだらかに 0 へ落とす窓（Tukey 窓）。Hanning より重なりの情報を多く残す。"""
    window = np.ones(n)
    k = int(alpha * (n - 1) / 2)
    if k > 0:
        t = np.arange(k)
        window[:k] = 0.5 * (1 - np.cos(np.pi * t / k))
        window[-k:] = window[:k][::-1]
    return window


def estimate_shift(image_a: np.ndarray, image_b: np.ndarray) -> tuple[float, float, float]:
    """2 枚の写真の景色のずれを位相限定相関で求める。

    b の景色が a から右へ dx 画素・下へ dy 画素ずれていれば ``(dx, dy, strength)`` を返す。
    strength は相関の山の高さを相関面全体の標準偏差で割った値（無関係な 2 枚ではおよそ 5〜7、
    重なりのある 2 枚では 10 以上になる）。2 枚は同じ大きさであること。

    Args:
        image_a: 1 枚目（RGB か灰色の配列）。
        image_b: 2 枚目（同じ大きさ）。

    Returns:
        ``(dx, dy, strength)``。

    Raises:
        ValueError: 大きさが違う時。
    """
    a = _to_gray(image_a)
    b = _to_gray(image_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    h, w = a.shape
    window = np.outer(_tukey(h), _tukey(w))
    fa = np.fft.fft2((a - a.mean()) * window)
    fb = np.fft.fft2((b - b.mean()) * window)
    cross = fb * np.conj(fa)
    cross /= np.maximum(np.abs(cross), 1e-12)
    corr = np.fft.ifft2(cross).real
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    std = float(corr.std())
    strength = float(corr[iy, ix]) / std if std > 0 else 0.0

    def refine(c_m: float, c_0: float, c_p: float) -> float:
        """3 点の放物線で山の位置を小数点以下まで詰める。"""
        denom = c_m - 2 * c_0 + c_p
        return 0.0 if abs(denom) < 1e-12 else 0.5 * (c_m - c_p) / denom

    fx = refine(corr[iy, (ix - 1) % w], corr[iy, ix], corr[iy, (ix + 1) % w])
    fy = refine(corr[(iy - 1) % h, ix], corr[iy, ix], corr[(iy + 1) % h, ix])
    dx = ix + fx
    dy = iy + fy
    if dx > w / 2:
        dx -= w
    if dy > h / 2:
        dy -= h
    return float(dx), float(dy), strength


#: ずれを測った組を信じる相関の強さ（estimate_shift の strength）の下限。
MIN_STRENGTH = 8.0


def calibrate_from_views(
    views: Sequence[tuple[float, np.ndarray]],
    fallback: PanCalibration,
    *,
    min_strength: float = MIN_STRENGTH,
) -> PanCalibration:
    """見回しで撮った写真（pan の順）から、画角と向きを測る。

    隣どうしの組ごとに景色の横のずれ dx を測り、``dx / Δpan``（1 度あたりの画素）を集める。
    首を右へ回すと景色は左へ流れるので、「pan を増やすと景色が左へ流れる」なら pan を増やすと右を向く
    （pan_sign = +1）。組の過半数で向きがそろい、値のばらつきが小さい時だけ採用し、
    そうでなければ設定値（fallback）を返す。

    Args:
        views: ``(pan, 写真の配列)`` の並び（同じ大きさの写真）。
        fallback: 測れなかった時に返す設定値。
        min_strength: 組を信じる相関の強さの下限。

    Returns:
        測れた時は ``source="scan"`` の校正、測れなかった時は fallback。
    """
    ordered = sorted(views, key=lambda v: v[0])
    rates: list[float] = []
    width = None
    for (pan_a, img_a), (pan_b, img_b) in zip(ordered, ordered[1:]):
        dpan = pan_b - pan_a
        if dpan == 0 or img_a.shape != img_b.shape:
            continue
        width = img_a.shape[1]
        dx, dy, strength = estimate_shift(img_a, img_b)
        # 縦に大きくずれた・横のずれが写真の 8 割を超える・相関が弱い組は捨てる。
        if strength < min_strength or abs(dx) > 0.8 * width or abs(dy) > 0.25 * img_a.shape[0]:
            continue
        if abs(dx) < 1.0:
            continue
        rates.append(dx / dpan)
    if width is None or not rates:
        return fallback
    signs = [1 if r > 0 else -1 for r in rates]
    majority = 1 if sum(signs) > 0 else -1
    agreeing = [abs(r) for r, s in zip(rates, signs) if s == majority]
    if len(agreeing) * 2 <= len(rates):
        return fallback
    px_per_deg = float(np.median(agreeing))
    spread = (max(agreeing) - min(agreeing)) / px_per_deg if len(agreeing) > 1 else 0.0
    if spread > 0.5:
        return fallback
    hfov = width / px_per_deg
    if not 10.0 <= hfov <= 150.0:
        return fallback
    # 景色が右へ流れる（rate > 0）なら pan を増やすと左を向いている → 右の物には pan を減らす。
    pan_sign = -majority
    return PanCalibration(pan_sign=pan_sign, hfov_deg=hfov, source="scan", pairs_used=len(agreeing))


@dataclass(frozen=True)
class Sighting:
    """見回しで見つけた 1 人（隣の写真の重なりは 1 人にまとめた後）。

    Attributes:
        world_pan: その人の方向の pan（可動範囲には収めない）。
        score: 確からしさ（まとめた中で一番高い値）。
        view_pan: 一番確かな検出を撮った pan。
        box: その検出の箱 ``(x1, y1, x2, y2)``（画素）。
        views: この人が写っていた写真の数。
    """

    world_pan: float
    score: float
    view_pan: float
    box: tuple[float, float, float, float]
    views: int = 1


def merge_sightings(
    candidates: Iterable[Sighting],
    *,
    same_person_deg: float = 10.0,
) -> list[Sighting]:
    """方向が ``same_person_deg`` 以内の検出を 1 人にまとめる（確かな順に代表を選ぶ）。

    Args:
        candidates: 写真ごとの検出から作った候補。
        same_person_deg: 同じ人と見なす方向の差（度）。

    Returns:
        確かな順の人の並び。
    """
    merged: list[Sighting] = []
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        for i, rep in enumerate(merged):
            if abs(rep.world_pan - cand.world_pan) <= same_person_deg:
                merged[i] = Sighting(rep.world_pan, rep.score, rep.view_pan, rep.box, rep.views + 1)
                break
        else:
            merged.append(cand)
    return merged
