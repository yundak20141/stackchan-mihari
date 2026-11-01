"""結果の絵（動画に映す・記録に残す）。

見回した写真を横に並べて検出の箱と pan を描き、振り向いた後の写真と判断の文を下に付けた 1 枚を作る。
文字は英数字だけ（OpenCV の既定の書体は日本語を描けないため。日本語の理由は記録の JSONL に残す）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

if TYPE_CHECKING:
    from .patrol import PatrolReport

GREEN = (60, 220, 60)
YELLOW = (0, 210, 255)
WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
RED = (60, 60, 230)


def _draw_view(image_rgb: np.ndarray, detections: Sequence, label: str, highlight: bool) -> np.ndarray:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.box)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), GREEN, 2)
        cv2.putText(bgr, f"person {det.score:.2f}", (x1 + 2, max(14, y1 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
    h, w = bgr.shape[:2]
    cv2.line(bgr, (w // 2, h - 12), (w // 2, h), WHITE, 1)
    cv2.rectangle(bgr, (0, 0), (w - 1, 22), (0, 0, 0), -1)
    cv2.putText(bgr, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW if highlight else WHITE, 1, cv2.LINE_AA)
    if highlight:
        cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), YELLOW, 3)
    return bgr


def _fit(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA) if (img.shape[1], img.shape[0]) != size else img


def render_report(report: "PatrolReport") -> np.ndarray:
    """1 回の見回しの結果を 1 枚の絵（BGR）にする。"""
    tile = (320, 240)
    tiles = []
    best_pan = report.sightings[0].view_pan if report.sightings else None
    # 部屋の左から右の順に並べる（pan を増やすと左を向く取り付けなら pan の大きい順）。
    ordered = sorted(report.views, key=lambda v: v.pan * report.calibration.pan_sign)
    for view in ordered:
        label = f"pan {view.pan}" + (f"  {len(view.detections)} person" if view.detections else "")
        tiles.append(_fit(_draw_view(view.frame.image(), view.detections, label, view.pan == best_pan), tile))
    top = np.hstack(tiles) if tiles else np.zeros((tile[1], tile[0], 3), np.uint8)
    width = top.shape[1]

    lines: list[tuple[str, tuple[int, int, int]]] = []
    if report.target_pan is not None and report.confirmed is False:
        lines.append((f"NOT CONFIRMED: score {report.sightings[0].score:.2f} at pan {report.sightings[0].view_pan}"
                      f" -> turned to pan {report.target_pan}, no person there", YELLOW))
        lines.append(("treated as no person (no notify)", GRAY))
    elif report.target_pan is not None:
        lines.append((f"FOUND {len(report.sightings)} person -> turned to pan {report.target_pan}"
                      f" (score {report.sightings[0].score:.2f})", GREEN))
        if report.residual_deg is not None:
            lines.append((f"after turning: person is {report.residual_deg:+.1f} deg from center", GREEN))
        if report.notified:
            lines.append(("notified: " + ", ".join(report.notified), GREEN))
    else:
        lines.append(("no person found", GRAY))
    cal = report.calibration
    lines.append((f"calibration: hfov {cal.hfov_deg:.1f} deg, pan_sign {cal.pan_sign:+d} ({cal.source})", WHITE))
    env = report.env
    if env is not None and env.valid:
        lines.append((f"room: {env.temp_c:.1f} C, {env.humidity:.0f} %RH -> {report.ac_decision.comfort.value}", WHITE))
    else:
        lines.append(("room: no temperature reading", GRAY))
    ir = report.ir_outcome
    action = report.ac_decision.action
    if action == "none":
        lines.append(("AC: no action", GRAY))
    elif ir.status == "dry_run":
        lines.append((f"AC: {action.upper()} -> IR DISABLED (would send {ir.payload})", YELLOW))
    elif ir.status == "sent":
        lines.append((f"AC: {action.upper()} -> IR SENT ({ir.payload})", RED))
    else:
        lines.append((f"AC: {action.upper()} -> IR {ir.status}", RED))
    lines.append((f"detector: {report.detector}  infer {report.infer_ms_mean:.0f} ms/frame  "
                  f"patrol {report.elapsed_s:.1f} s", GRAY))

    after_h = tile[1] if report.after_frame is not None else 0
    text_h = 26 * len(lines) + 12
    panel_h = max(after_h, text_h)
    panel = np.zeros((panel_h, width, 3), np.uint8)
    x0 = 10
    if report.after_frame is not None:
        after = _fit(_draw_view(report.after_frame.image(), report.after_detections,
                                f"after turn: pan {report.after_frame.pan}", True), tile)
        panel[:tile[1], :tile[0]] = after
        x0 = tile[0] + 14
    for i, (text, color) in enumerate(lines):
        cv2.putText(panel, text[: max(20, (width - x0) // 9)], (x0, 26 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return np.vstack([top, panel])
