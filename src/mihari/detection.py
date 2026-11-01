"""人物検出（AI の推論だけを使う。学習はしない）。

本番は torchvision の Faster R-CNN（COCO で学習済みの公開の重み）で「person」の箱を出す。
GPU（CUDA）で動かすのが既定で、CUDA が使えない時は黙って CPU に落とさずに止まる。
CPU で動かしたい時は ``device="cpu"`` を明示する（GPU の無い人が試す時のため）。

``MarkerDetector`` はテストとシミュレーション専用の検出器で、真っ赤紫（マゼンタ）の塊を「人」とみなす。
AI の重みが無くても判断の流れ全体を試せるようにするためのもの。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """1 つの検出。

    Attributes:
        box: 箱 ``(x1, y1, x2, y2)``（画素）。
        score: 確からしさ（0..1）。
        label: 種類（ここでは常に ``"person"``）。
    """

    box: tuple[float, float, float, float]
    score: float
    label: str = "person"

    @property
    def cx(self) -> float:
        """箱の中心の横位置。"""
        return (self.box[0] + self.box[2]) / 2

    @property
    def height(self) -> float:
        """箱の高さ。"""
        return self.box[3] - self.box[1]


class Detector(Protocol):
    """人物検出器の約束。"""

    name: str

    def detect(self, image: np.ndarray) -> list[Detection]:
        """RGB の写真（H, W, 3・uint8）から人の箱を返す（確かな順）。"""
        ...


class GpuUnavailableError(RuntimeError):
    """CUDA で動かす設定なのに GPU が使えない時に上げる（CPU へは黙って落とさない）。"""


class MarkerDetector:
    """テスト・シミュレーション用: マゼンタ（R と B が高く G が低い）の塊を人とみなす。

    Args:
        min_pixels: 人とみなす塊の画素数の下限。
    """

    name = "marker"

    def __init__(self, min_pixels: int = 30) -> None:
        self.min_pixels = min_pixels

    def detect(self, image: np.ndarray) -> list[Detection]:
        rgb = image[..., :3].astype(np.int32)
        mask = (rgb[..., 0] > 180) & (rgb[..., 2] > 180) & (rgb[..., 1] < 90)
        if int(mask.sum()) < self.min_pixels:
            return []
        # 列ごとに塊を分ける（横に離れた塊は別の人）。
        cols = np.where(mask.any(axis=0))[0]
        groups: list[list[int]] = [[int(cols[0])]]
        for c in cols[1:]:
            if c - groups[-1][-1] > 3:
                groups.append([int(c)])
            else:
                groups[-1].append(int(c))
        found: list[Detection] = []
        for group in groups:
            sub = mask[:, group[0]: group[-1] + 1]
            if int(sub.sum()) < self.min_pixels:
                continue
            rows = np.where(sub.any(axis=1))[0]
            box = (float(group[0]), float(rows[0]), float(group[-1] + 1), float(rows[-1] + 1))
            found.append(Detection(box=box, score=0.99))
        return found


#: 使えるモデルの名前と torchvision の組み立て方。
MODEL_CHOICES = ("resnet50", "mobilenet")


class TorchvisionPersonDetector:
    """torchvision の Faster R-CNN（COCO 学習済み）で人を検出する。

    Args:
        model: ``"resnet50"``（Faster R-CNN ResNet-50 FPN v2・精度重視）か
            ``"mobilenet"``（Faster R-CNN MobileNetV3-Large FPN・速さ重視）。
        device: ``"cuda"``（既定）か ``"cpu"``。cuda の時に GPU が使えなければ ``GpuUnavailableError``。
        min_score: 返す検出の確からしさの下限。
        model_dir: 重みの置き場（``TORCH_HOME`` に設定する）。初回は公開の配布元から自動で取得する。
        input_min_size: モデルに入れる前に縮める・伸ばす短辺の画素（既定 480）。本体カメラは 320x240 なので、
            torchvision の既定の 800 まで引き伸ばすと推論が約 2 倍遅くなるだけで検出はほぼ変わらない
            （母艦の RTX 3070 Laptop で 1 枚 345 ms → 173 ms・同じ写真で person 0.999）。

    Raises:
        GpuUnavailableError: device が cuda なのに CUDA が使えない時。
        ValueError: 知らないモデル名・装置名の時。
    """

    def __init__(
        self,
        model: str = "resnet50",
        *,
        device: str = "cuda",
        min_score: float = 0.8,
        model_dir: Path | None = None,
        input_min_size: int = 480,
    ) -> None:
        import torch

        if model not in MODEL_CHOICES:
            raise ValueError(f"unknown model '{model}' (choose from {', '.join(MODEL_CHOICES)})")
        if device not in ("cuda", "cpu"):
            raise ValueError("device must be 'cuda' or 'cpu'")
        if device == "cuda" and not torch.cuda.is_available():
            raise GpuUnavailableError(
                "CUDA が使えません。GPU 版の torch が入っているか（README の手順）と、NVIDIA のドライバを確かめてください。"
                " GPU の無い環境で試すだけなら --device cpu を明示してください（黙って CPU には落としません）。"
            )
        if model_dir is not None:
            os.environ["TORCH_HOME"] = str(Path(model_dir).resolve())
        from torchvision.models import detection as tvd

        if model == "resnet50":
            weights = tvd.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            net = tvd.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=min_score,
                                                 min_size=input_min_size, max_size=max(800, input_min_size * 2))
        else:
            weights = tvd.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
            net = tvd.fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=min_score,
                                                        min_size=input_min_size, max_size=max(800, input_min_size * 2))
        categories = list(weights.meta["categories"])
        self.person_label = categories.index("person")
        self.torch = torch
        self.device = torch.device(device)
        self.net = net.eval().to(self.device)
        actual = next(self.net.parameters()).device.type
        if actual != device:
            raise GpuUnavailableError(f"model is on {actual}, expected {device}")
        self.min_score = min_score
        self.name = f"fasterrcnn-{model}@{device}"
        self.last_inference_ms: float = 0.0

    def detect(self, image: np.ndarray) -> list[Detection]:
        torch = self.torch
        tensor = torch.from_numpy(np.array(image[..., :3], dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div(255.0)
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.net([tensor.to(self.device)])[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_inference_ms = (time.perf_counter() - start) * 1000
        boxes = out["boxes"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        found = [
            Detection(box=tuple(float(v) for v in box), score=float(score))
            for box, label, score in zip(boxes, labels, scores)
            if int(label) == self.person_label and float(score) >= self.min_score
        ]
        found.sort(key=lambda d: d.score, reverse=True)
        return found


def build_detector(kind: str, *, model: str, device: str, min_score: float, model_dir: Path) -> Detector:
    """名前から検出器を作る（``torch`` か ``marker``）。"""
    if kind == "marker":
        return MarkerDetector()
    if kind == "torch":
        return TorchvisionPersonDetector(model, device=device, min_score=min_score, model_dir=model_dir)
    raise ValueError(f"unknown detector '{kind}'")
