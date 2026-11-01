"""人物検出器が GPU の無い時に黙って CPU へ落ちないことのテスト（重みは読み込まない）。"""
from __future__ import annotations

import numpy as np
import pytest

from mihari.detection import GpuUnavailableError, MarkerDetector, TorchvisionPersonDetector, build_detector


def test_cuda_requested_but_unavailable_raises(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(GpuUnavailableError):
        TorchvisionPersonDetector("resnet50", device="cuda")


def test_unknown_model_and_device_are_rejected() -> None:
    with pytest.raises(ValueError):
        TorchvisionPersonDetector("yolo-nonexistent", device="cuda")
    with pytest.raises(ValueError):
        TorchvisionPersonDetector("resnet50", device="mps")


def test_build_detector_names() -> None:
    assert isinstance(build_detector("marker", model="resnet50", device="cuda", min_score=0.5,
                                     model_dir=__import__("pathlib").Path(".models")), MarkerDetector)
    with pytest.raises(ValueError):
        build_detector("nope", model="resnet50", device="cuda", min_score=0.5,
                       model_dir=__import__("pathlib").Path(".models"))


def test_marker_detector_finds_separate_blobs() -> None:
    img = np.full((240, 320, 3), (120, 160, 120), np.uint8)
    img[50:200, 30:60] = (255, 0, 255)
    img[80:220, 200:240] = (255, 0, 255)
    found = MarkerDetector().detect(img)
    assert sorted(round(d.cx) for d in found) == [45, 220]
    assert MarkerDetector().detect(np.zeros((240, 320, 3), np.uint8)) == []
