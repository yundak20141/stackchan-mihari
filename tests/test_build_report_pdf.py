"""提出用のレポート PDF の下ごしらえ（scripts/build_report_pdf.py）のテスト。ブラウザは起こさない。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_report_pdf.py"
spec = importlib.util.spec_from_file_location("build_report_pdf", SCRIPT)
brp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = brp
spec.loader.exec_module(brp)

MD = """# 説明資料（下書き）: 見張り番

提出名: 説明資料_PAI最終課題_<ACCOUNT>.pdf

> 下書きの扱い: 【実機で埋める】の所は後で入れる。

## 1. 概要

- 首を **左右** に振る
- `--enable-ir` の時だけ送る

| 試したこと | 結果 |
|---|---|
| 人が左 | ずれ −0.3 度 |
"""


def test_prepare_drops_draft_marks_and_fills_account() -> None:
    out = brp.prepare(MD, "demo_id")
    assert out.splitlines()[0] == "# 説明資料: 見張り番"
    assert "下書きの扱い" not in out
    assert "<ACCOUNT>" not in out and "説明資料_PAI最終課題_demo_id.pdf" in out


def test_prepare_keeps_placeholder_without_account() -> None:
    out = brp.prepare(MD, brp.PLACEHOLDER)
    assert "<ACCOUNT>" in out


def test_md_to_html_renders_blocks_and_escapes() -> None:
    html = brp.md_to_html(brp.prepare(MD, brp.PLACEHOLDER))
    assert "<h1>説明資料: 見張り番</h1>" in html
    assert "&lt;ACCOUNT&gt;" in html and "<ACCOUNT>" not in html   # 山かっこはタグにしない
    assert "<li>首を <strong>左右</strong> に振る</li>" in html
    assert "<code>--enable-ir</code>" in html
    assert "<th>試したこと</th>" in html and "<td>ずれ −0.3 度</td>" in html
    assert "---" not in html                                        # 表の区切り行は出さない


def test_md_to_html_embeds_images_from_base_dir(tmp_path: Path) -> None:
    from PIL import Image

    Image.new("RGB", (4, 3), (10, 200, 140)).save(tmp_path / "shot.jpg")
    html = brp.md_to_html("![見回しの結果](shot.jpg)\n\n本文", base=tmp_path)
    assert '<figure><img src="data:image/jpeg;base64,' in html          # 外のファイルに頼らず埋め込む
    assert "<figcaption>見回しの結果</figcaption></figure>" in html
    assert "<p>本文</p>" in html


def test_md_to_html_stops_on_missing_image(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(SystemExit):
        brp.md_to_html("![無い絵](nothing.png)", base=tmp_path)
