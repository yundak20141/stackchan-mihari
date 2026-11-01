"""公開前の点検（scripts/check_public_safe.py）のテスト: わざと仕込んだ物を見つけ、きれいなら通す。

仕込む値はすべて架空で、部品をつないで実行時に作る（このテストのファイル自体が点検に引っかからないように）。
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_public_safe.py"
spec = importlib.util.spec_from_file_location("check_public_safe", SCRIPT)
cps = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = cps  # dataclass が自分のモジュールを引けるように
spec.loader.exec_module(cps)

NL = "\n"
BS = "\\"
DOT = "."
AT = "@"


def _j(*parts: str) -> str:
    """部品をつなぐ。"""
    return "".join(parts)


FAKE_LAN = DOT.join(["192", "168", "77", "123"])
FAKE_TAILNET = DOT.join(["100", "99", "12", "34"])
FAKE_TEN = DOT.join(["10", "0", "0", "5"])
FAKE_MAIL = _j("someone", AT, "gmail", DOT, "com")

PLANTED = [
    (_j('HOST = "', FAKE_LAN, '"', NL), "private ip"),
    (_j('TS = "', FAKE_TAILNET, '"', NL), "private ip"),
    (_j('NET = "', FAKE_TEN, '"', NL), "private ip"),
    (_j("#define WIFI_", 'SSID "', "fake-ap", '"', NL), "wifi credential"),
    (_j("ss", "id: FakeNet5G", NL), "wifi credential"),
    (_j('KEY = "', "s", "k-", "a" * 32, '"', NL), "openai/anthropic style key"),
    (_j('T = "', "gh", "p_", "b" * 36, '"', NL), "github token"),
    (_j("api", '_key = "', "c" * 18, '"', NL), "assigned secret"),
    (_j("-----BEGIN OPENSSH ", "PRIVATE KEY-----", NL), "private key"),
    (_j("contact: ", FAKE_MAIL, NL), "email"),
    (_j('path = "C:', BS, "Users", BS, 'alice"', NL), "user path"),
]


def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "tester"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "tester@users.noreply.github.com"], check=True)
    return tmp_path


def kinds(findings) -> set[str]:
    return {f.kind for f in findings}


def test_clean_repo_passes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MIHARI_PRIVATE_WORDS", raising=False)
    root = git_repo(tmp_path)
    (root / "app.py").write_text(
        'URL = "ws://127.0.0.1:8080"  # localhost\nDOC = "ws://192.0.2.10:8080"  # RFC 5737\n'
        'VERSION = "1.2.3.4"\nAUTHOR = "noreply@anthropic.com"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "app.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "init"], check=True)
    assert cps.check(root) == []


@pytest.mark.parametrize(("text", "kind"), PLANTED)
def test_planted_findings_are_detected(tmp_path: Path, monkeypatch, text: str, kind: str) -> None:
    monkeypatch.delenv("MIHARI_PRIVATE_WORDS", raising=False)
    root = git_repo(tmp_path)
    (root / "leak.py").write_text(text, encoding="utf-8")
    assert kind in kinds(cps.check(root))


def test_private_words_from_env(tmp_path: Path, monkeypatch) -> None:
    root = git_repo(tmp_path)
    (root / "notes.md").write_text("作者: Taro Example\n", encoding="utf-8")
    monkeypatch.setenv("MIHARI_PRIVATE_WORDS", "taro example")
    found = cps.check(root)
    assert "private word" in kinds(found)
    assert all("Taro Example" not in str(f) for f in found)   # 伏せ字で出す


def test_private_words_two_char_japanese(tmp_path: Path, monkeypatch) -> None:
    root = git_repo(tmp_path)
    (root / "notes.md").write_text("作者: 山田 太郎\n", encoding="utf-8")
    monkeypatch.setenv("MIHARI_PRIVATE_WORDS", "山田,ab")   # 日本語の 2 文字は拾い、英数字の 2 文字は捨てる
    assert cps.load_private_words(root) == ["山田"]
    assert "private word" in kinds(cps.check(root))


def test_forbidden_and_large_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MIHARI_PRIVATE_WORDS", raising=False)
    root = git_repo(tmp_path)
    (root / ".env").write_text("X=1\n", encoding="utf-8")
    (root / "big.bin").write_bytes(b"\0" * (cps.MAX_BYTES + 1))
    found = kinds(cps.check(root))
    assert "forbidden file" in found
    assert "file over 10MB" in found


def test_ignored_files_are_not_scanned(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MIHARI_PRIVATE_WORDS", raising=False)
    root = git_repo(tmp_path)
    (root / ".gitignore").write_text("runs/\n", encoding="utf-8")
    (root / "runs").mkdir()
    (root / "runs" / "log.txt").write_text(FAKE_LAN + NL, encoding="utf-8")
    assert cps.check(root) == []


def test_commit_email_is_checked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MIHARI_PRIVATE_WORDS", raising=False)
    root = git_repo(tmp_path)
    (root / "a.txt").write_text("ok\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "a.txt"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.email=" + _j("person", AT, "gmail", DOT, "com"),
                    "commit", "-q", "-m", "x"], check=True)
    assert "commit email" in kinds(cps.check(root))


def test_history_is_scanned_even_after_removal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MIHARI_PRIVATE_WORDS", raising=False)
    root = git_repo(tmp_path)
    leak = root / "cfg.py"
    leak.write_text(_j('HOST = "', FAKE_LAN, '"', NL), encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "cfg.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "leak"], check=True)
    leak.write_text('HOST = "127.0.0.1"\n', encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-am", "fix"], check=True)
    found = cps.check(root)
    assert [f.kind for f in found] == ["private ip"]
    assert found[0].path.startswith("(history ")
    assert cps.check(root, history=False) == []
