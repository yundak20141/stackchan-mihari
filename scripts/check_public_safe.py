"""公開前の点検: リポジトリに外へ出してはいけない物が入っていないかを探す。

調べる物（git に入る予定のファイル＝追跡中と、無視されていない未追跡のファイル）:

1. 家の中のアドレス: 192.168.x.x・10.x.x.x・172.16〜31.x.x・100.64〜127.x.x（Tailscale など）・169.254.x.x
2. Wi-Fi の SSID・パスワードの書き込み（``ssid = "..."`` のような代入）
3. トークン・鍵らしい文字列（sk-・ghp_・AKIA・JWT・秘密鍵の見出し・token= の長い値 など）
4. メールアドレス（noreply の宛先と example.* は除く）
5. 利用者名の見える絶対パス（C:\\Users\\<名前>・/home/<名前>・/Users/<名前>）
6. 外に出してはいけない言葉（環境変数 MIHARI_PRIVATE_WORDS か ``.mihari/private_words.txt`` に
   カンマか改行で並べる。点検の中身には書かない）
7. 10MB を超えるファイル
8. 入っていてはいけないファイル（.env・config_private.h・秘密鍵）
9. git の履歴の作者・コミッターのメールアドレス
10. git の履歴の全 commit で足された行（1〜6 と同じ物。いまは消した物も公開すると履歴から読めるため）

見つかった物は「ファイル:行: 種類: 伏せ字の抜粋」で出し、1 件でもあれば終了コード 1。
使い方: ``python scripts/check_public_safe.py``（リポジトリの直下で）
"""
from __future__ import annotations

import argparse
import ipaddress
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024

PRIVATE_NETS = [ipaddress.ip_network(n) for n in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "169.254.0.0/16", "fc00::/7",
)]
IPV4 = re.compile(r"(?<![\d.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?![\d.])")
SSID = re.compile(r"(?i)\b(wifi_?ssid|ssid|wifi_?pass(?:word)?|wlan_?pass)\b"
                  r"(?:\s*[:=]\s*[\"']?([^\s\"'<>{}]{2,})|\s+[\"']([^\"']{2,})[\"'])")
TOKEN_PATTERNS = [
    ("openai/anthropic style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("github token", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("aws access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("jwt / tunnel token", re.compile(r"\beyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{10,}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("assigned secret", re.compile(
        r"(?i)\b(token|secret|api[_-]?key|passwd|password)\b\s*[:=]\s*[\"']([^\"'\s]{12,})[\"']")),
]
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
EMAIL_OK = re.compile(r"(?i)^(noreply@anthropic\.com|[^@]+@users\.noreply\.github\.com|[^@]+@example\.(com|org|net)"
                      r"|git@github\.com)$")
USER_PATH = re.compile(r"(?i)([a-z]:[\\/]+users[\\/]+|/home/|/Users/)(?!<|\$|%|public\b|default\b)([A-Za-z0-9._-]+)")
FORBIDDEN_NAMES = re.compile(r"(?i)(^|/)(\.env|config_private\.h|id_rsa|id_ed25519|.*\.pem|.*\.key)$")
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".sh", ".ps1", ".bat",
    ".cmd", ".example", ".gitignore", ".html", ".css", ".js", ".ts", ".csv", ".h", ".c", ".cpp", "",
}


@dataclass
class Finding:
    """見つかった 1 件。"""

    path: str
    line: int
    kind: str
    excerpt: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{where}: {self.kind}: {self.excerpt}"


def mask(text: str) -> str:
    """抜粋を伏せ字にする（先頭 3 文字だけ見せる）。"""
    text = text.strip()
    return text if len(text) <= 3 else text[:3] + "*" * min(8, len(text) - 3)


def git_files(root: Path) -> list[str]:
    """git に入る予定のファイル（追跡中＋無視されていない未追跡）。"""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                         capture_output=True, check=True)
    return sorted({p for p in out.stdout.decode("utf-8", "replace").split("\0") if p})


def load_private_words(root: Path) -> list[str]:
    """外に出してはいけない言葉（環境変数と、git に入らない手元のファイルから）。"""
    words: list[str] = []
    env = os.environ.get("MIHARI_PRIVATE_WORDS", "")
    words += re.split(r"[,\n]", env)
    local = root / ".mihari" / "private_words.txt"
    if local.exists():
        words += re.split(r"[,\n]", local.read_text(encoding="utf-8"))
    # 英数字は 3 文字以上（短いと誤検出が多い）、日本語の姓名は 2 文字でも拾う
    return sorted(
        {w.strip() for w in words if len(w.strip()) >= 3 or (len(w.strip()) >= 2 and not w.strip().isascii())},
        key=str.lower,
    )


def is_text(path: Path) -> bool:
    """中身を読む対象か（拡張子と NUL バイトで判定）。"""
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in ("LICENSE", "Makefile"):
        return False
    try:
        head = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\0" not in head


def scan_text(rel: str, text: str, private_words: list[str]) -> list[Finding]:
    """1 ファイルの中身を調べる。"""
    found: list[Finding] = []
    lowered_words = [(w, w.lower()) for w in private_words]
    for no, line in enumerate(text.splitlines(), 1):
        for m in IPV4.finditer(line):
            try:
                addr = ipaddress.ip_address(m.group(0))
            except ValueError:
                continue
            if any(addr in net for net in PRIVATE_NETS):
                found.append(Finding(rel, no, "private ip", mask(m.group(0))))
        for m in SSID.finditer(line):
            found.append(Finding(rel, no, "wifi credential", f"{m.group(1)}={mask(m.group(2) or m.group(3))}"))
        for kind, pat in TOKEN_PATTERNS:
            for m in pat.finditer(line):
                found.append(Finding(rel, no, kind, mask(m.group(0))))
        for m in EMAIL.finditer(line):
            if not EMAIL_OK.match(m.group(0)):
                found.append(Finding(rel, no, "email", mask(m.group(0))))
        for m in USER_PATH.finditer(line):
            found.append(Finding(rel, no, "user path", mask(m.group(0))))
        low = line.lower()
        for word, lw in lowered_words:
            if lw in low:
                found.append(Finding(rel, no, "private word", mask(word)))
    return found


def scan_git_identities(root: Path, private_words: list[str]) -> list[Finding]:
    """git の履歴の作者・コミッターを調べる（公開すると履歴ごと見える）。"""
    res = subprocess.run(["git", "-C", str(root), "log", "--all", "--format=%an%x00%ae%x00%cn%x00%ce"],
                         capture_output=True)
    if res.returncode != 0:
        return []  # まだ commit が無い
    found: list[Finding] = []
    seen: set[str] = set()
    for line in res.stdout.decode("utf-8", "replace").splitlines():
        for value in line.split("\0"):
            if not value or value in seen:
                continue
            seen.add(value)
            if "@" in value and not EMAIL_OK.match(value):
                found.append(Finding("(git log)", 0, "commit email", mask(value)))
            for word in private_words:
                if word.lower() in value.lower():
                    found.append(Finding("(git log)", 0, "private word in commit identity", mask(word)))
    return found


def scan_git_history(root: Path, private_words: list[str]) -> list[Finding]:
    """git の履歴の全 commit で足された行を調べる（いまは消した物も、公開すると履歴から読める）。"""
    res = subprocess.run(["git", "-C", str(root), "log", "--all", "-p", "--no-color", "--format=commit %h"],
                         capture_output=True)
    if res.returncode != 0:
        return []
    found: list[Finding] = []
    commit = "?"
    path = "?"
    for line in res.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("commit "):
            commit = line.split()[1]
        elif line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+") and not line.startswith("+++"):
            if path == "scripts/check_public_safe.py":
                continue
            for f in scan_text(f"(history {commit}) {path}", line[1:], private_words):
                found.append(f)
    # 同じ物が複数の commit にあれば 1 件にまとめる
    unique: dict[tuple[str, str, str], Finding] = {}
    for f in found:
        unique.setdefault((f.path.split(") ", 1)[-1], f.kind, f.excerpt), f)
    return list(unique.values())


def check(root: Path, *, history: bool = True) -> list[Finding]:
    """リポジトリ全体（いまのファイルと、git の履歴）を調べて、見つかった物を返す。"""
    root = root.resolve()
    private_words = load_private_words(root)
    findings: list[Finding] = []
    for rel in git_files(root):
        path = root / rel
        if not path.is_file():
            continue
        if FORBIDDEN_NAMES.search(rel.replace("\\", "/")):
            findings.append(Finding(rel, 0, "forbidden file", rel))
        size = path.stat().st_size
        if size > MAX_BYTES:
            findings.append(Finding(rel, 0, "file over 10MB", f"{size / 1024 / 1024:.1f} MB"))
        if rel.replace("\\", "/") == "scripts/check_public_safe.py":
            continue  # 点検そのものは模様の定義を持つので中身は調べない（言葉は持たない）
        if is_text(path):
            text = path.read_text(encoding="utf-8", errors="replace")
            findings += scan_text(rel, text, private_words)
    findings += scan_git_identities(root, private_words)
    if history:
        current = {(f.path, f.kind, f.excerpt) for f in findings}
        findings += [f for f in scan_git_history(root, private_words)
                     if (f.path.split(") ", 1)[-1], f.kind, f.excerpt) not in current]
    return findings


def main(argv: list[str] | None = None) -> int:
    """入り口。"""
    p = argparse.ArgumentParser(description="公開前の点検")
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = p.parse_args(argv)
    root = args.root
    files = git_files(root)
    words = load_private_words(root)
    findings = check(root)
    print(f"[check_public_safe] root={root.name} files={len(files)} private_words={len(words)}")
    for f in findings:
        print(f"  NG {f}")
    if findings:
        print(f"[check_public_safe] {len(findings)} finding(s). Fix them before publishing.")
        return 1
    print("[check_public_safe] OK: no private IPs, Wi-Fi credentials, tokens, emails, user paths, "
          "private words, forbidden files or files over 10MB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
