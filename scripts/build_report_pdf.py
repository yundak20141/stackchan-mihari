"""提出用のレポート PDF を docs/report-draft.md から作る（画面に何も出さない）。

- Markdown は外の部品を使わず、レポートで使う書き方（見出し・段落・箇条書き・番号つき・表・
  インラインのコード・太字・引用・1 行だけの絵）だけをこの中で HTML に直す。絵は HTML に埋め込む
- 「下書きの扱い」の引用は提出物から外し、``<ACCOUNT>`` を ``--account`` の値に置き換える
- PDF は Chrome の headless で印刷する。使い捨てのプロフィールを使うので、普段の Chrome には触れない

使い方: ``python scripts/build_report_pdf.py --account <Omnicampus のアカウント名か ID>``
出力: ``dist/説明資料_PAI最終課題_<アカウント>.pdf``（dist/ は git に入れない）
"""
from __future__ import annotations

import argparse
import base64
import html
import mimetypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = "<ACCOUNT>"
CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: "Yu Gothic UI", "Yu Gothic", "Meiryo", sans-serif; font-size: 10.5pt; line-height: 1.6; color: #111; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 4px; margin: 0 0 10px; }
h2 { font-size: 13pt; border-left: 5px solid #333; padding-left: 8px; margin: 18px 0 6px; }
h3 { font-size: 11pt; margin: 12px 0 4px; }
p, ul, ol { margin: 4px 0 8px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 3px 6px; vertical-align: top; }
th { background: #eee; }
code { font-family: Consolas, "MS Gothic", monospace; background: #f2f2f2; padding: 0 3px; }
blockquote { margin: 6px 0; padding: 4px 10px; border-left: 3px solid #aaa; color: #444; }
figure { margin: 8px 0 12px; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 90mm; border: 1px solid #ccc; }
figcaption { font-size: 9pt; color: #444; margin-top: 3px; }
"""


def inline(text: str) -> str:
    """1 行の中の書き方（インラインのコード・太字）を HTML に直す。ほかの文字はエスケープする。"""
    parts = re.split(r"(`[^`]+`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            out.append("<code>" + html.escape(part[1:-1]) + "</code>")
        else:
            esc = html.escape(part)
            out.append(re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc))
    return "".join(out)


def figure(alt: str, src: str, base: Path) -> str:
    """1 行だけの絵（``![説明](パス)``）を、画像を埋め込んだ figure にする。絵が無ければ止まる。"""
    path = base / src
    if not path.is_file():
        raise SystemExit(f"レポートの絵が見つかりません: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<figure><img src="data:{mime};base64,{data}" alt="{html.escape(alt)}">'
            f"<figcaption>{inline(alt)}</figcaption></figure>")


def md_to_html(md: str, base: Path | None = None) -> str:
    """レポートで使う範囲の Markdown を HTML の本文に直す。絵のパスは ``base``（既定は docs/）から見る。"""
    base = base if base is not None else ROOT / "docs"
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s)
        if m:
            out.append(figure(m.group(1), m.group(2), base))
            i += 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
            i += 1
            continue
        if s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            body = [r for r in cells if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]
            head, rest = body[0], body[1:]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>")
            out += ["<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rest]
            out.append("</tbody></table>")
            continue
        if s.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + "<br>".join(inline(q) for q in quote if q) + "</blockquote>")
            continue
        if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
            ordered = bool(re.match(r"^\d+\.\s+", s))
            pat = r"^\d+\.\s+" if ordered else r"^[-*]\s+"
            items = []
            while i < len(lines) and re.match(pat, lines[i].strip()):
                items.append(re.sub(pat, "", lines[i].strip()))
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(t)}</li>" for t in items) + f"</{tag}>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,3}\s|\||>|[-*]\s|\d+\.\s|!\[)", lines[i].strip()):
            para.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline("".join(para)) + "</p>")
    return "\n".join(out)


def prepare(md: str, account: str) -> str:
    """提出物にする前の手当て: 下書きの注意書きを外し、アカウント名を入れる。"""
    kept = [ln for ln in md.splitlines() if not ln.strip().startswith("> 下書きの扱い")]
    text = "\n".join(kept).replace(PLACEHOLDER, account)
    return re.sub(r"^(# .*?)（下書き）", r"\1", text, count=1, flags=re.M)


def build_html(md: str, account: str, base: Path | None = None) -> str:
    """PDF にする HTML 全体を作る。"""
    body = md_to_html(prepare(md, account), base)
    return f"<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\"><style>{CSS}</style></head><body>{body}</body></html>"


def find_browser() -> Path:
    """PDF の印刷に使うブラウザを探す（無ければ止まる）。"""
    for p in CHROME_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("Chrome も Edge も見つからないので PDF を作れません")


def main() -> int:
    ap = argparse.ArgumentParser(description="提出用のレポート PDF を作る")
    ap.add_argument("--account", default=PLACEHOLDER, help="Omnicampus のアカウント名か ID（ファイル名と本文に入る）")
    ap.add_argument("--src", default=str(ROOT / "docs" / "report-draft.md"))
    ap.add_argument("--out-dir", default=str(ROOT / "dist"))
    args = ap.parse_args()

    md = Path(args.src).read_text(encoding="utf-8")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.account if args.account != PLACEHOLDER else "ACCOUNT"
    html_path = out_dir / "report.html"
    pdf_path = out_dir / f"説明資料_PAI最終課題_{name}.pdf"
    html_path.write_text(build_html(md, args.account, Path(args.src).parent), encoding="utf-8")

    profile = tempfile.mkdtemp(prefix="mihari-pdf-")
    try:
        cmd = [str(find_browser()), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
               f"--user-data-dir={profile}", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", html_path.as_uri()]
        res = subprocess.run(cmd, capture_output=True, timeout=120)
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    if res.returncode != 0 or not pdf_path.exists():
        sys.stderr.write(res.stderr.decode("utf-8", "replace")[-2000:])
        return 1
    print(f"[build_report_pdf] {pdf_path} ({pdf_path.stat().st_size} bytes)")
    if args.account == PLACEHOLDER:
        print("[build_report_pdf] 注意: --account を付けていないので、本文とファイル名のアカウントは仮のまま")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
