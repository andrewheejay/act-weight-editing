"""Assembles report/report.md: the referee's verdict block copied byte for byte, then the body text in report/body.md. Re-run after editing body.md."""
from pathlib import Path

T = Path(__file__).resolve().parents[1]
v = (T / "test" / "VERDICT.md").read_text()
b, e = "<!-- referee-verdict-begin -->", "<!-- referee-verdict-end -->"
block = v[v.find(b): v.find(e) + len(e)]
body = (T / "report" / "body.md").read_text()
assert "{{VERDICT}}" in body
(T / "report" / "report.md").write_text(body.replace("{{VERDICT}}", block))
