"""Build publish/post.md from publish/post_body.md and the two referee verdict blocks.

python publish/write_post.py

The line {{VERDICT:<task>}} in the body is replaced by the block between the referee-verdict markers of
experiments/<task>/test/VERDICT.md, copied byte for byte. Nothing else is filled in, so every number in the body is typed
by hand from the files named next to it.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- referee-verdict-begin -->", "<!-- referee-verdict-end -->"


def block(task):
    text = (ROOT / "experiments" / task / "test" / "VERDICT.md").read_text()
    i, j = text.index(BEGIN), text.index(END) + len(END)
    return text[i:j]


def main():
    body = (ROOT / "publish" / "post_body.md").read_text()
    out = re.sub(r"\{\{VERDICT:(\w+)\}\}", lambda m: block(m.group(1)), body)
    assert "{{" not in out
    (ROOT / "publish" / "post.md").write_text(out)
    for t in ("transfer_cube", "insertion"):
        assert block(t) in out
    print("publish/post.md", len(out.split()), "words")


if __name__ == "__main__":
    main()
