"""Build publish/atlas/index.html, one self-contained page for the 48 heads of both checkpoints.

python publish/build_atlas.py

Reads only committed files: experiments/<task>/stage2/atlas/heads.json and its figures, stage2/sweep_summary.json,
stage2/split_half.json and freeze/edit.json. Runs no rollout and no model. The figures are downscaled and embedded as JPEG.
"""
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "publish" / "atlas" / "index.html"

TASKS = {
    "transfer_cube": {
        "title": "Transfer cube (lerobot/act_aloha_sim_transfer_cube_human)",
        "primary": "benchmark success",
        "signal_key": "cube_signal_drop",
        "figures": [
            ("cross_attention.png", "Where each cross-attention head looks (original policy)"),
            ("cube_signal.png", "Cube-position signal lost when a head is removed or doubled"),
            ("overlap.png", "Overlap of each head's effect map with the failure map (Spearman, step 0)"),
            ("joint_time_profiles_x0.png", "Which joints and which steps of the chunk move when a head is removed"),
            ("effect_maps_x0_step0.png", "Effect size over the start square when each head is removed (step 0)"),
        ],
    },
    "insertion": {
        "title": "Insertion (lerobot/act_aloha_sim_insertion_human)",
        "primary": "strict handover (peg held in the socket for 25 steps)",
        "signal_key": "object_signal_drop",
        "figures": [
            ("cross_attention.png", "Where each cross-attention head looks (original policy)"),
            ("object_signal.png", "Peg and socket position signal lost when a head is removed or doubled"),
            ("overlap.png", "Overlap of each head's effect map with the strict-failure maps (Spearman, step 0)"),
            ("joint_time_profiles_x0.png", "Which joints and which steps of the chunk move when a head is removed"),
            ("effect_maps_x0_peg_step200.png", "Effect size over the peg grid when each head is removed (step 200)"),
        ],
    },
}


def block_prefix(block):
    if block.startswith("model.encoder.layers."):
        return "enc" + block.split(".")[3]
    if block == "model.decoder.layers.0.multihead_attn":
        return "cross"
    if block == "model.decoder.layers.0.self_attn":
        return "dself"
    raise ValueError(block)


def edit_heads(task):
    e = json.loads((ROOT / "experiments" / task / "freeze" / "edit.json").read_text())
    return {f"{block_prefix(op['block'])}.h{op['head']}": op["scale"] for op in e["ops"] if op["op"] == "scale_head"}, e["candidate"]


def img_b64(path, width=1100):
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=78, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def mean2(d):
    """Insertion stores peg and socket grids separately. The page shows their mean, as the atlas rules do for overlap."""
    if isinstance(d, dict):
        return round((d["peg"] + d["sock"]) / 2, 4)
    return d


def rows(task):
    base = ROOT / "experiments" / task / "stage2"
    heads = json.loads((base / "atlas" / "heads.json").read_text())["heads"]
    arms = json.loads((base / "sweep_summary.json").read_text())["arms"]
    split = json.loads((base / "split_half.json").read_text())
    kept, rechecked = set(split["kept"]), set(split["rechecked"])
    edit, cand = edit_heads(task)
    sig = TASKS[task]["signal_key"]
    out = []
    for h in heads:
        r = {"id": h["id"], "kind": h["kind"], "edit": edit.get(h["id"])}
        for s in ("x0", "x2"):
            r[f"rms0_{s}"] = mean2(h["effect_rms"][s]["0"]) if task == "transfer_cube" else mean2({g: h["effect_rms"][s][g]["0"] for g in ("peg", "sock")})
            r[f"sig_{s}"] = mean2(h[sig][s])
            ov = h["overlap"][s]
            r[f"ov_{s}"] = ov["step0"] if task == "transfer_cube" else ov["step0_strict"]["mean"]
            a = arms[f"{h['id']}.{s}"]
            r[f"gain_{s}"] = round(100 * a["gain"], 1)
            r[f"fb_{s}"] = f"{a['fixed']}/{a['broken']}"
            r[f"p_{s}"] = a["p_mcnemar"]
            arm = f"{h['id']}.{s}"
            r[f"split_{s}"] = ("kept" if arm in kept else "dropped") if arm in rechecked else ""
        j = h["joints"]["x0"]["0"]
        r["joints"] = ", ".join(f"{k} {v:.3f}" for k, v in j.items())
        out.append(r)
    return out, edit, cand, arms["original"]


CSS = """
body{font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;color:#1d1d1f;background:#fafafa}
main{max-width:1200px;margin:0 auto;padding:24px}
h1{font-size:26px;margin:0 0 8px} h2{font-size:21px;margin:36px 0 8px;border-top:1px solid #ddd;padding-top:20px} h3{font-size:17px;margin:20px 0 6px}
p,li{max-width:900px} code{background:#eee;padding:0 3px;border-radius:3px}
.tabs button{font:inherit;padding:6px 14px;margin-right:6px;border:1px solid #bbb;background:#fff;border-radius:6px;cursor:pointer}
.tabs button.on{background:#1d1d1f;color:#fff;border-color:#1d1d1f}
table{border-collapse:collapse;font-size:13px;background:#fff;width:100%}
th,td{padding:4px 6px;border-bottom:1px solid #e5e5e5;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:#f0f0f0;cursor:pointer;user-select:none} th.l,td.l{text-align:left}
tr.edit td{background:#fff6d6} td.pos{color:#0a6b2e} td.neg{color:#a3261b}
.tag{display:inline-block;font-size:11px;padding:0 5px;border-radius:4px;background:#1d1d1f;color:#fff;margin-left:4px}
.wrap{overflow-x:auto;max-height:70vh;border:1px solid #ddd}
figure{margin:16px 0} figure img{max-width:100%;border:1px solid #ddd;background:#fff} figcaption{font-size:13px;color:#555}
.note{font-size:13px;color:#555} input{font:inherit;padding:4px 8px;margin:8px 0}
"""

JS = """
function show(t){document.querySelectorAll('section.task').forEach(s=>s.style.display=s.id===t?'block':'none');
 document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('on',b.dataset.t===t));}
function sortBy(th){const tb=th.closest('table').tBodies[0],i=[...th.parentNode.children].indexOf(th);
 const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
 const rs=[...tb.rows];rs.sort((a,b)=>{const x=a.cells[i].dataset.v??a.cells[i].textContent,y=b.cells[i].dataset.v??b.cells[i].textContent;
 const nx=parseFloat(x),ny=parseFloat(y);const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:x.localeCompare(y);return asc?c:-c;});
 rs.forEach(r=>tb.appendChild(r));}
function filt(inp){const q=inp.value.toLowerCase();inp.parentNode.querySelectorAll('tbody tr').forEach(r=>
 r.style.display=r.textContent.toLowerCase().includes(q)?'':'none');}
"""


def fmt_p(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


def cell(v, cls="", sign=False):
    if v is None or v == "":
        return '<td data-v=""></td>'
    c = cls
    if sign and isinstance(v, (int, float)):
        c += " pos" if v > 0 else " neg" if v < 0 else ""
        txt = f"{v:+.1f}"
    else:
        txt = html.escape(str(v))
    dv = f' data-v="{v}"' if isinstance(v, (int, float)) else ""
    return f'<td class="{c.strip()}"{dv}>{txt}</td>'


def table(task):
    rs, edit, cand, orig = rows(task)
    head = [("head", "l"), ("block kind", "l"), ("open-loop effect, x0", ""), ("open-loop effect, x2", ""),
            ("signal lost, x0", ""), ("signal lost, x2", ""), ("failure-map overlap, x0", ""), ("failure-map overlap, x2", ""),
            ("closed-loop gain x0 (pts)", ""), ("fixed/broken x0", ""), ("p x0", ""), ("split half x0", ""),
            ("closed-loop gain x2 (pts)", ""), ("fixed/broken x2", ""), ("p x2", ""), ("split half x2", ""),
            ("joints moved most when removed (step 0)", "l")]
    th = "".join(f'<th class="{c}" onclick="sortBy(this)">{html.escape(n)}</th>' for n, c in head)
    body = []
    for r in rs:
        tag = f'<span class="tag">in edit, x{r["edit"]:g}</span>' if r["edit"] is not None else ""
        tds = [f'<td class="l" data-v="{r["id"]}">{r["id"]}{tag}</td>', cell(r["kind"], "l"),
               cell(r["rms0_x0"]), cell(r["rms0_x2"]), cell(r["sig_x0"]), cell(r["sig_x2"]), cell(r["ov_x0"]), cell(r["ov_x2"]),
               cell(r["gain_x0"], sign=True), cell(r["fb_x0"]), f'<td data-v="{r["p_x0"]}">{fmt_p(r["p_x0"])}</td>', cell(r["split_x0"]),
               cell(r["gain_x2"], sign=True), cell(r["fb_x2"]), f'<td data-v="{r["p_x2"]}">{fmt_p(r["p_x2"])}</td>', cell(r["split_x2"]),
               cell(r["joints"], "l")]
        body.append(f'<tr class="{"edit" if r["edit"] is not None else ""}">{"".join(tds)}</tr>')
    base_rate = orig.get("success", orig.get("strict"))
    edit_txt = ", ".join(f"{k} x{v:g}" for k, v in edit.items())
    return (f'<p>Frozen edit for this task: <b>{cand}</b>, {html.escape(edit_txt)} (highlighted rows). The original policy\'s primary score on '
            f'design seeds 20000 to 20499 is {100 * base_rate:.1f}% ({TASKS[task]["primary"]}).</p>'
            f'<input placeholder="filter rows, e.g. cross or enc3" oninput="filt(this)">'
            f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'), len(rs)


def section(task):
    t = TASKS[task]
    tbl, n = table(task)
    figs = []
    for fn, cap in t["figures"]:
        p = ROOT / "experiments" / task / "stage2" / "atlas" / fn
        if p.exists():
            figs.append(f'<figure><img alt="{html.escape(cap)}" src="data:image/jpeg;base64,{img_b64(p)}">'
                        f'<figcaption>{html.escape(cap)}. Source: <code>experiments/{task}/stage2/atlas/{fn}</code></figcaption></figure>')
    assert n == 48, (task, n)
    return (f'<section class="task" id="{task}"><h2>{html.escape(t["title"])}</h2>{tbl}'
            f'<h3>Figures</h3>{"".join(figs)}</section>')


def main():
    intro = """
<h1>ACT head atlas: what each of the 48 attention heads does</h1>
<p>Two public ACT checkpoints from LeRobot, one per ALOHA simulation task. Each has 48 attention heads that run at inference:
4 encoder self-attention layers of 8 heads (<code>enc0</code> to <code>enc3</code>, mixing camera, joint and latent tokens), the decoder's
cross-attention (<code>cross</code>, where each action step reads the scene) and the decoder's self-attention (<code>dself</code>, whose
output is constant, so scaling it acts as a fixed internal offset). A head is edited by scaling its 64 columns of the block's
<code>out_proj</code> weight: x0 removes it, x2 doubles it.</p>
<p><b>How to read the columns.</b></p>
<ul>
<li><b>Open-loop effect</b>: RMS change of the 100-step action chunk (normalized action units, all 14 joints) when the head is scaled,
averaged over the recorded step-0 frames of the 41 x 41 start grid. No rollout: recorded states are fed to the edited policy.
For insertion the peg and socket grids are averaged.</li>
<li><b>Signal lost</b>: 1 minus the variance of the edited step-0 chunk over the grid poses, divided by the original's. Positive means
the head carries object-position signal that is lost without it (cube for transfer cube, mean of peg and socket for insertion).</li>
<li><b>Failure-map overlap</b>: Spearman correlation over the grid of a head's step-0 effect map with the smoothed failure map of the
original policy. It was the pre-registered way to pick 5 heads for a recheck.</li>
<li><b>Closed-loop gain</b>: change of the primary score, in points, over 500 design seeds (20000 to 20499) when that single head is
scaled, with seeds fixed/broken and the exact McNemar p. This is a screen of 96 arms; the p values are not corrected and several
large ones will be noise. <b>Split half</b>: the 15 arms rechecked on seeds 20500 to 20999 and whether the sign held.</li>
</ul>
<p class="note">Everything here comes from design seeds and recorded grid states, which the agent used to choose its edits. None of it is
test evidence. The held-out results are in the referee's verdicts, <code>experiments/&lt;task&gt;/test/VERDICT.md</code>. The atlas numbers
also did not explain the edits. On transfer cube the head the written hypothesis relied on (enc3.h4) loses almost no cube signal when
removed (-0.001), and on both tasks the pre-set tracking numbers did not move as predicted (see each task's <code>report/report.md</code>). Built by
<code>python publish/build_atlas.py</code> from committed files only.</p>
<div class="tabs"><button data-t="transfer_cube" onclick="show('transfer_cube')">Transfer cube</button>
<button data-t="insertion" onclick="show('insertion')">Insertion</button></div>
"""
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>ACT head atlas</title><style>{CSS}</style></head><body><main>{intro}'
            f'{section("transfer_cube")}{section("insertion")}</main><script>{JS}show("transfer_cube");</script></body></html>')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
