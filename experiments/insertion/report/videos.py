"""Side-by-side videos of the original (left) and the agent's edit (right) on the same DESIGN seeds, for the insertion report. Design seeds only, the
test seeds are not mine to run. Adapted from experiments/transfer_cube/report/videos.py (2026-11-26, session 28, after the test was read).

python experiments/insertion/report/videos.py      -> report/videos/*.mp4, report/videos/index.json, report/video_runs.jsonl

Seeds come from the earlier design results (stage1/design_default.jsonl against r1 in stage3/combo_AB.jsonl.gz) with a fixed rule and rng 0, on the
primary score (strict success): 2 fixed class-3 failures (met, no strict insertion), 2 fixed other failures, 2 broken, 1 both succeed, 1 both fail.
Every rollout is checked against the trajectory hash of the earlier row. Agent budget, 16 rollouts.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from awe import pool  # noqa: E402
from awe.arms import from_manifest, make_spec  # noqa: E402

T = ROOT / "experiments" / "insertion"
OUT = T / "report" / "videos"


def rows(p):
    with (gzip.open if str(p).endswith(".gz") else open)(p, "rt") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def pick():
    o = {r["seed"]: r for r in rows(T / "stage1" / "design_default.jsonl")}
    e = {r["seed"]: r for r in rows(T / "stage3" / "combo_AB.jsonl.gz") if r["arm"] == "r1"}
    rng = np.random.default_rng(0)
    S = sorted(set(o) & set(e))
    fx = [s for s in S if e[s]["strict_success"] and not o[s]["strict_success"]]
    sets = {"fixed_class3_failure": ([s for s in fx if o[s]["failure_class"] == "3"], 2),
            "fixed_other_failure": ([s for s in fx if o[s]["failure_class"] != "3"], 2),
            "broken_by_edit": ([s for s in S if o[s]["strict_success"] and not e[s]["strict_success"]], 2),
            "both_succeed": ([s for s in S if o[s]["strict_success"] and e[s]["strict_success"]], 1),
            "both_fail": ([s for s in S if not o[s]["strict_success"] and not e[s]["strict_success"]], 1)}
    out = {}
    for k, (ss, n) in sets.items():
        for i in sorted(rng.choice(len(ss), n, replace=False)):
            out[ss[i]] = k
    return out, {k: len(v[0]) for k, v in sets.items()}, o, e


def main():
    import av
    import cv2

    OUT.mkdir(parents=True, exist_ok=True)
    kind, counts, o, e = pick()
    man = json.loads((T / "freeze" / "test_manifest.json").read_text())
    arms = {"original": make_spec("original"), "agent_edit": from_manifest(man, "agent_edit")}
    frames, rec = {}, {}

    def cb(r, f):
        frames[(r["arm"], r["seed"])] = f
        rec[(r["arm"], r["seed"])] = r

    pool.run("insertion", [(s, seed, None) for s in arms.values() for seed in sorted(kind)], T / "report" / "video_runs.jsonl",
             budget="agent", purpose="report videos", frames_cb=cb, n_frames=100)
    index = []
    for seed in sorted(kind):
        a, b = frames[("original", seed)], frames[("agent_edit", seed)]
        n = min(len(a), len(b))
        name = f"{kind[seed]}_{seed}.mp4"
        with av.open(str(OUT / name), "w") as c:
            st = c.add_stream("libx264", rate=12)
            st.width, st.height, st.pix_fmt = 640, 260, "yuv420p"
            for t in range(n):
                im = np.full((260, 640, 3), 250, np.uint8)
                for k, f in enumerate((a[t], b[t])):
                    im[20:260, 320 * k : 320 * (k + 1)] = cv2.resize(f, (320, 240))
                cv2.putText(im, "original", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
                cv2.putText(im, f"edit  (design seed {seed}, step {t * 4})", (328, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
                for p in st.encode(av.VideoFrame.from_ndarray(im, format="rgb24")):
                    c.mux(p)
            for p in st.encode():
                c.mux(p)
        index.append({"seed": seed, "kind": kind[seed], "file": name, "original_class": rec[("original", seed)]["failure_class"], "edit_class": rec[("agent_edit", seed)]["failure_class"], "original_strict": rec[("original", seed)]["strict_success"], "edit_strict": rec[("agent_edit", seed)]["strict_success"],
                      "hash_matches_earlier": rec[("original", seed)]["traj_hash"] == o[seed]["traj_hash"] and rec[("agent_edit", seed)]["traj_hash"] == e[seed]["traj_hash"]})
    (OUT / "index.json").write_text(json.dumps({"rule": "rng 0, see videos.py", "pool_sizes": counts, "videos": index}, indent=1) + "\n")
    print(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
