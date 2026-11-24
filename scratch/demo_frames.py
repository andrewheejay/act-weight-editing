"""First frame of each of the 50 training episodes, from the downloaded dataset. Run with `python -I`.
argv: dataset dir, output npz. Reads the parquet and the mp4 as data only."""
import sys
import numpy as np
import pandas as pd
import av

root, out = sys.argv[1], sys.argv[2]
ep = pd.read_parquet(root + "/meta/episodes/chunk-000/file-000.parquet")
print(ep.shape, [c for c in ep.columns if "top" in c or c in ("episode_index", "length", "dataset_from_index")][:12])
c_from = "videos/observation.images.top/from_timestamp"
t0 = ep[c_from].values if c_from in ep else None
idx = ep["videos/observation.images.top/file_index"].values if "videos/observation.images.top/file_index" in ep else None
print(t0[:3] if t0 is not None else None, idx[:3] if idx is not None else None, ep["length"].values[:5])
want = {int(round(t * 50)): k for k, t in enumerate(t0)}
con = av.open(root + "/videos/observation.images.top/chunk-000/file-000.mp4")
st = con.streams.video[0]
st.thread_type = "AUTO"
frames, n = {}, 0
for fr in con.decode(st):
    if n in want:
        frames[want[n]] = fr.to_ndarray(format="rgb24")
    n += 1
print("decoded", n, "frames, kept", len(frames))
np.savez_compressed(out, first=np.stack([frames[k] for k in range(len(t0))]))
