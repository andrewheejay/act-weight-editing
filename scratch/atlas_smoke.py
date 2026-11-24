import time, numpy as np
from awe import render; render.configure(render.chosen("transfer_cube"))
from awe import atlas, arms
from awe.paths import cache_dir
F=atlas.Fwd("transfer_cube"); d=cache_dir("transfer_cube")/"atlas"; meta=np.load(d/"meta.npz"); feats=np.load(d/"feats.npy",mmap_mode="r"); t=F.t
rec=atlas.check_fwd("transfer_cube",F,meta,feats)
for s in atlas.STEPS:
    e=[x for q,x in rec if q==s]; print("step",s,"n",len(e),"max",max(e),"median",np.median(e))
# cached-feature forward against the stock LeRobot forward, for the original and for edited weights
import numpy as np
from lerobot.policies.act.modeling_act import ACTPolicy
m=F.m; B=8
ids=np.arange(0,4244,600)[:B]
with t.inference_mode():
    f=t.tensor(np.array(feats[ids])).cuda(); st=t.tensor(meta["state"][ids]).cuda()
    class BB(t.nn.Module):
        def forward(self,x): return {"feature_map":f}
    bb=m.backbone
    vs=[v for v in atlas.variants() if v[0] in ("enc0.h3.x0","enc2.h1.x2","enc3.h7.sharp2","cross.h2.x0","dself.h4.x2","lat5.-2","lat9.+2","cross.h1.soft0.5")]
    for name,b,h,nm,op in [("original","enc",0,"",None)]+vs:
        F.P.set_arm(arms.make_spec("original"))
        x0,pos=F.embed(f,st); mem,outs=F.enc(x0,pos,keep=True); base=F.dec(mem,pos)
        if op is None: y=base
        else:
            F.P.set_arm(arms.make_spec(name,[op])); sl=atlas.start_layer(b)
            y=F.dec(mem,pos) if sl==9 else (F.dec(F.enc(F.embed(f,st)[0],pos),pos) if b=="latent" else F.dec(F.enc(outs[sl],pos,start=sl),pos))
        m.backbone=BB()
        stock=F.P.policy.predict_action_chunk({"observation.state":st,"observation.images.top":t.zeros(len(ids),3,480,640).cuda()})
        m.backbone=bb
        print(name,"restart vs stock forward max diff %.2e, effect rms %.4f"%(float((y-stock).abs().max()),float((y-base).pow(2).mean().sqrt())))
    F.P.set_arm(arms.make_spec("original"))
    ff=t.tensor(np.array(feats[:48])).cuda(); ss=t.tensor(meta["state"][:48]).cuda()
    t.cuda.synchronize(); t0=time.time(); x0,pos=F.embed(ff,ss); mem=F.enc(x0,pos); y=F.dec(mem,pos); y.sum().item(); print("full fwd of 48",time.time()-t0)
    t0=time.time(); y=F.dec(mem,pos).sum().item(); print("dec only",time.time()-t0)
    w=F.cross_attn(mem,pos); print(w.shape, float(w.sum(-1).min()), float(w.sum(-1).max()))
