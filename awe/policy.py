"""The LeRobot ACT policy at batch size 1, with closed-form weight edits applied on top of the base weights.

Batch size 1 always, so a seed's result does not depend on which other seeds run beside it.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

# Import order matters. triton and torch carry their own LLVM, and loading them after Mesa's LLVM (the software
# renderer) crashes the process, so everything that brings LLVM is imported here, before awe.env touches EGL.
try:
    import triton  # noqa: F401
except Exception:
    pass
import torch
import torch._dynamo  # noqa: F401
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors

from . import edits
from .arms import make_spec
from .paths import checkpoint_dir


def torch_setup():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True  # what lerobot-eval sets
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)


class Policy:
    def __init__(self, task: str, device: str = "cuda", ckpt=None):
        torch_setup()
        self.device = device
        self.ckpt = str(ckpt or checkpoint_dir(task))
        self.policy = ACTPolicy.from_pretrained(self.ckpt).to(device).eval()
        self.policy.config.device = device
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self.ckpt,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.base = {k: v.detach().clone() for k, v in self.policy.state_dict().items()}
        self.spec = None
        self.touched = set()
        self.set_arm(make_spec())

    # ---- arms -----------------------------------------------------------------------------
    def set_arm(self, spec: dict):
        """Restore the base weights, then apply the arm's weight ops."""
        if self.spec == spec:
            return
        sd = self.policy.state_dict()  # references to the live tensors
        with torch.no_grad():
            for k in self.touched:
                sd[k].copy_(self.base[k])
            keys = sorted({k for op in spec["ops"] for k in edits.touched(op)})
            work = {k: self.base[k].clone() for k in keys}
            # ops may read untouched tensors (the latent shift reads a weight), so give them the base values
            view = dict(self.base)
            view.update(work)
            for op in spec["ops"]:
                edits.apply_op(view, op)
            for k in keys:
                sd[k].copy_(view[k])
        self.touched = set(keys)
        self.spec = spec
        cfg = self.policy.config
        cfg.n_action_steps = int(spec["n_action_steps"])
        coeff = spec.get("temporal_ensemble_coeff")
        cfg.temporal_ensemble_coeff = coeff
        if coeff is not None:
            cfg.n_action_steps = 1
            self.policy.temporal_ensembler = ACTTemporalEnsembler(coeff, cfg.chunk_size)
        self.policy.reset()

    # ---- inference --------------------------------------------------------------------------
    @torch.inference_mode()
    def _batch(self, image, agent_pos):
        obs = {"pixels": {"top": np.ascontiguousarray(image)[None]}, "agent_pos": np.asarray(agent_pos)[None]}
        b = preprocess_observation(obs)
        b["task"] = [""]
        return self.pre(b)

    @torch.inference_mode()
    def chunk(self, image, agent_pos) -> np.ndarray:
        """Predicted actions as float32 array (chunk_size, 14), unnormalized. Always the full chunk."""
        a = self.policy.predict_action_chunk(self._batch(image, agent_pos))[0]  # (chunk, 14)
        return self.post(a).cpu().numpy()

    @torch.inference_mode()
    def ensemble_action(self, image, agent_pos) -> np.ndarray:
        act = self.policy.select_action(self._batch(image, agent_pos))  # ensembler state inside
        return self.post(act).cpu().numpy()[0]

    def reset(self):
        self.policy.reset()
