"""In-process rollouts for the tests. Each one gets a ledger line, like every other rollout."""
from awe.env import FastAloha
from awe.ledger import Ledger
from awe.policy import Policy
from awe import rollout

_cache = {}


def runner():
    if not _cache:
        _cache["pol"] = Policy("transfer_cube")
        _cache["env"] = FastAloha("transfer_cube")
        _cache["led"] = Ledger("transfer_cube", "setup", "test suite")
    return _cache["env"], _cache["pol"]


def run(seed, spec):
    env, pol = runner()
    _cache["led"].log(spec["id"], seed)
    pol.set_arm(spec)
    arr = rollout.run_episode(env, pol, seed=seed)
    return arr, rollout.summarize(arr, spec["id"], seed)
