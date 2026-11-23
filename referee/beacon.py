"""Public randomness for the test seeds.

The test seeds do not exist until after the freeze is public. They are derived from a round of the
drand quicknet beacon that is emitted at least ten minutes after the freeze commitment is published.
Nobody, including the agent and the people running it, can know that round's value in advance, and
anybody can check it afterwards against the network's public key.

Round arithmetic and the verification scheme follow the drand documentation. quicknet uses unchained
BLS signatures on G1 with the public key on G2 (scheme bls-unchained-g1-rfc9380).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request

DST_G1 = b"BLS_SIG_BLS12381G1_XMD:SHA-256_SSWU_RO_NUL_"

# A real quicknet round, kept as an offline test vector.
TEST_VECTOR = {
    "round": 42,
    "signature": "95a9f9f5b231b7714de1553105d8ffdf3dcda24cfdb1e689319bccf79a9c8ce430a91b811fbfaf763900bc998b5d686a",
    "randomness": "8ada64bae5c6c0f5540a6a13af56e663240edfbd2c76ac6a8f27671eb7259ce3",
}


def time_of_round(rnd: int, genesis: int, period: int) -> int:
    """Unix time at which a round is emitted. Round 1 is emitted at the genesis time."""
    return genesis + (rnd - 1) * period


def first_round_at_or_after(t: float, genesis: int, period: int) -> int:
    if t <= genesis:
        return 1
    return int(math.ceil((t - genesis) / period)) + 1


def target_round(published_at: float, bcfg: dict) -> int:
    """The round the seeds come from. The first round at least delay_seconds after publication."""
    return first_round_at_or_after(published_at + bcfg["delay_seconds"], bcfg["genesis_time"], bcfg["period"])


def randomness_of(signature_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(signature_hex)).hexdigest()


def bls_available() -> bool:
    try:
        import py_ecc  # noqa: F401

        return True
    except ImportError:
        return False


def verify_signature(rnd: int, signature_hex: str, public_key_hex: str) -> bool:
    """Check a quicknet beacon against the network public key. Needs the py_ecc package."""
    from py_ecc.bls.hash_to_curve import hash_to_G1
    from py_ecc.bls.point_compression import decompress_G1, decompress_G2
    from py_ecc.optimized_bls12_381 import G2, curve_order, is_inf, multiply, pairing

    pkb = bytes.fromhex(public_key_hex)
    sigb = bytes.fromhex(signature_hex)
    if len(pkb) != 96 or len(sigb) != 48:
        return False
    try:
        pk = decompress_G2((int.from_bytes(pkb[:48], "big"), int.from_bytes(pkb[48:], "big")))
        sig = decompress_G1(int.from_bytes(sigb, "big"))
    except Exception:
        return False
    if is_inf(sig) or not is_inf(multiply(sig, curve_order)):
        return False
    msg = hashlib.sha256(int(rnd).to_bytes(8, "big")).digest()
    hm = hash_to_G1(msg, DST_G1, hashlib.sha256)
    return pairing(G2, sig) == pairing(pk, hm)


def _get_json(url: str, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "awe-referee/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_round(rnd: int, bcfg: dict, *, require_bls: bool = True, log=print) -> dict:
    """Fetch one round from the public relays and check it.

    With py_ecc installed, one relay is enough, because a valid signature proves itself.
    Without it, two relays must agree and the record says the signature was not verified here.
    """
    if require_bls and not bls_available():
        raise RuntimeError("py_ecc is not installed, so the beacon cannot be verified. pip install py_ecc")
    seen: dict[str, list[str]] = {}
    errors = []
    for relay in bcfg["relays"]:
        url = f"{relay}/{bcfg['chain_hash']}/public/{rnd}"
        try:
            d = _get_json(url)
        except Exception as e:  # a relay being down is normal
            errors.append(f"{relay}: {type(e).__name__} {e}")
            continue
        sig = str(d.get("signature", "")).lower()
        if int(d.get("round", -1)) != rnd or not sig:
            errors.append(f"{relay}: unexpected reply")
            continue
        seen.setdefault(sig, []).append(relay)
        if bls_available():
            if verify_signature(rnd, sig, bcfg["public_key"]):
                return {
                    "round": rnd,
                    "signature": sig,
                    "randomness": randomness_of(sig),
                    "relays": seen[sig],
                    "bls_verified": True,
                }
            errors.append(f"{relay}: signature failed verification")
            seen.pop(sig, None)
        elif len(seen[sig]) >= 2:
            return {
                "round": rnd,
                "signature": sig,
                "randomness": randomness_of(sig),
                "relays": seen[sig],
                "bls_verified": False,
            }
    raise RuntimeError(f"could not get a verified beacon for round {rnd}: " + "; ".join(errors))


def wait_for_round(rnd: int, bcfg: dict, *, log=print, heartbeat=None, poll: float = 20.0) -> dict:
    """Sleep until the round's time has passed, then fetch it, retrying while relays catch up."""
    due = time_of_round(rnd, bcfg["genesis_time"], bcfg["period"])
    while True:
        left = due - time.time()
        if left <= 0:
            break
        log(f"beacon round {rnd} is due in {int(left)} s")
        if heartbeat:
            heartbeat()
        time.sleep(min(poll, max(1.0, left)))
    last = None
    for attempt in range(60):
        try:
            return fetch_round(rnd, bcfg, log=log)
        except RuntimeError as e:
            last = e
            if "py_ecc" in str(e):
                raise
            log(f"beacon not available yet ({e}). retrying")
            if heartbeat:
                heartbeat()
            time.sleep(poll)
    raise RuntimeError(f"gave up waiting for beacon round {rnd}: {last}")


def seed_block(randomness_hex: str, task: str, commit: str, tcfg: dict) -> dict:
    """The test seeds for one freeze. A block of consecutive seeds with a start nobody could predict."""
    material = "|".join([tcfg["domain"], task, commit, randomness_hex]).encode()
    digest = hashlib.sha256(material).digest()
    start = tcfg["base"] + int.from_bytes(digest[:8], "big") % tcfg["span"]
    return {
        "start": start,
        "count": tcfg["count"],
        "end": start + tcfg["count"] - 1,
        "derivation": "start = base + int(sha256(domain|task|commit|randomness)[:8]) mod span",
        "digest": digest.hex(),
    }


def selfcheck() -> list[str]:
    """Offline checks that need no network. Returns a list of problems, empty when all is well."""
    from .common import load_config

    b = load_config()["beacon"]
    problems = []
    if randomness_of(TEST_VECTOR["signature"]) != TEST_VECTOR["randomness"]:
        problems.append("test vector randomness does not match its signature")
    if first_round_at_or_after(1776643200, b["genesis_time"], b["period"]) != 27946612:
        problems.append("round arithmetic disagrees with the published quicknet example")
    if time_of_round(1, b["genesis_time"], b["period"]) != b["genesis_time"]:
        problems.append("round 1 should be emitted at the genesis time")
    if bls_available():
        if not verify_signature(TEST_VECTOR["round"], TEST_VECTOR["signature"], b["public_key"]):
            problems.append("the real round 42 beacon failed verification")
        if verify_signature(TEST_VECTOR["round"] + 1, TEST_VECTOR["signature"], b["public_key"]):
            problems.append("a beacon verified for the wrong round")
    else:
        problems.append("py_ecc is not installed, so signatures cannot be verified")
    return problems
