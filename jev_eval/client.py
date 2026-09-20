"""Cached Jev runner over arbitrary (state, questions) items, plus an offline mock.

Records are keyed by (hash of the state, fingerprint of questions + model id), so a changed
document, word, option or model re-queries. The served model must equal the requested one or the
run stops. Mirrors jev_calibration.jev_client.run, which is fixed to that study's rows and questions.
"""
import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .fingerprint import plain, question_fingerprint


def real_client():
    from dotenv import load_dotenv
    from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
    load_dotenv()
    return AsyncTypeSafeClient(retry=RetryPolicy(max_retries=6, backoff_max=30.0))


def state_key(state) -> str:
    return hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest()[:16]


def run(items, questions, model, cache_path=None, client_factory=real_client, concurrency=8) -> dict[str, dict]:
    """items: [{"id", "state"}]. Returns item id -> record for every item that answered.
    cache_path=None means no persistence (mock runs)."""
    fp = question_fingerprint(questions, model)
    cache = Path(cache_path) if cache_path else None
    have, bad = {}, 0
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        for line in cache.read_text().splitlines() if cache.exists() else []:
            try:
                r = json.loads(line)
                if r["fp"] == fp:
                    have[r["key"]] = r
            except (ValueError, KeyError):  # truncated write or a record from an older cache format
                bad += 1
        if bad:
            print(f"skipped {bad} unreadable line(s) in {cache}; those items are re-queried")
    keys = {it["id"]: state_key(it["state"]) for it in items}
    todo = list({keys[it["id"]]: it for it in items if keys[it["id"]] not in have}.values())  # one call per distinct state

    async def go():
        sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
        async with client_factory() as client:
            async def one(it):
                async with sem:
                    t0 = time.perf_counter()
                    try:
                        resp = await client.system_one(state=it["state"], questions=questions, model=model)
                    except Exception as e:  # failed items are retried on the next run
                        print(f"FAILED {it['id']}: {type(e).__name__}: {e}")
                        return
                if resp.model != model:
                    raise RuntimeError(f"asked for model {model!r}, server answered with {resp.model!r}; use the exact id")
                rec = {"key": keys[it["id"]], "id": it["id"], "fp": fp, "model": resp.model,
                       "latency_s": time.perf_counter() - t0, "usage": plain(resp.usage) if resp.usage else None,
                       "answers": {k: plain(a) for k, a in resp.answers.items()}}
                async with lock:
                    if cache:
                        with cache.open("a") as f:
                            f.write(json.dumps(rec) + "\n")
                    have[rec["key"]] = rec
            await asyncio.gather(*(one(it) for it in todo))

    asyncio.run(go())
    return {it["id"]: have[keys[it["id"]]] for it in items if keys[it["id"]] in have}


class MockJev:
    """Offline stand-in for AsyncTypeSafeClient. answer_fn(state, questions) -> serialized answers."""

    def __init__(self, answer_fn):
        self.answer_fn = answer_fn

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def system_one(self, state, questions, model=None):
        return SimpleNamespace(answers=self.answer_fn(state, questions), model=model, usage=None)


def truth_by_state(items):
    """state -> truth for mocks, which must not see item ids."""
    table = {state_key(it["state"]): it["truth"] for it in items}
    return lambda state: table[state_key(state)]


def noisy_oracle(lookup, noise=0.25, miss_rate=0.1, seed=0):
    """Mock answers that know the truth but report it with noise and occasional misses, rounded to
    2 decimals like the real model so ties occur. Noul and Choice only."""
    rng = np.random.default_rng(seed)

    def answer(state, questions):
        truth, out = lookup(state), {}
        for name, q in questions.items():
            q, t = plain(q), truth[name]
            if q["type"] == "score":
                raise NotImplementedError("noisy_oracle mocks Noul and Choice only")
            p_true = 1 - noise * rng.random()
            if rng.random() < miss_rate:
                p_true = 1 - p_true
            if q["type"] == "noul":
                out[name] = {"type": "noul", "noul": round(float(p_true if t else 1 - p_true), 2)}
            else:
                others = [k for k in q["criteria"] if k != t]
                spread = rng.dirichlet(np.ones(len(others))) * (1 - p_true)
                probs = {t: round(float(p_true), 2), **{k: round(float(v), 2) for k, v in zip(others, spread)}}
                top = max(probs, key=probs.get)
                out[name] = {"type": "choice", "choice": top, "probabilities": probs, "confidence": 2 * probs[top] - 1}
        return out

    return answer
