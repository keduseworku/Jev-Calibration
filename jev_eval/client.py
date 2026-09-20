"""Cached, resumable Jev runner over arbitrary (state, questions) items, plus an offline mock.

Cache records are keyed by (item id, question fingerprint). Re-running with changed wording or a
different model re-queries instead of reusing stale answers. Mirrors jev_calibration.jev_client
but takes any state shape and any question set.
"""
import asyncio
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .fingerprint import plain, question_fingerprint


def serialize_answers(answers: Mapping) -> dict:
    return {name: plain(a) for name, a in answers.items()}


class JevRunner:
    """items: [{"id": str, "state": <JSON>}]. questions: one mapping shared by all items."""

    def __init__(self, cache_path: str | Path, model: str | None = None, concurrency: int = 8,
                 client_factory: Callable | None = None):
        self.cache_path = Path(cache_path)
        self.model = model
        self.concurrency = concurrency
        self.client_factory = client_factory or self._real_client

    @staticmethod
    def _real_client():
        from dotenv import load_dotenv
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
        load_dotenv()
        return AsyncTypeSafeClient(retry=RetryPolicy(max_retries=6, backoff_max=30.0))

    def load_cache(self, fp: str) -> dict[str, dict]:
        out = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r["fp"] == fp:
                        out[r["id"]] = r
        return out

    async def run_async(self, items: list[dict], questions: Mapping, limit: int | None = None) -> dict[str, dict]:
        fp = question_fingerprint(questions, self.model)
        have = self.load_cache(fp)
        todo = [it for it in items if it["id"] not in have][:limit]
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        sem, lock = asyncio.Semaphore(self.concurrency), asyncio.Lock()

        async with self.client_factory() as client:
            async def one(it):
                async with sem:
                    t0 = time.perf_counter()
                    try:
                        resp = await client.system_one(state=it["state"], questions=questions, model=self.model)
                    except Exception as e:  # failed ids are retried on the next run
                        print(f"FAILED {it['id']}: {type(e).__name__}: {e}")
                        return
                    usage = resp.usage
                    rec = {"id": it["id"], "fp": fp, "model": getattr(resp, "model", self.model),
                           "latency_s": time.perf_counter() - t0,
                           "usage": plain(usage) if usage is not None else None,
                           "answers": serialize_answers(resp.answers)}
                async with lock:
                    with self.cache_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    have[it["id"]] = rec

            await asyncio.gather(*(one(it) for it in todo))
        return {it["id"]: have[it["id"]] for it in items if it["id"] in have}

    def run(self, items, questions, limit=None) -> dict[str, dict]:
        return asyncio.run(self.run_async(items, questions, limit=limit))


class MockJev:
    """Offline stand-in for AsyncTypeSafeClient.

    answer_fn(state, questions) -> serialized answers. Use noisy_oracle(...) to build one from item
    truth so tests and dry runs exercise the whole pipeline without an API key.
    """

    def __init__(self, answer_fn: Callable[[dict, Mapping], dict], model: str = "mock-jev"):
        self.answer_fn, self.model = answer_fn, model

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def system_one(self, state, questions, model=None, **_):
        return SimpleNamespace(answers=self.answer_fn(state, questions), model=model or self.model, usage=None)


def noisy_oracle(truth_by_state: Callable[[dict], dict], noise: float = 0.25, seed: int = 0,
                 miss_rate: float = 0.1) -> Callable:
    """Build a MockJev answer_fn that knows the truth but reports it with noise and occasional misses.

    truth_by_state(state) -> {question_name: truth}; truth is bool (noul), option key (choice), or
    level index (score). Probabilities are rounded to 2 decimals like the real model.
    """
    rng = np.random.default_rng(seed)

    def _peaked(keys, true_key):
        n = len(keys)
        base = rng.dirichlet(np.ones(n) * 0.3)
        if rng.random() >= miss_rate:  # usually put most mass on the truth
            base = base * noise
            base[keys.index(true_key)] += 1 - noise
        p = np.round(base / base.sum(), 2)
        p[np.argmax(p)] += round(1 - p.sum(), 2)
        return {k: float(v) for k, v in zip(keys, p)}

    def answer_fn(state, questions):
        truth = truth_by_state(state)
        out = {}
        for name, q in questions.items():
            q = plain(q)
            t = truth.get(name)
            if q["type"] == "noul":
                p = (1 - noise * rng.random()) if t else noise * rng.random()
                if rng.random() < miss_rate:
                    p = 1 - p
                out[name] = {"type": "noul", "noul": round(float(p), 2)}
            elif q["type"] == "choice":
                keys = list(q["criteria"].keys())
                probs = _peaked(keys, t if t in keys else keys[-1])
                top = max(probs, key=probs.get)
                out[name] = {"type": "choice", "choice": top, "probabilities": probs,
                             "confidence": round(2 * probs[top] - 1, 2)}
            elif q["type"] == "score":
                levels = [str(i) for i in range(len(q["criteria"]))]
                probs = _peaked(levels, str(t if t is not None else 0))
                score = sum(int(k) * v for k, v in probs.items())
                out[name] = {"type": "score", "score": round(score, 2), "probabilities": probs,
                             "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                             "confidence": round(max(probs.values()), 2)}
        return out

    return answer_fn
