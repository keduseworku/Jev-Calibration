"""Resumable async runner: one Jev request per example, raw answers appended to a JSONL file."""
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from .questions import QUESTIONS


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def serialize(response) -> dict:
    ans = response.answers
    out = {}
    for name, a in ans.items():
        out[name] = a.model_dump() if hasattr(a, "model_dump") else dict(a)
    return out


async def run(rows: list[dict], out_path: str | Path = "data/jev_raw.jsonl",
              concurrency: int = 8, limit: int | None = None) -> int:
    load_dotenv()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = done_ids(out_path)
    todo = [r for r in rows if r["id"] not in seen][:limit]
    sem, lock, count = asyncio.Semaphore(concurrency), asyncio.Lock(), 0

    async with AsyncTypeSafeClient(retry=RetryPolicy(max_retries=6, backoff_max=30.0)) as client:
        async def one(row):
            nonlocal count
            async with sem:
                t0 = time.perf_counter()
                try:
                    resp = await client.system_one(state={"text": row["text"]}, questions=QUESTIONS)
                except Exception as e:  # keep going; failed ids are retried on the next run
                    print(f"FAILED {row['id']}: {type(e).__name__}: {e}")
                    return
                rec = {"id": row["id"], "model": resp.model, "latency_s": time.perf_counter() - t0,
                       "usage": resp.usage.model_dump() if resp.usage else None,
                       "answers": serialize(resp)}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"{count}/{len(todo)}")

        await asyncio.gather(*(one(r) for r in todo))
    return count
