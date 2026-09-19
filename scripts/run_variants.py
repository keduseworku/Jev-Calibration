"""Run the variant panel (one request, 8 questions) over the dataset. Resumable."""
import argparse, asyncio, random
from jev_calibration.dataset import load_dataset
from jev_calibration.jev_client import run
from jev_calibration.splits import load_or_make_splits
from jev_calibration.variants import QUESTIONS

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int); ap.add_argument("--concurrency", type=int, default=8)
a = ap.parse_args()
rows = load_dataset(); load_or_make_splits(rows)
random.Random(7).shuffle(rows)
print("done:", asyncio.run(run(rows, out_path="data/jev_variants.jsonl", limit=a.limit,
                               concurrency=a.concurrency, questions=QUESTIONS)))
