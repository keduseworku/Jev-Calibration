"""Run Jev over the dataset (resumable). Usage: run_jev.py [--limit N] [--concurrency N]"""
import argparse, asyncio
from jev_calibration.dataset import load_dataset
from jev_calibration.jev_client import run
from jev_calibration.splits import load_or_make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int); ap.add_argument("--concurrency", type=int, default=8)
a = ap.parse_args()
import random
rows = load_dataset(); load_or_make_splits(rows)
random.Random(7).shuffle(rows)  # fixed order so --limit gives a random sample, not just strong_positive
print("done:", asyncio.run(run(rows, limit=a.limit, concurrency=a.concurrency)))
