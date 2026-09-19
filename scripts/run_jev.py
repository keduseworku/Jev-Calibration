"""Run Jev over the dataset (resumable). Usage: run_jev.py [--limit N] [--concurrency N]"""
import argparse, asyncio
from jev_calibration.dataset import load_dataset
from jev_calibration.jev_client import run
from jev_calibration.splits import load_or_make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int); ap.add_argument("--concurrency", type=int, default=8)
a = ap.parse_args()
rows = load_dataset(); load_or_make_splits(rows)
print("done:", asyncio.run(run(rows, limit=a.limit, concurrency=a.concurrency)))
