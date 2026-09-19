"""Stratified calibration/test split, fixed seed, persisted so it never drifts."""
import json
from pathlib import Path

from sklearn.model_selection import train_test_split


def make_splits(rows: list[dict], calib_fraction: float = 0.6, seed: int = 42) -> dict[str, str]:
    ids = [r["id"] for r in rows]
    strata = [r["category"] for r in rows]
    calib, test = train_test_split(ids, train_size=calib_fraction, random_state=seed, stratify=strata)
    return {**{i: "calibration" for i in calib}, **{i: "test" for i in test}}


def load_or_make_splits(rows: list[dict], path: str | Path = "data/splits.json") -> dict[str, str]:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    splits = make_splits(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=0, sort_keys=True))
    return splits
