"""Load the sentiment dataset copied from AnthusAI/Classification-with-Confidence."""
import hashlib
from pathlib import Path

FILES = {
    "strong_positive.txt": ("positive", "strong"),
    "strong_negative.txt": ("negative", "strong"),
    "medium_positive.txt": ("positive", "medium"),
    "medium_negative.txt": ("negative", "medium"),
    "weak_positive.txt": ("positive", "weak"),
    "weak_negative.txt": ("negative", "weak"),
    "neutral_positive.txt": ("positive", "neutral"),
    "neutral_negative.txt": ("negative", "neutral"),
}


def example_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_dataset(dataset_dir: str | Path = "dataset") -> list[dict]:
    """Return rows {id, text, expected, strength, category}; duplicate texts are dropped."""
    rows, seen = [], set()
    for filename, (expected, strength) in FILES.items():
        for line in (Path(dataset_dir) / filename).read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            eid = example_id(text)
            if eid in seen:
                continue
            seen.add(eid)
            rows.append({"id": eid, "text": text, "expected": expected,
                         "strength": strength, "category": f"{strength}_{expected}"})
    return rows
