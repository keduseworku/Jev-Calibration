"""Raw Jev answers -> one row per example with every calibratable signal."""
import json
from pathlib import Path

import pandas as pd


def load_features(rows: list[dict], splits: dict[str, str], raw_path="data/jev_raw.jsonl") -> pd.DataFrame:
    meta = {r["id"]: r for r in rows}
    recs = []
    for line in Path(raw_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m, a = meta[r["id"]], r["answers"]
        p_pos = a["positive"]["noul"]
        ch = a["sentiment"]
        top = ch["probabilities"][ch["choice"]]
        recs.append({
            "id": r["id"], "split": splits[r["id"]], "strength": m["strength"], "category": m["category"],
            "expected": m["expected"], "latency_s": r["latency_s"],
            # Noul as a binary class probability
            "noul_p_pos": p_pos, "noul_pred": "positive" if p_pos >= 0.5 else "negative",
            "noul_top": max(p_pos, 1 - p_pos),
            # Choice
            "choice_pred": ch["choice"], "choice_top": top, "choice_confidence": ch["confidence"],
        })
    df = pd.DataFrame(recs).drop_duplicates("id")
    df["y_pos"] = (df.expected == "positive").astype(int)
    df["noul_correct"] = (df.noul_pred == df.expected).astype(int)
    df["choice_correct"] = (df.choice_pred == df.expected).astype(int)
    return df
