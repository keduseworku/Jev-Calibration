"""Per-question calibration + bracket over (items, records, questions), and the shared task CLI."""
import argparse
import json
from pathlib import Path

import numpy as np

from .bracket import bracket, floor_scores
from .calibration import auroc, calibrate_question, predicted, raw_score
from .client import MockJev, noisy_oracle, run, truth_by_state
from .fingerprint import plain, question_fingerprint


def n_options(q) -> int:
    q = plain(q)
    return 2 if q["type"] == "noul" else len(q["criteria"])


def evaluate_questions(items, records, questions, model, target_acc=0.95, seed=0) -> dict:
    present = [it for it in items if it["id"] in records]
    out = {"fingerprint": question_fingerprint(questions, model), "n_items": len(present), "questions": {}}
    if not present:
        out["skipped"] = "no item answered"
        return out
    for name, q in questions.items():
        ans = [records[it["id"]]["answers"][name] for it in present]
        truth = [it["truth"][name] for it in present]
        raw = np.array([raw_score(a) for a in ans])
        correct = np.array([predicted(a) == t for a, t in zip(ans, truth)], float)
        row = {"bracket": bracket(floor_scores(truth, n_options(q)), correct, np.ones(len(present)), seed=seed)}
        if plain(q)["type"] == "noul":  # class AUROC of P(yes) vs the label: rank-based, no split needed
            row["auroc_class"] = auroc([a["noul"] for a in ans], truth)
        try:
            row["calibration"] = calibrate_question(raw, correct, target_acc, seed=seed)[1]
        except ValueError as e:
            row["calibration"] = {"skipped": str(e)}
        out["questions"][name] = row
    return out


def print_table(res: dict):
    print(f"fingerprint {res['fingerprint']}  n={res['n_items']}")
    print(f"{'question':26s}    jev  floor    gap ECEraw ECEiso  cov@t  acc@t")
    for q, r in res["questions"].items():
        b, c = r["bracket"], r["calibration"]
        t = c.get("threshold", {})
        cells = [b["jev"]["point"], b["floor"]["point"], b["gap"]["point"], c.get("raw", {}).get("ece"),
                 c.get("isotonic", {}).get("ece"), t.get("holdout_coverage"), t.get("holdout_accuracy")]
        print(f"{q:26s} " + " ".join("     -" if v is None else f"{v:6.3f}" for v in cells))
    if "extra" in res:
        print("extra:", json.dumps({k: v for k, v in res["extra"].items() if not k.endswith("_ids")}))


def run_task(task):
    """generate -> run (cached or mock) -> evaluate -> write. The task module exposes NAME, generate(n, seed,
    **knobs), questions(), optional KNOBS (argparse specs), DEFAULT_N and extra_metrics(items, records, target_acc, seed)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=getattr(task, "DEFAULT_N", 300))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--mock", action="store_true", help="offline noisy oracle, nothing cached")
    ap.add_argument("--target-acc", type=float, default=0.95)
    ap.add_argument("--out", default=None, help="default: results/eval/<task>-<config>.json")
    knob_specs = getattr(task, "KNOBS", {})
    for flag, spec in knob_specs.items():
        ap.add_argument(flag, **spec)
    a = ap.parse_args()
    knobs = {d: getattr(a, d) for d in (f.lstrip("-").replace("-", "_") for f in knob_specs)}
    slug = (f"n{a.n}-s{a.seed}-t{a.target_acc}" + "".join(f"-{k}{v}" for k, v in sorted(knobs.items()))
            + ("-mock" if a.mock else f"-{a.model}"))
    out = Path(a.out or f"results/eval/{task.NAME}-{slug}.json")
    items, qs = task.generate(a.n, seed=a.seed, **knobs), task.questions()
    if a.mock:
        records = run(items, qs, a.model, client_factory=lambda: MockJev(noisy_oracle(truth_by_state(items), seed=a.seed)))
    else:
        records = run(items, qs, a.model, cache_path=f"data/eval_{task.NAME}.jsonl")
    res = evaluate_questions(items, records, qs, a.model, a.target_acc, a.seed)
    if hasattr(task, "extra_metrics"):
        try:
            res["extra"] = task.extra_metrics(items, records, a.target_acc, a.seed)
        except ValueError as e:
            res["extra"] = {"skipped": str(e)}
    res["config"] = vars(a)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, allow_nan=False))
    print_table(res)
    print("wrote", out)
