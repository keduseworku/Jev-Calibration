"""Shared reporting: turn (items, records, question set) into per-question raw score / correctness arrays,
run calibration + threshold + bracket, and return a JSON-able dict."""
import numpy as np

from .bracket import accuracy_metric, bracket
from .calibration import calibrate_question, predicted, raw_score, threshold_for_accuracy
from .fingerprint import question_fingerprint


def per_question_arrays(items, records, qname):
    ids = [it["id"] for it in items if it["id"] in records]
    raw = np.array([raw_score(records[i]["answers"][qname]) for i in ids])
    pred = [predicted(records[i]["answers"][qname]) for i in ids]
    truth = [it["truth"][qname] for it in items if it["id"] in records]
    correct = np.array([p == t for p, t in zip(pred, truth)], int)
    return ids, raw, pred, correct


def evaluate_questions(items, records, questions, model, random_decider, target_acc=0.95, n_boot=500, seed=0) -> dict:
    fp = question_fingerprint(questions, model)
    out = {"fingerprint": fp, "n_items": len(records), "questions": {}}
    present = [it for it in items if it["id"] in records]
    for qname in questions:
        ids, raw, pred, correct = per_question_arrays(present, records, qname)
        pred_by_id = dict(zip(ids, pred))
        row = {"n": int(len(ids)), "accuracy_raw_argmax": float(correct.mean())}
        try:
            cq = calibrate_question(raw, correct, qname, fp, seed=seed)
            row["calibration"] = cq.report
            row["threshold"] = threshold_for_accuracy(cq.predict(raw), correct, target_acc)
        except ValueError as e:
            row["calibration"] = {"skipped": str(e)}
        deciders = {"random": random_decider(qname, seed), "jev": lambda it, q=qname: pred_by_id[it["id"]],
                    "oracle": lambda it, q=qname: it["truth"][q]}
        row["bracket"] = bracket(present, deciders, lambda d, its, q=qname: accuracy_metric(
            d, [{"truth": it["truth"][q]} for it in its]), n_boot=n_boot, seed=seed)
        out["questions"][qname] = row
    return out


def print_table(result: dict):
    print(f"fingerprint {result['fingerprint']}  n={result['n_items']}")
    print(f"{'question':28s} {'acc':>6s} {'rand':>6s} {'orac':>6s} {'gap%':>6s} {'ECEraw':>7s} {'ECEiso':>7s} {'cov@t':>6s}")
    for q, r in result["questions"].items():
        b = r["bracket"]["metrics"]
        g = r["bracket"]["gap_fraction"]["jev"]["point"]
        cal = r.get("calibration", {})
        e_raw = cal.get("raw", {}).get("ece"); e_iso = cal.get("isotonic", {}).get("ece")
        cov = r.get("threshold", {}).get("coverage")
        f = lambda v, w=6, d=3: (f"{v:{w}.{d}f}" if isinstance(v, (int, float)) and v is not None else " " * (w - 1) + "-")
        print(f"{q:28s} {f(r['accuracy_raw_argmax'])} {f(b['random']['point'])} {f(b['oracle']['point'])} "
              f"{f(g)} {f(e_raw, 7)} {f(e_iso, 7)} {f(cov)}")
