"""Calibrate one question from any labeled JSONL: rows need a raw top-probability and a 0/1 outcome.
Writes a CalibratedQuestion bound to --fingerprint (use jev_eval.fingerprint.question_fingerprint)."""
import argparse, json
from pathlib import Path
from jev_eval.calibration import calibrate_question, threshold_for_accuracy

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", required=True); ap.add_argument("--score-key", required=True); ap.add_argument("--outcome-key", required=True)
ap.add_argument("--name", required=True); ap.add_argument("--fingerprint", required=True)
ap.add_argument("--method", default="isotonic", choices=["isotonic", "platt_logit"])
ap.add_argument("--target-acc", type=float, default=0.95); ap.add_argument("--out", default=None)
a = ap.parse_args()
rows = [json.loads(l) for l in Path(a.jsonl).read_text().splitlines() if l.strip()]
raw = [r[a.score_key] for r in rows]; y = [int(r[a.outcome_key]) for r in rows]
cq = calibrate_question(raw, y, a.name, a.fingerprint, method=a.method)
thr = threshold_for_accuracy(cq.predict(raw), y, a.target_acc)
out = a.out or f"results/calibrator_{a.name}_{a.fingerprint}.json"
Path(out).parent.mkdir(exist_ok=True); cq.save(out)
print(json.dumps({"report": cq.report, "threshold": thr, "saved": out}, indent=1))
