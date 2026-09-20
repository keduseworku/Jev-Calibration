"""Demo #4 eval: reflex arc on seeded traps, clean vs injected arms. --mock runs offline."""
import argparse, json, random
from pathlib import Path
import numpy as np
from jev_eval.calibration import auroc
from jev_eval.client import JevRunner, MockJev, noisy_oracle
from jev_eval.report import evaluate_questions, per_question_arrays, print_table
from jev_eval.tasks import reflex as T

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--p-injected", type=float, default=0.4)
ap.add_argument("--mock", action="store_true"); ap.add_argument("--model", default=None)
ap.add_argument("--cache", default="data/eval_reflex.jsonl"); ap.add_argument("--out", default="results/eval_reflex.json")
a = ap.parse_args()

items = T.generate(a.n, seed=a.seed, p_injected=a.p_injected)
qs = T.questions()
factory = (lambda: MockJev(noisy_oracle(T.truth_lookup(items), seed=a.seed))) if a.mock else None
runner = JevRunner(a.cache if not a.mock else "data/mock_reflex.jsonl", model=a.model, client_factory=factory)
records = runner.run(items, qs)
rnd = lambda q, seed: (lambda it, r=random.Random(seed): r.random() < 0.5)
res = evaluate_questions(items, records, qs, a.model, rnd, seed=a.seed)

# Guardrail-specific numbers: catch rate on blockable, false-block rate on benign, split by arm.
present = [it for it in items if it["id"] in records]
arms = {}
for arm in ("clean", "injected"):
    sub = [it for it in present if it["arm"] == arm]
    ids, raw, pred, correct = per_question_arrays(sub, records, "should_block")
    truth = np.array([it["truth"]["should_block"] for it in sub]); pred = np.array(pred, bool)
    p_block = np.array([records[i]["answers"]["should_block"]["noul"] for i in ids])
    arms[arm] = {"n": len(sub),
                 "catch_rate": float((pred & truth).sum() / max(1, truth.sum())),
                 "false_block_rate": float((pred & ~truth).sum() / max(1, (~truth).sum())),
                 "auroc_should_block": auroc(p_block, truth.astype(int))}
res["guardrail_by_arm"] = arms
res["config"] = vars(a)
Path(a.out).parent.mkdir(exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
print_table(res)
for arm, r in arms.items():
    print(f"{arm:9s} n={r['n']:4d} catch={r['catch_rate']:.3f} false_block={r['false_block_rate']:.3f} auroc={r['auroc_should_block']}")
print("wrote", a.out)
