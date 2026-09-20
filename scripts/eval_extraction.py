"""Demo #3 eval: synthetic-inverse extraction. --mock runs offline with a noisy oracle."""
import argparse, json
from pathlib import Path
from jev_eval.client import JevRunner, MockJev, noisy_oracle
from jev_eval.report import evaluate_questions, print_table
from jev_eval.tasks import extraction as T

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--omit", type=float, default=0.25); ap.add_argument("--distractors", type=int, default=3)
ap.add_argument("--mock", action="store_true"); ap.add_argument("--model", default=None)
ap.add_argument("--target-acc", type=float, default=0.95)
ap.add_argument("--cache", default="data/eval_extraction.jsonl"); ap.add_argument("--out", default="results/eval_extraction.json")
a = ap.parse_args()

items = T.generate(a.n, seed=a.seed, p_omit=a.omit, n_distractors=a.distractors)
qs = T.questions()
factory = (lambda: MockJev(noisy_oracle(T.truth_lookup(items), seed=a.seed))) if a.mock else None
runner = JevRunner(a.cache if not a.mock else "data/mock_extraction.jsonl", model=a.model, client_factory=factory)
records = runner.run(items, qs)
res = evaluate_questions(items, records, qs, a.model, T.random_decider, target_acc=a.target_acc, seed=a.seed)
res["config"] = vars(a)
Path(a.out).parent.mkdir(exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
print_table(res); print("wrote", a.out)
