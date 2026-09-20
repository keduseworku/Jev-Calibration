# Eval scaffold (jev_eval)

Built on the upstream `jev_calibration` package (calibrators, metrics) from AnthusAI/Jev-Calibration.
Read `PREREGISTRATION.md` first. Everything runs offline with `--mock` (a noisy oracle standing in for Jev).

```bash
.venv/bin/pytest -q                                        # upstream + scaffold tests, no API key
.venv/bin/python scripts/eval_extraction.py --mock --n 300   # demo #3, offline dry run
.venv/bin/python scripts/eval_reflex.py --mock --n 300       # demo #4, offline dry run
# real runs: put TYPESAFE_API_KEY in .env, drop --mock, answers are cached in data/eval_*.jsonl
.venv/bin/python scripts/eval_calibrate_question.py --jsonl my.jsonl --score-key p --outcome-key y \
    --name should_block --fingerprint <fp>                  # calibrate one question from any labels
```

Modules: `client` (cached runner keyed by question fingerprint, offline mock), `fingerprint`,
`calibration` (per-question isotonic/Platt bundle bound to a fingerprint, accuracy-targeted thresholds),
`bracket` (random / Jev / oracle with bootstrap CIs), `phrasing` (paraphrase panel with McNemar),
`counterfactual` (sampling of rejected items, Wilson CIs), `report`, and `tasks/` (extraction, reflex,
tree_prune, rts).
