# Jev-Calibration

Does Jev's probability (and `confidence`) mean what it says? This repo measures it, fits **Platt scaling** on a
calibration split, and evaluates on an untouched test split.

It is the follow-up to [Making Decisions Instead of Generating Text](https://anth.us/blog/making-decisions-instead-of-generating-text/)
and reuses the sentiment dataset and calibration methods from
[AnthusAI/Classification-with-Confidence](https://github.com/AnthusAI/Classification-with-Confidence).

## Design
- **Data:** the 10,000-line sentiment dataset (strong/medium/weak/neutral x positive/negative), copied from
  Classification-with-Confidence. Exact within-category repeats are dropped -> **8,801 unique examples**
  (no text appears with conflicting labels). The `neutral_*` files carry arbitrary "domain bias" labels, so accuracy
  there is expected to be near chance; the per-strength breakdown shows it separately.
- **Split:** stratified 60% calibration / 40% test, seed 42, persisted in `data/splits.json`.
- **One request, two questions per example:** a `Noul` ("is the sentiment positive?") and a `Choice`
  (`positive`/`negative`). Raw answers are stored in `data/jev_raw.jsonl`, so analysis never re-calls the API.
- **Signals calibrated:** `noul_top` = max(p, 1-p), `choice_top` = probability of the chosen option,
  `choice_confidence` = Jev's `confidence` statistic (target: P(prediction correct)); and `noul_p_pos` as a binary
  class probability (target: label is positive).
- **Calibrators:** Platt on logit(score) (main), Platt on the raw score (as in the earlier repo), isotonic.
  All are fit on the calibration split only.
- **Metrics:** ECE (equal-width and equal-mass), MCE, Brier, log loss, accuracy-vs-coverage.

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env            # add TYPESAFE_API_KEY
.venv/bin/python scripts/smoke.py                 # 5 examples, prints raw responses
.venv/bin/python scripts/run_jev.py --limit 500   # pilot (resumable)
.venv/bin/python scripts/run_jev.py               # everything
.venv/bin/python scripts/analyze.py               # results/metrics.json + images/
.venv/bin/pytest                                  # no API key needed
```

## Results
_Pending the Jev run._
