# Getting Calibrated Confidence from Jev

> **TL;DR**: Jev returns a probability with every answer, but a probability isn't automatically an *accuracy*. On 8,801 labeled sentiment examples, Jev's raw probabilities were systematically off (expected calibration error 0.117). A two-parameter Platt curve cut that to 0.052. **Isotonic regression cut it to 0.008**, because the miscalibration wasn't sigmoid-shaped. Once calibrated, the way you phrase the question matters much less than you'd expect.

This is the follow-up to [Making Decisions Instead of Generating Text](https://anth.us/blog/making-decisions-instead-of-generating-text/), which ended with "test calibration against a held-out set." It reuses the dataset and calibration ideas from [Classification-with-Confidence](https://github.com/AnthusAI/Classification-with-Confidence), where we squeezed confidence out of a local LLM's token log-probabilities. Here the model hands us the probabilities directly.

## What Jev gives you

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) accepts some text ("state") and a set of typed questions, and returns a typed answer to each:

| Question | Answer | Probability information |
|---|---|---|
| **Noul** | yes/no | `noul` = P(yes) |
| **Choice** | one of your options | `probabilities` per option, plus `confidence` |
| **Score** | a rating on your rubric | `probabilities` per level, an expected `score`, plus `confidence` |

Three things we observed on `jev-1.13.0`:

1. **Probabilities are rounded to two decimals**, so many answers tie at exactly 0.00 or 1.00. That matters for calibration.
2. **For a two-option Choice, `confidence` is `2·p_top − 1`** (to within rounding). It carries no information beyond the top probability. Its documentation calls it "a statistic computed from the probability distribution", so this isn't a surprise, but it means you can't get a second opinion from it on binary questions.
3. **It's cheap and fast.** About 330 input tokens per request (~$0.12 for the whole dataset at the published $42 per billion input tokens; output tokens are free), median latency 0.24 seconds, and no rate-limiting at concurrency 8.

## Do these numbers mean what they say?

If Jev says 80% positive, are 80% of such texts positive? We checked with the same 10,000-line sentiment set as the earlier project (strong / medium / weak / neutral × positive / negative). Exact repeats were dropped, leaving **8,801 unique examples**, split 60/40 (stratified, seed 42) into a **calibration** set (5,280) and an untouched **test** set (3,521).

Each example goes to Jev as **one request with several questions**. The first one is a Noul: *"Is the overall sentiment of this text positive?"*

Accuracy is very uneven across the tiers, and that is most of the story:

| Tier | Accuracy (`noul_pos`, test) |
|---|---|
| strong | 1.000 |
| medium | 0.997 |
| weak | 0.668 |
| neutral | 0.494 |

The `neutral_*` files carry deliberately arbitrary labels (a "domain bias" experiment from the original project), so ~50% there is the expected result, not a Jev failure. We report results with and without that tier.

A **reliability diagram** groups predictions by predicted probability and plots the fraction that were actually positive. A calibrated model sits on the diagonal. Jev's raw probabilities don't:

![Reliability, raw vs Platt vs isotonic](images/variants/reliability_grid.png)

Left column: raw. Jev is under-confident about positives in the middle of the range and its curve is bumpy. On the Choice variants it's worse: at a raw P(positive) of about 0.1, roughly 46% of the labels are positive.

## Two ways to fix it

Both learn a mapping from Jev's raw probability to observed frequency, fit on the calibration set only.

- **Platt scaling** fits a logistic curve: two parameters, smooth, works with little data.
- **Isotonic regression** learns any *monotone* (never-decreasing) step function: more flexible, needs more data.

![What each method does](images/variants/mapping_noul_pos.png)

The dots are what actually happened. There's a plateau: for raw scores from about 0.33 to 0.62, the true positive rate sits near 65%. A sigmoid can't bend that way. It splits the difference and stays off the data at both ends. Isotonic follows the data.

Held-out results for the `noul_pos` question (test split, lower is better):

| | ECE | MCE | Brier | Log loss |
|---|---|---|---|---|
| Raw Jev | 0.117 | 0.265 | 0.162 | 0.467 |
| + Platt | 0.052 | 0.134 | 0.143 | 0.420 |
| + Isotonic | **0.008** | **0.068** | **0.138** | **0.401** |

ECE is the average gap between predicted probability and observed frequency. Neither method changes *which* answers rank above others (Platt can't, and isotonic only adds ties), so accuracy and AUC barely move (0.872 → 0.874). Calibration makes the number trustworthy; it doesn't make Jev smarter.

This held for every question setup we tried:

![ECE by setup and method](images/variants/ece_all.png)

Raw ECE ranged from 0.064 to 0.160 across setups, Platt from 0.026 to 0.115, and **isotonic from 0.006 to 0.018**.

## Does the way you ask matter?

We sent a panel of alternative setups for the same question, all in one request per example (`jev_calibration/variants.py`):

- `noul_pos` / `noul_neg`: ask "is it positive?" vs "is it negative?"
- `noul_favorable`: different wording
- `choice2`, `choice2_swapped`, `choice2_described`: a two-way Choice, reversed option order, and options with descriptions
- `choice3`: adds a `neutral` option
- `score5` / `score5_mean`: a five-level rubric, using the level probabilities or the expected score
- `ensemble_noul` / `ensemble_all`: averages across framings

What we found:

- **Raw calibration depends a lot on the setup, isotonic-calibrated quality much less.** After isotonic, Brier scores land in a narrow 0.135–0.158 band; a floor set largely by the arbitrary neutral labels.
- **Ranking quality (AUC) differs a little:** `noul_favorable` 0.885 and `ensemble_noul` 0.883 lead; `noul_neg` 0.827 and `score5` 0.832 trail. The expected score from a Score (`score5_mean`, 0.874) beat treating its levels as a Choice (0.832).
- **Averaging framings gives the best *raw* calibration** (`ensemble_noul` ECE 0.064) but it still benefits from calibration.
- **Option order and request composition barely matter.** Reversing the Choice options changed accuracy by under a point. The same Noul asked inside a different request differed from the earlier answer by 0.0074 on average (max 0.19).
- **Full results:** [`results/variants.json`](results/variants.json). Charts: [`images/variants/`](images/variants/).

![Accuracy by tier](images/variants/accuracy_by_strength.png)

## Using calibrated confidence

The reason to calibrate is a routing policy: auto-accept what Jev is sure about, send the rest to a person. Ranking by calibrated confidence and accepting from the top:

![Accuracy vs coverage](images/variants/accuracy_vs_coverage_no_neutral.png)

On the non-neutral tiers, roughly the top half of decisions are accepted at 98–100% accuracy; beyond that accuracy decays. To keep accuracy at or above 95%, you could auto-accept about 65–74% of non-neutral decisions depending on the setup (about 49–52% if arbitrary-label neutral examples are included). The exact figures are in `results/variants.json`; they are sensitive to ties from Jev's rounding, so treat differences of a few points as noise.

Because the mapping is now in units of accuracy, "accept above 90%" means what it says, and you can set different thresholds per question and per consequence.

## Limitations

- **One dataset, one model version, one run.** Sentiment on a constructed dataset says little about call-QA rubrics. Re-check on your own data.
- **We looked at the test split for many variants.** Picking a "best" setup from eleven candidates on the same test data is optimistic, and we haven't computed confidence intervals. The gap between Platt and isotonic is large enough to trust; the ordering among setups isn't.
- **Isotonic needs data.** We had 5,280 calibration examples. We haven't measured how it degrades with 200 or 500, which is where Platt's two parameters may win. That's the obvious next experiment.
- **Rounded probabilities and ties** limit resolution, and isotonic can only reproduce ~100 distinct levels.
- **Calibration is per question, per model version.** Refit when the question wording, data source, or model changes.
- **`confidence` for two-option questions adds nothing** over the probabilities. It may differ for Choices with more options; we didn't isolate that.

## Run it yourself

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env                       # add TYPESAFE_API_KEY (console.typesafe.ai)
.venv/bin/python scripts/smoke.py          # 5 examples, prints raw responses
.venv/bin/python scripts/run_jev.py        # base run (resumable): data/jev_raw.jsonl
.venv/bin/python scripts/analyze.py        # top-label calibration: results/metrics.json
.venv/bin/python scripts/run_variants.py   # variant panel (resumable): data/jev_variants.jsonl
.venv/bin/python scripts/analyze_variants.py   # results/variants.json + images/variants/
.venv/bin/pytest                           # no API key needed
```

Raw Jev answers are committed under `data/`, so the analysis scripts reproduce every number and chart here without an API key.

## References

- [Making Decisions Instead of Generating Text](https://anth.us/blog/making-decisions-instead-of-generating-text/)
- [How to Make a Text Classifier Using an LLM](https://anth.us/blog/fine-tuned-classification-with-confidence/) and its [repository](https://github.com/AnthusAI/Classification-with-Confidence)
- [TypeSafe: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [Jev docs](https://docs.typesafe.ai/introduction), [Confidence](https://docs.typesafe.ai/confidence)
- [Platt scaling](https://en.wikipedia.org/wiki/Platt_scaling), [isotonic regression](https://scikit-learn.org/stable/modules/isotonic.html), [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
