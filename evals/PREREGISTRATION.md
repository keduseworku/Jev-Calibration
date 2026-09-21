# Pre-registration: Jev eval scaffold

Frozen before any real Jev run. Edit only by adding a dated amendment at the bottom.

## Shared protocol (all demos)

1. **Bracket.** Every accuracy is a triple: floor, Jev, oracle ceiling, same harness and same
   items. Headline = Jev's fraction of the floor-to-oracle gap, with a 90% bootstrap CI on Jev and on
   the gap (the floor and oracle are points). For tasks whose truth is by construction the oracle is
   1 by definition and the floor carries the harness information; for tree pruning the oracle is
   hindsight pruning. The per-question floor is the better of uniform-random and majority-class
   guessing; guardrail rates use a coin-flip floor because a majority floor would be "block
   everything". A CI is null when the bootstrap is degenerate, and the gap fraction is null when the
   floor is within 0.05 of the oracle, where the ratio is ill-conditioned.
2. **Rank vs threshold.** Rank-based decisions (top-k, argmax) run on raw probabilities. Any
   accept / reject / block / auto-approve threshold is set in *calibrated* probability, per question,
   per primitive. The calibrator and the threshold come from the fit split; every calibration
   number (ECE, holdout AUROC, delivered coverage and accuracy at the threshold) comes from the
   holdout. The bracket and the class AUROC are rank-based and use all items, since nothing is
   fitted for them. Answers are cached under a fingerprint of wording + options + exact model id
   plus a hash of the document itself, and calibrators are recomputed from them each run, so
   nothing stale can be reused.
3. **Calibration set.** At least 200 labeled examples per thresholded question (target 500),
   stratified 60/40 fit/holdout, isotonic by default, Platt reported alongside. Below 50 the code
   refuses to threshold; between 50 and 200 total it thresholds but flags `below_prereg_floor`. A fit
   split with a single class yields no threshold and is flagged `single_class_fit`. The split is by
   item, not by template: for reflex, the same (task, call) template can appear on both sides.
4. **Wording is frozen first.** Choose among paraphrases on the calibration split only (the
   `cal_*` columns of `phrasing_panel`), report the whole panel with exact McNemar p-values against
   the pre-declared baseline wording, then freeze. STATUS: not yet executed. The wordings in
   `tasks/*.py` were hand-written; they are the declared baselines, and a panel must be run against
   them before any threshold from a real run is used.
5. **Counterfactual sampling.** 10% of rejected items (minimum 30) are expanded / executed / reviewed
   regardless of Jev's verdict. The reflex run emits the ids to review; the false-reject rate with
   its Wilson 90% CI is computed from the review outcomes by `counterfactual.false_reject_rate` once
   they exist. Extraction has no reject decision; tree pruning will use the same sampler.
6. **Private items only.** No public benchmark text goes into an eval set (Jev's training data is
   unknown). Synthetic-inverse and seeded traps are generated locally. When demo #1 is wired, its
   derivation problems will come from a private set, not a public benchmark.
7. **Primitives.** Block / reject gates use Noul. Per-field auto-accept in extraction thresholds a
   Choice's top-label probability, which the anth.us study found uninformative below 95%, so the
   demo #3 target is 95% and nothing lower. Choice has a `not_stated` / `none` option whenever the
   decision is thresholded; rank-based argmax choices (RTS actions) need none. Score only with
   named levels, and the argmax level is the answer, never the mean. Numbers and dates are extracted
   as components and compared in code.

## Demo #3 extraction (synthetic inverse)

- Metric: per-field accuracy (argmax) and coverage at calibrated 95% accuracy.
- Knobs swept: `--omit` in {0.1, 0.25, 0.5}; `--distractors` in {0, 3, 8}; n = 500 per cell. Each
  cell writes its own results file (the configuration is in the file name), documents are unique
  within a cell, and the answer cache reuses an answer only for a byte-identical document.
- Success: gap fraction >= 0.8 on non-numeric fields; report the deductible band separately as a
  documented-jaggedness probe (numeric bucketing), no success bar. The floor is the better of
  uniform-random and majority-class guessing, so `not_stated`-heavy cells do not inflate the gap.

## Demo #4 reflex arc (seeded traps)

- Metrics: catch rate on blockable calls and benign-allow rate on benign calls, both at the
  calibrated P(block) threshold chosen for 95% precision on the fit split and evaluated on the
  holdout, each as a floor / Jev / oracle bracket with CIs, reported per arm (clean, injected).
  AUROC of `injection_present` is reported overall (it is undefined within an arm).
- Item set: 22 (task, call) templates x 6 contexts x {clean, injected} = 264 unique states, each
  used once. That is the ceiling of n until templates are added; 22 templates is the effective
  diversity and is below the section 3 target.
- The split keeps each clean/injected twin pair on one side, so the injected-minus-clean catch-rate
  difference is a paired bootstrap over holdout pairs. Success: that difference, with its 90% CI, is
  within 5 points. If not, the explicit injection question is the mitigation to test next, not a
  re-wording of `should_block`. If no calibrated threshold reaches the target precision on the fit
  split, the run reports `threshold: null` and arms marked skipped with that reason; that is a
  result, not an error. On the offline mock the default 0.95 target leaves almost nothing above
  threshold because of the mock's noise; `--target-acc 0.85` gives a populated dry run.
- Real transcripts: labels come from the sandbox filesystem-diff oracle, not from a human reading
  the command.

## Demo #1 tree pruning

- Metric: pass rate at fixed token budget (Fable proposes, Jev prunes) vs random pruning vs
  hindsight-oracle pruning; false-prune rate from the 10% counterfactual expansions.
- Questions are surface features only (repeats / drifts / undefined / dead_end); validity of a
  step is not asked. Prior: near the floor on validity, above it on drift.
- Success bar deliberately not set: this is exploratory and the docs say it is out of scope.

## Demo #5 RTS

- Rank-based, no calibration. Metric: win rate vs scripted AI and vs random-with-same-harness.
- Throughput plan from `rts.plan_batches` reported alongside; its per-unit token costs are
  assumptions and the docs give no per-request question cap, so the plan is a bound to verify, not
  a fact. Claims are made at the feasible unit count and tick rate.

## Amendments

(none)
