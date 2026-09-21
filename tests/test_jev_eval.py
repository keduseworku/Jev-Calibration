import json
import re

import numpy as np
import pytest
from typesafe_sdk import Noul

from jev_calibration import metrics
from jev_eval.bracket import bracket, difference, floor_scores
from jev_eval.calibration import calibrate_question, predicted, threshold_for_accuracy
from jev_eval.client import MockJev, noisy_oracle, run, truth_by_state
from jev_eval.counterfactual import false_reject_rate, sample_rejected
from jev_eval.fingerprint import question_fingerprint
from jev_eval.phrasing import mcnemar_exact, phrasing_panel
from jev_eval.report import evaluate_questions
from jev_eval.tasks import extraction, reflex, rts, tree_prune

MODEL = "jev-test"


def test_fingerprint_changes_with_wording_and_model():
    q1, q2 = {"a": Noul(instructions="Is it positive?")}, {"a": Noul(instructions="Is it favorable?")}
    assert question_fingerprint(q1, MODEL) != question_fingerprint(q2, MODEL)
    assert question_fingerprint(q1, "jev-1.13.0") != question_fingerprint(q1, "jev-1.14.0")
    assert question_fingerprint(q1, MODEL) == question_fingerprint({"a": {"type": "noul", "instructions": "Is it positive?", "criteria": None}}, MODEL)
    with pytest.raises(ValueError):
        question_fingerprint(q1, None)


def test_run_caches_by_fingerprint_and_checks_served_model(tmp_path):
    items = extraction.generate(20, seed=1)
    calls = {"n": 0}
    oracle = noisy_oracle(truth_by_state(items), seed=1)

    def counting(state, qs):
        calls["n"] += 1
        return oracle(state, qs)

    qs, cache = extraction.questions(), tmp_path / "c.jsonl"
    r1 = run(items, qs, MODEL, cache, client_factory=lambda: MockJev(counting))
    r2 = run(items, qs, MODEL, cache, client_factory=lambda: MockJev(counting))
    assert len(r1) == 20 and r1.keys() == r2.keys() and calls["n"] == 20
    run(items, {**qs, "coverage_type": Noul(instructions="different")}, MODEL, cache, client_factory=lambda: MockJev(counting))
    assert calls["n"] == 40  # new wording -> new fingerprint -> re-queried
    cache.write_text(cache.read_text() + "\n")  # interrupted write leaves a blank line
    changed = extraction.generate(20, seed=1, omit=0.9, distractors=0)  # same ids, different documents
    run(changed, qs, MODEL, cache, client_factory=lambda: MockJev(counting))
    assert calls["n"] == 60  # cache is keyed by state, not by id

    class WrongModel(MockJev):
        async def system_one(self, state, questions, model=None):
            return await super().system_one(state, questions, model="other")
    with pytest.raises(RuntimeError):
        run(items, qs, MODEL, client_factory=lambda: WrongModel(counting))


def _overconfident(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    raw = np.round(np.clip(np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n)) ** 0.5, 0.5, 1), 2)
    return raw, y


def test_calibrate_question_threshold_from_fit_numbers_from_holdout():
    raw, y = _overconfident()
    iso, rep, ho = calibrate_question(raw, y, target_acc=0.9)
    assert rep["isotonic"]["ece"] < rep["raw"]["ece"] and rep["n_holdout"] == len(ho) == 240
    assert rep["below_prereg_floor"] is False and rep["single_class_fit"] is False
    t = rep["threshold"]
    fit = np.setdiff1d(np.arange(len(raw)), ho)
    assert t["threshold"] == threshold_for_accuracy(iso.predict(raw[fit]), y[fit], 0.9)["threshold"]  # chosen on fit
    applied = metrics.coverage_curve(iso.predict(raw[ho]), y[ho], [t["threshold"]])[0]  # as applied, ties included
    assert (applied["coverage"], applied["accuracy"]) == (t["holdout_coverage"], t["holdout_accuracy"])
    assert calibrate_question(raw[:250], y[:250])[1]["below_prereg_floor"] is False  # total, not fit size
    assert calibrate_question(raw[:199], y[:199])[1]["below_prereg_floor"] is True
    groups = np.arange(600) // 2  # twins must land on the same side
    _, _, ho_g = calibrate_question(raw, y, groups=groups)
    assert all(((groups == g).sum() == np.isin(np.where(groups == g)[0], ho_g).sum()) for g in set(groups[ho_g]))
    with pytest.raises(ValueError):
        calibrate_question(raw[:20], y[:20])
    _, rep1, _ = calibrate_question(np.linspace(0.5, 1, 60), np.ones(60))  # no negatives: no threshold, no Platt
    assert rep1["single_class_fit"] and rep1["threshold"]["threshold"] is None and "platt" not in rep1


def test_threshold_for_accuracy_counts_ties():
    conf, correct = np.array([0.96, 0.96, 0.96, 0.7]), np.array([1, 1, 0, 1])
    assert threshold_for_accuracy(conf, correct, 0.95)["threshold"] is None  # prefix [1,1] would pass; ties do not
    assert threshold_for_accuracy(conf, correct, 0.7) == {"threshold": 0.7, "coverage": 1.0, "accuracy": 0.75}  # lowest t that reaches target


def test_bracket_and_floor():
    rng = np.random.default_rng(0)
    truth = list(rng.integers(0, 4, 400))
    jev = np.array([rng.random() < 0.7 for _ in truth], float)
    res = bracket(floor_scores(truth, 4), jev, np.ones(400), n_boot=200)
    assert res["oracle"]["point"] == 1.0 and res["floor"]["point"] >= 0.25
    g = res["gap"]
    assert 0.4 < g["point"] < 0.9 and g["ci90"][0] < g["point"] < g["ci90"][1]
    assert floor_scores(["a"] * 90 + ["b"] * 10, 2).mean() == 0.9  # majority beats random
    tiny = bracket(floor_scores([1] * 10, 2), np.ones(10), np.ones(10), n_boot=50)  # floor meets oracle
    assert tiny["gap"]["point"] is None and tiny["gap"]["ci90"] is None
    near = bracket(np.full(50, 0.99), np.full(50, 0.6), np.ones(50), n_boot=50)  # ill-conditioned, not -39
    assert near["gap"]["point"] is None
    assert bracket([0.5], [1], [1], n_boot=50)["jev"]["ci90"] is None  # singleton: no CI, no crash
    d = difference(np.array([1, 1, 1, 0]), np.array([1, 0, 0, 0]), paired=True, n_boot=100)
    assert d["point"] == 0.5 and d["ci90"][0] <= 0.5 <= d["ci90"][1] and d["paired"]


def test_counterfactual():
    ids = [f"b{i}" for i in range(500)]
    s = sample_rejected(ids, rate=0.1, seed=0)
    assert len(s) == 50 and all(isinstance(x, str) for x in s) and len(set(s)) == 50
    r = false_reject_rate(30, 3)
    assert r["ci90"][0] < 0.1 < r["ci90"][1]
    assert false_reject_rate(0, 0)["rate"] is None


def test_phrasing_panel_reports_both_splits():
    raw, y = _overconfident(800)
    cal = np.arange(800) % 5 != 0
    noisier = np.clip(raw + np.random.default_rng(1).normal(0, 0.2, 800), 0, 1)
    rows = {r["variant"]: r for r in phrasing_panel({"base": raw, "same": list(raw), "noisier": noisier}, y, cal, baseline="base")}
    assert rows["base"]["cal_mcnemar_p_vs_baseline"] is None and rows["same"]["cal_mcnemar_p_vs_baseline"] == 1.0
    assert rows["noisier"]["cal_auroc"] < rows["base"]["cal_auroc"] and "test_ece_isotonic" in rows["base"]
    assert mcnemar_exact([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_extraction_truth_matches_document_and_ids_unique():
    items = extraction.generate(400, seed=3, omit=0.3, distractors=8)
    assert len({it["id"] for it in items}) == 400 and set(extraction.questions()) == set(items[0]["truth"])
    assert len({it["state"]["document"] for it in items}) == 400
    for it in items:  # an unstated field's options never appear as tokens anywhere in the document
        tokens = set(re.findall(r"[a-z0-9]+", it["state"]["document"].lower()))
        for f, opts in extraction.FIELDS.items():
            if it["truth"][f] == "not_stated":
                assert not tokens & {o.lower() for o in opts[:-1]}, (f, it["state"]["document"])
    for it in items:
        doc, t = it["state"]["document"], it["truth"]
        assert (t["coverage_type"] in doc) == (t["coverage_type"] != "not_stated")
        assert (t["effective_month"] == "not_stated") == (t["effective_year"] == "not_stated")
        if t["effective_year"] != "not_stated":
            assert t["effective_year"] in doc and t["effective_month"].capitalize() in doc
        else:
            assert not any(m.capitalize() in doc for m in extraction.MONTHS)
        if t["cancellation_notice_days"] == "not_stated":
            assert not any(f"{d} days" in doc for d in ("10", "30", "60", "90"))
        assert ("flood endorsement" in doc) == t["flood_rider"]
    rate = np.mean([it["truth"]["effective_year"] == "not_stated" for it in items])
    assert 0.2 < rate < 0.4  # the date pair shares one omission draw


def test_reflex_labels_and_uniqueness():
    items = reflex.generate(seed=0)
    assert len(items) == reflex.DEFAULT_N
    assert len({str(it["state"]) for it in items}) == len(items)
    for it in items:
        t = it["truth"]
        assert t["should_block"] == (t["destructive"] or t["out_of_scope"])
        assert t["injection_present"] == any(s in it["state"]["context_excerpt"] for s in reflex.INJECTIONS)
    both = [it for it in items if it["truth"]["destructive"] and it["truth"]["out_of_scope"]]
    assert both  # labels are per template, not per pool
    assert set(reflex.questions()) == set(items[0]["truth"])
    with pytest.raises(ValueError):
        reflex.generate(reflex.DEFAULT_N + 1)


def test_reflex_extra_metrics_offline():
    items = reflex.generate(seed=0)
    records = run(items, reflex.questions(), MODEL, client_factory=lambda: MockJev(noisy_oracle(truth_by_state(items), seed=0)))
    res = evaluate_questions(items, records, reflex.questions(), MODEL)
    assert set(res["questions"]) == set(reflex.questions())
    extra = reflex.extra_metrics(items, records, 0.9, 0)
    assert set(extra["arms"]) == {"clean", "injected"} and len(extra["counterfactual_review_ids"]) >= 30
    assert 0 <= extra["arms"]["clean"]["catch_rate"]["jev"]["point"] <= 1
    d = extra["catch_rate_injected_minus_clean"]
    assert d["paired"] and d["n_pairs"] >= 10 and res["questions"]["injection_present"]["auroc_class"] > 0.5
    raws = [records[it["id"]]["answers"]["should_block"]["noul"] for it in items]
    assert len(set(raws)) < len(raws) / 2  # mock rounds to 2 decimals: ties exist end to end
    assert predicted({"type": "score", "probabilities": {"calm": 0.1, "tense": 0.6, "hostile": 0.3}}) == "tense"
    hard = reflex.extra_metrics(items, records, 0.999, 0)  # near-unreachable target: no crash, no NaN, reasons recorded
    json.dumps(hard, allow_nan=False)
    assert hard["precision_threshold"]["holdout_coverage"] in (None, 0.0) or hard["precision_threshold"]["holdout_coverage"] < 0.1


def test_tree_prune_rank_breaks_ties_at_random():
    br = [{"id": f"b{i}", "step": f"step {i}"} for i in range(6)]
    qs = tree_prune.questions(len(br))
    assert len(qs) == 4 * len(br) and "branches[3].step" in qs["b3_dead_end"].instructions
    assert tree_prune.pack("g", "s", br)["branches"][2] == {"id": "b2", "step": "step 2"}
    answers = {f"b{i}_dead_end": {"type": "noul", "noul": 0.1 if i in (0, 3) else 0.8} for i in range(6)}
    keep, rejected = tree_prune.rank(answers, br, k=2)
    assert set(keep) == {"b0", "b3"} and len(rejected) == 4
    tied = {f"b{i}_dead_end": {"type": "noul", "noul": 0.5} for i in range(6)}
    assert tree_prune.rank(tied, br, 2, seed=1)[0] != tree_prune.rank(tied, br, 2, seed=2)[0]


def test_rts_batch_planner():
    p, q = rts.plan_batches(1000, 5.0), rts.plan_batches(200, 1.0)
    assert not p["feasible"] and p["units_per_request"] >= 300 and q["feasible"]
    with pytest.raises(ValueError):
        rts.plan_batches(10, 0)
    assert len(rts.questions(3)) == 6
