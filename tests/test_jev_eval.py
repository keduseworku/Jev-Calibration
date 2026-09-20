import numpy as np
import pytest
from typesafe_sdk import Noul

from jev_eval.bracket import accuracy_metric, bracket
from jev_eval.calibration import CalibratedQuestion, calibrate_question, threshold_for_accuracy
from jev_eval.client import JevRunner, MockJev, noisy_oracle
from jev_eval.counterfactual import expansions_for_precision, false_reject_rate, sample_rejected
from jev_eval.fingerprint import question_fingerprint
from jev_eval.phrasing import mcnemar_exact, phrasing_panel
from jev_eval.tasks import extraction, reflex, rts, tree_prune


def test_fingerprint_changes_with_wording_and_model():
    q1 = {"a": Noul(instructions="Is it positive?")}
    q2 = {"a": Noul(instructions="Is it favorable?")}
    assert question_fingerprint(q1, None) != question_fingerprint(q2, None)
    assert question_fingerprint(q1, "jev-1.13.0") != question_fingerprint(q1, "jev-1.14.0")
    assert question_fingerprint(q1, None) == question_fingerprint({"a": {"type": "noul", "instructions": "Is it positive?", "criteria": None}}, None)


def test_runner_caches_by_fingerprint(tmp_path):
    items = extraction.generate(20, seed=1)
    calls = {"n": 0}
    oracle = noisy_oracle(extraction.truth_lookup(items), seed=1)

    def counting(state, qs):
        calls["n"] += 1
        return oracle(state, qs)

    runner = JevRunner(tmp_path / "c.jsonl", client_factory=lambda: MockJev(counting))
    qs = extraction.questions()
    r1 = runner.run(items, qs); r2 = runner.run(items, qs)
    assert len(r1) == 20 and r1.keys() == r2.keys() and calls["n"] == 20
    qs2 = dict(qs); qs2["coverage_type"] = Noul(instructions="different")
    runner.run(items, qs2)
    assert calls["n"] == 40  # new wording -> new fingerprint -> re-queried


def _noisy_scores(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    raw = np.clip(np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n)) ** 0.5, 0.5, 1)  # overconfident
    return raw, y


def test_calibrate_question_reduces_ece_and_binds_fingerprint(tmp_path):
    raw, y = _noisy_scores()
    cq = calibrate_question(raw, y, "q", "fp1")
    assert cq.report["isotonic"]["ece"] < cq.report["raw"]["ece"]
    cq.check("fp1")
    with pytest.raises(ValueError):
        cq.check("fp2")
    cq.save(tmp_path / "c.json")
    back = CalibratedQuestion.load(tmp_path / "c.json")
    assert np.allclose(back.predict(raw), cq.predict(raw))
    with pytest.raises(ValueError):
        calibrate_question(raw[:20], y[:20], "q", "fp1")


def test_threshold_for_accuracy():
    conf = np.array([0.99, 0.9, 0.8, 0.7, 0.6]); correct = np.array([1, 1, 1, 0, 1])
    t = threshold_for_accuracy(conf, correct, 0.95)
    assert t["threshold"] == 0.8 and t["coverage"] == 0.6 and t["accuracy"] == 1.0
    assert threshold_for_accuracy(conf, np.zeros(5), 0.5)["threshold"] is None


def test_bracket_gap_fraction():
    rng = np.random.default_rng(0)
    items = [{"truth": int(rng.integers(0, 4))} for _ in range(400)]
    jev = lambda it: it["truth"] if rng.random() < 0.7 else int(rng.integers(0, 4))
    res = bracket(items, {"random": lambda it: int(rng.integers(0, 4)), "jev": jev, "oracle": lambda it: it["truth"]},
                  accuracy_metric, n_boot=200)
    assert res["metrics"]["oracle"]["point"] == 1.0
    g = res["gap_fraction"]["jev"]["point"]
    assert 0.5 < g < 0.95 and res["gap_fraction"]["jev"]["ci90"][0] < g < res["gap_fraction"]["jev"]["ci90"][1]


def test_counterfactual():
    ids = [f"b{i}" for i in range(500)]; rej = [i % 2 == 0 for i in range(500)]
    s = sample_rejected(ids, rej, rate=0.1, seed=0)
    assert len(s) == 30 and all(int(x[1:]) % 2 == 0 for x in s)
    r = false_reject_rate(30, 3)
    assert r["ci90"][0] < 0.1 < r["ci90"][1]
    assert expansions_for_precision(0.1, 0.05) == 98


def test_phrasing_panel():
    raw, y = _noisy_scores(800)
    cal = np.arange(800) % 5 != 0
    rows = phrasing_panel({"base": raw, "same": raw.copy(), "noisier": np.clip(raw + np.random.default_rng(1).normal(0, 0.2, 800), 0, 1)},
                          y, cal, baseline="base")
    by = {r["variant"]: r for r in rows}
    assert by["base"]["mcnemar_p_vs_baseline"] is None and by["same"]["mcnemar_p_vs_baseline"] == 1.0
    assert by["noisier"]["auroc"] < by["base"]["auroc"]
    assert mcnemar_exact([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_extraction_truth_matches_document():
    items = extraction.generate(200, seed=3, p_omit=0.3, n_distractors=2)
    qs = extraction.questions()
    assert set(qs) == set(items[0]["truth"])
    for it in items:
        doc, t = it["state"]["document"], it["truth"]
        if t["coverage_type"] != "not_stated":
            assert t["coverage_type"] in doc
        else:
            assert "insurance policy" not in doc
        assert (t["effective_month"] == "not_stated") == (t["effective_year"] == "not_stated")
        if t["effective_year"] != "not_stated":
            assert t["effective_year"] in doc and t["effective_month"].capitalize() in doc
        assert ("flood endorsement" in doc) == t["flood_rider"]
    assert 0.15 < np.mean([it["truth"]["policyholder_state"] == "not_stated" for it in items]) < 0.45


def test_reflex_labels_and_arms():
    items = reflex.generate(90, seed=0, p_injected=0.5)
    kinds = {it["kind"] for it in items}
    assert kinds == {"benign", "destructive", "out_of_scope"}
    for it in items:
        t = it["truth"]
        assert t["should_block"] == (t["destructive"] or t["out_of_scope"])
        assert t["injection_present"] == (it["arm"] == "injected")
        assert (it["arm"] == "injected") == any(s in it["state"]["context_excerpt"] for s in reflex.INJECTIONS)
    assert set(reflex.questions()) == set(items[0]["truth"])


def test_tree_prune_rank_and_counterfactual():
    fx = tree_prune.FIXTURE; br = fx["branches"]
    qs = tree_prune.questions(len(br))
    assert len(qs) == 4 * len(br) and "branches[3].step" in qs["b3_dead_end"].instructions
    answers = {f"b{i}_dead_end": {"type": "noul", "noul": 0.1 if b.outcome else 0.8} for i, b in enumerate(br)}
    keep, rejected = tree_prune.rank(answers, br, k=2)
    assert set(keep) == {"b0", "b3"} and len(rejected) == 4
    assert set(tree_prune.counterfactual_expansions(rejected, rate=0.5)) <= set(rejected)


def test_rts_batch_planner():
    p = rts.plan_batches(1000, 5.0)
    assert not p["feasible"] and p["units_per_request"] >= 300
    q = rts.plan_batches(200, 1.0)
    assert q["feasible"]
    assert len(rts.questions(3)) == 6
