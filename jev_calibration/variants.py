"""A panel of alternative ways to ask Jev the same sentiment question, sent in ONE request per example.

Each variant is reduced to a single number, p_pos = P(text is positive), so they can be calibrated
and compared on equal terms.
"""
from typesafe_sdk import Choice, Noul, Score

POS_NEG = {"positive": None, "negative": None}

QUESTIONS = {
    # repeats of the base run: a determinism / independence check under a different request composition
    "noul_pos": Noul(instructions="Is the overall sentiment of this text positive?"),
    "choice2": Choice(instructions="What is the overall sentiment of this text?", criteria=POS_NEG),
    # asked from the other side
    "noul_neg": Noul(instructions="Is the overall sentiment of this text negative?"),
    # different wording
    "noul_favorable": Noul(instructions="Does the writer express a favorable, approving, or satisfied attitude?"),
    # option order / descriptions
    "choice2_swapped": Choice(instructions="What is the overall sentiment of this text?",
                              criteria={"negative": None, "positive": None}),
    "choice2_described": Choice(
        instructions="What is the overall sentiment of this text?",
        criteria={"positive": "The writer is pleased, approving, or optimistic.",
                  "negative": "The writer is displeased, critical, or pessimistic."}),
    # extra category: lets Jev say "neither"
    "choice3": Choice(instructions="What is the overall sentiment of this text?",
                      criteria={"positive": None, "negative": None, "neutral": None}),
    # graded rubric
    "score5": Score(instructions="Rate the overall sentiment of this text.",
                    criteria=["very negative", "negative", "neutral", "positive", "very positive"]),
}


def _ratio(pos, neg):
    return pos / (pos + neg) if (pos + neg) > 0 else 0.5


def p_pos_signals(answers: dict) -> dict[str, float]:
    """answers = serialized Jev answers -> {variant: P(positive)} (plus derived variants)."""
    a = answers
    out = {
        "noul_pos": a["noul_pos"]["noul"],
        "noul_neg": 1 - a["noul_neg"]["noul"],
        "noul_favorable": a["noul_favorable"]["noul"],
    }
    for k in ("choice2", "choice2_swapped", "choice2_described"):
        out[k] = a[k]["probabilities"]["positive"]
    p = a["choice3"]["probabilities"]
    out["choice3"] = _ratio(p["positive"], p["negative"])
    s = a["score5"]["probabilities"]
    out["score5"] = _ratio(s.get("3", 0) + s.get("4", 0), s.get("0", 0) + s.get("1", 0))
    out["score5_mean"] = a["score5"]["score"] / 4
    # simple ensembles across independent question framings
    out["ensemble_noul"] = (out["noul_pos"] + out["noul_neg"] + out["noul_favorable"]) / 3
    out["ensemble_all"] = sum(out[k] for k in ("noul_pos", "noul_neg", "noul_favorable", "choice2",
                                               "choice2_swapped", "choice2_described", "choice3", "score5")) / 8
    return out
