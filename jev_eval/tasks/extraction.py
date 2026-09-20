"""Demo #3: structured extraction, evaluated by synthetic inverse.

Ground truth by construction: a random record is rendered into a document, so every field's truth is
exact. Numeric and date fields are asked as components / named bands (docs: Jev cannot compare
numbers or dates; keep arithmetic in code). Every Choice has a `not_stated` option (docs + anth.us).
Knobs that probe documented jaggedness: p_omit (not_stated rate), n_distractors (context rot),
numeric deductible rendered as a dollar amount that must be bucketed (number weakness).
"""
import hashlib
import random

from typesafe_sdk import Choice, Noul

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"]
FIELDS = {
    "coverage_type": ["auto", "homeowners", "renters", "life", "not_stated"],
    "deductible_band": ["under_500", "500_to_999", "1000_to_2499", "2500_or_more", "not_stated"],
    "effective_month": MONTHS + ["not_stated"],
    "effective_year": ["2024", "2025", "2026", "2027", "2028", "not_stated"],
    "cancellation_notice_days": ["10", "30", "60", "90", "not_stated"],
    "policyholder_state": ["NY", "NJ", "CT", "PA", "CA", "TX", "not_stated"],
}
NOUL_FIELDS = {"flood_rider": "Does `document` state that a flood rider or flood endorsement is included in the policy?"}
BAND_RANGES = {"under_500": (100, 499), "500_to_999": (500, 999), "1000_to_2499": (1000, 2499), "2500_or_more": (2500, 10000)}
STATE_NAMES = {"NY": "New York", "NJ": "New Jersey", "CT": "Connecticut", "PA": "Pennsylvania", "CA": "California", "TX": "Texas"}

DISTRACTORS = [
    "Customer service is available Monday through Friday from 8:00 a.m. to 6:00 p.m. Eastern time at 1-800-555-0142.",
    "Our company was founded in 1987 and has served policyholders across 41 states.",
    "Claims may be filed online, by phone, or by mail; most claims are acknowledged within 15 business days.",
    "We do not sell personal information. See the privacy notice dated March 2019 for details.",
    "A late payment fee of $25 applies to premiums received more than 10 days after the due date.",
    "This document is provided for convenience; the policy contract governs in the event of a conflict.",
    "Discounts may be available for bundling multiple policies, subject to underwriting approval in 2023 guidelines.",
    "Roadside assistance, where purchased, is administered by a third-party provider with a 60-minute target response time.",
]


def _sentence(field, value, rng):
    if field == "coverage_type":
        return f"This is a {value} insurance policy."
    if field == "deductible_band":
        lo, hi = BAND_RANGES[value]
        amt = rng.randrange(lo, hi + 1, 50) if hi - lo >= 50 else lo
        return f"The deductible is ${amt:,} per claim."
    if field == "effective_month":
        return None  # rendered jointly with the year
    if field == "effective_year":
        return None
    if field == "cancellation_notice_days":
        return f"Either party may cancel with {value} days' written notice."
    if field == "policyholder_state":
        return f"The policyholder resides in {STATE_NAMES[value]}."
    raise KeyError(field)


def generate(n: int = 300, seed: int = 0, p_omit: float = 0.25, n_distractors: int = 3) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        truth, sentences = {}, []
        for f, opts in FIELDS.items():
            real = opts[:-1]
            truth[f] = "not_stated" if rng.random() < p_omit else rng.choice(real)
        # month and year are stated together or not at all, so components stay consistent
        if truth["effective_month"] == "not_stated" or truth["effective_year"] == "not_stated":
            truth["effective_month"] = truth["effective_year"] = "not_stated"
        else:
            day = rng.randint(1, 28)
            sentences.append(f"Coverage becomes effective on {truth['effective_month'].capitalize()} {day}, {truth['effective_year']}.")
        for f in FIELDS:
            if truth[f] != "not_stated":
                s = _sentence(f, truth[f], rng)
                if s:
                    sentences.append(s)
        truth["flood_rider"] = rng.random() < 0.4
        if truth["flood_rider"]:
            sentences.append("A flood endorsement is included at no additional charge.")
        sentences += rng.sample(DISTRACTORS, min(n_distractors, len(DISTRACTORS)))
        rng.shuffle(sentences)
        doc = " ".join(sentences)
        items.append({"id": hashlib.sha1(doc.encode()).hexdigest()[:12], "state": {"document": doc}, "truth": truth})
    return items


def questions() -> dict:
    qs = {}
    for f, opts in FIELDS.items():
        label = f.replace("_", " ")
        qs[f] = Choice(
            instructions=f"Which {label} does `document` state? Choose not_stated if the document does not say.",
            criteria={o: ("The document does not state this." if o == "not_stated" else None) for o in opts})
    for f, text in NOUL_FIELDS.items():
        qs[f] = Noul(instructions=text)
    return qs


def truth_lookup(items):
    by_doc = {it["state"]["document"]: it["truth"] for it in items}
    return lambda state: by_doc[state["document"]]


def random_decider(field, seed=0):
    rng = random.Random(seed)
    opts = FIELDS.get(field, [True, False])
    return lambda it: rng.choice(opts)
