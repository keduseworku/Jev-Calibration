"""Demo #3: structured extraction, evaluated by synthetic inverse.

A random record is rendered into a document, so every field's truth is exact by construction.
Numeric and date fields are asked as components / named bands (docs: Jev cannot compare numbers or
dates; keep arithmetic in code). Every Choice has a `not_stated` option. No sentence, template or
distractor, contains a token that is an option of a field the document leaves unstated; documents
are unique within a run. Knobs: omit (not_stated rate per field; the
month/year pair shares one draw), distractors (context-rot probe). The deductible band is a
documented-jaggedness probe: a dollar amount that must be bucketed.
"""
import random

from typesafe_sdk import Choice, Noul

NAME = "extraction"
KNOBS = {"--omit": {"type": float, "default": 0.25}, "--distractors": {"type": int, "default": 3}}
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
BAND_RANGES = {"under_500": (100, 450), "500_to_999": (500, 950), "1000_to_2499": (1000, 2450), "2500_or_more": (2500, 10000)}
STATE_NAMES = {"NY": "New York", "NJ": "New Jersey", "CT": "Connecticut", "PA": "Pennsylvania", "CA": "California", "TX": "Texas"}
DISTRACTORS = [
    "Customer service is available on weekdays during business hours by phone or online chat.",
    "Our company was founded in 1987 and serves policyholders across many states.",
    "Claims can be filed online, by phone, or by mail, and are acknowledged within three weeks.",
    "We do not sell personal information; see the privacy notice for details.",
    "A late payment fee applies to premiums received after the due date.",
    "This summary is provided for convenience; the policy contract governs in the event of a conflict.",
    "Discounts are sometimes available for bundling multiple policies, subject to underwriting approval.",
    "Roadside assistance, where purchased, is administered by a third-party provider with a one-hour target response time.",
]


def generate(n: int = 300, seed: int = 0, omit: float = 0.25, distractors: int = 3) -> list[dict]:
    rng = random.Random(seed)
    items, seen, attempts = [], set(), 0
    while len(items) < n:
        attempts += 1
        if attempts > 50 * n:
            raise ValueError(f"cannot generate {n} unique documents at omit={omit}, distractors={distractors}")
        i = len(items)
        truth = {f: ("not_stated" if rng.random() < omit else rng.choice(opts[:-1]))
                 for f, opts in FIELDS.items() if not f.startswith("effective_")}
        if rng.random() < omit:  # one draw for the date pair, so `omit` is the rate for every field
            truth["effective_month"] = truth["effective_year"] = "not_stated"
        else:
            truth["effective_month"], truth["effective_year"] = rng.choice(MONTHS), rng.choice(FIELDS["effective_year"][:-1])
        truth["flood_rider"] = rng.random() < 0.4
        s = []
        if truth["coverage_type"] != "not_stated":
            s.append(f"This is a {truth['coverage_type']} insurance policy.")
        if truth["deductible_band"] != "not_stated":
            lo, hi = BAND_RANGES[truth["deductible_band"]]
            s.append(f"The deductible is ${rng.randrange(lo, hi + 1, 50):,} per claim.")
        if truth["effective_year"] != "not_stated":
            day = rng.choice([d for d in range(1, 29) if str(d) not in FIELDS["cancellation_notice_days"]])
            s.append(f"Coverage becomes effective on {truth['effective_month'].capitalize()} {day}, {truth['effective_year']}.")
        if truth["cancellation_notice_days"] != "not_stated":
            s.append(f"Either party can cancel with {truth['cancellation_notice_days']} days' written notice.")
        if truth["policyholder_state"] != "not_stated":
            s.append(f"The policyholder resides in {STATE_NAMES[truth['policyholder_state']]}.")
        if truth["flood_rider"]:
            s.append("A flood endorsement is included at no additional charge.")
        s += rng.sample(DISTRACTORS, min(distractors, len(DISTRACTORS)))
        rng.shuffle(s)
        doc = " ".join(s) or "No details are provided."
        if doc in seen:
            continue
        seen.add(doc)
        items.append({"id": f"ext-{seed}-{i}", "state": {"document": doc}, "truth": truth})
    return items


def questions() -> dict:
    qs = {f: Choice(instructions=f"Which {f.replace('_', ' ')} does `document` state? Choose not_stated if the document does not say.",
                    criteria={o: ("The document does not state this." if o == "not_stated" else None) for o in opts})
          for f, opts in FIELDS.items()}
    qs["flood_rider"] = Noul(instructions="Does `document` state that a flood rider or flood endorsement is included in the policy?")
    return qs
