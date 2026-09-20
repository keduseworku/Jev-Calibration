"""Demo #1: tree search where an LLM proposes branches and Jev prunes.

Docs say System Two reasoning (is this step valid?) is out of scope, so the questions are surface
features: repeats earlier work, drifts from the goal, uses an undefined quantity, dead end. Top-k keep
is rank-based (no calibration). A 'prune below p' rule needs calibration against expanded outcomes,
which counterfactual.sample_rejected supplies. One Noul per branch via path references, never a k-way
Choice over branches (anth.us: Choice is uninformative below 95% confidence).
"""
import random

from typesafe_sdk import Noul

FEATURES = {
    "dead_end": "Is `branches[{i}].step` unlikely to move `so_far` toward `goal`?",
    "repeats": "Does `branches[{i}].step` repeat or restate work already present in `so_far`?",
    "drifts": "Does `branches[{i}].step` pursue something other than `goal`?",
    "undefined": "Does `branches[{i}].step` use a symbol or quantity that is defined neither in `so_far` nor in the step itself?",
}


def pack(goal: str, so_far: str, branches: list[dict]) -> dict:
    """branches: [{"id", "step"}]. The state whose paths the questions reference."""
    return {"goal": goal, "so_far": so_far, "branches": branches}


def questions(n_branches: int) -> dict:
    return {f"b{i}_{f}": Noul(instructions=text.format(i=i)) for i in range(n_branches) for f, text in FEATURES.items()}


def rank(answers: dict, branches: list[dict], k: int, seed: int = 0) -> tuple[list[str], list[str]]:
    """Keep the k branches with the lowest P(dead_end); ties (2-decimal probabilities) broken at random."""
    rng = random.Random(seed)
    order = sorted(range(len(branches)), key=lambda i: (answers[f"b{i}_dead_end"]["noul"], rng.random()))
    return [branches[i]["id"] for i in order[:k]], [branches[i]["id"] for i in order[k:]]
