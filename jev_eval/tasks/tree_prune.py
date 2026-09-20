"""Demo #1: tree search where an LLM proposes branches and Jev prunes.

Docs say System Two reasoning (is this step valid?) is out of scope, so the questions here are
surface features a fast judge can plausibly see: repeats earlier work, drifts from the goal, uses an
undefined quantity. Top-k keep is rank-based (no calibration needed). A 'prune below p' rule needs
calibration against expanded outcomes, which counterfactual.sample_rejected supplies.
One Noul per branch via path references; never a k-way Choice over branches (anth.us: Choice is
uninformative below 95% confidence).
"""
from dataclasses import dataclass

from typesafe_sdk import Noul

from ..counterfactual import sample_rejected

FEATURES = {
    "repeats": "Does `branches[{i}].step` repeat or restate work already present in `so_far`?",
    "drifts": "Does `branches[{i}].step` pursue something other than `goal`?",
    "undefined": "Does `branches[{i}].step` use a symbol or quantity that is defined neither in `so_far` nor in the step itself?",
    "dead_end": "Is `branches[{i}].step` unlikely to move `so_far` toward `goal`?",
}


@dataclass
class Branch:
    id: str
    step: str
    outcome: bool | None = None  # True if expanding this branch reached the goal; None = not expanded


def pack(goal: str, so_far: str, branches: list[Branch]) -> dict:
    return {"goal": goal, "so_far": so_far, "branches": [{"id": b.id, "step": b.step} for b in branches]}


def questions(n_branches: int, features=("dead_end", "repeats", "drifts", "undefined")) -> dict:
    return {f"b{i}_{f}": Noul(instructions=FEATURES[f].format(i=i)) for i in range(n_branches) for f in features}


def rank(answers: dict, branches: list[Branch], k: int, key: str = "dead_end") -> tuple[list[str], list[str]]:
    """Keep the k branches with the lowest P(dead_end). Rank-based: raw probabilities are fine here."""
    scored = sorted(range(len(branches)), key=lambda i: answers[f"b{i}_{key}"]["noul"])
    keep = [branches[i].id for i in scored[:k]]
    rejected = [branches[i].id for i in scored[k:]]
    return keep, rejected


def counterfactual_expansions(rejected_ids: list[str], rate: float = 0.10, seed: int = 0) -> list[str]:
    return sample_rejected(rejected_ids, [True] * len(rejected_ids), rate=rate, seed=seed)


# Tiny hand-written fixture for smoke tests. Not an eval set.
FIXTURE = {
    "goal": "Derive the Jeans length for an isothermal gas of sound speed c_s and density rho.",
    "so_far": "Linearised continuity and Euler equations with self-gravity; Poisson equation for the perturbed potential.",
    "branches": [
        Branch("b0", "Combine the linearised equations into a single wave equation for the density perturbation and Fourier transform it.", True),
        Branch("b1", "Write down the linearised continuity and Euler equations again with self-gravity.", False),
        Branch("b2", "Compute the virial temperature of a galaxy cluster from its mass.", False),
        Branch("b3", "Set the dispersion relation omega^2 = c_s^2 k^2 - 4 pi G rho to zero and solve for k.", True),
        Branch("b4", "Use the magnetic field strength B_0 to estimate the Alfven speed and add it to c_s.", False),
        Branch("b5", "Evaluate the integral of the perturbed potential over the sphere of radius R_vir.", False),
    ],
}
