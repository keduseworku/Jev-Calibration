"""A question set's fingerprint: hash of every wording, criterion, primitive and the model id.

Calibrators are bound to a fingerprint. Change a word, an option, or the model version and the
fingerprint changes, which forces a refit instead of silently reusing a stale calibration
(anth.us finding: wording is part of the model; docs: calibration is per question, per version).
"""
import hashlib
import json
from collections.abc import Mapping


def plain(q) -> dict:
    """Serialize an SDK question/answer or a raw dict the same way: the SDK omits None fields."""
    d = q.model_dump() if hasattr(q, "model_dump") else dict(q)
    return {k: v for k, v in d.items() if v is not None}


def question_fingerprint(questions: Mapping, model: str | None) -> str:
    payload = {"model": model or "default", "questions": {k: plain(v) for k, v in questions.items()}}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
