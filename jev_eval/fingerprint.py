"""A question set's fingerprint: hash of every wording, criterion, primitive and the exact model id.

Answer caches are keyed by it, and calibrators are recomputed from those answers each run, so a
changed word, option or model can never meet a stale calibration (anth.us: wording is part of the
model; docs: calibration is per question, per version). The model id must be explicit: the server
default can roll over underneath a cache.
"""
import hashlib
import json
from collections.abc import Mapping


def plain(q) -> dict:
    """Serialize an SDK question/answer or a raw dict the same way: the SDK omits None fields."""
    d = q.model_dump() if hasattr(q, "model_dump") else dict(q)
    return {k: v for k, v in d.items() if v is not None}


def question_fingerprint(questions: Mapping, model: str) -> str:
    if not model:
        raise ValueError("an explicit model id is required (e.g. 'jev-1.13.0'); the server default can change under you")
    payload = {"model": model, "questions": {k: plain(v) for k, v in questions.items()}}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
