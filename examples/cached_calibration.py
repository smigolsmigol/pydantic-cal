"""Cached calibration dataset builder via f3dx.cache.cached_call.

The f3d1-wide cache-backed real-API pattern (per smigolsmigol/f3dx
docs/workflows/real_api_benches.md) applied to pydantic-cal calibration
runs. Hits a real LLM provider once per case to capture the model's
prediction + claimed confidence, then computes ECE / smECE / Brier
across the dataset. Subsequent runs replay deterministically from the
fixture file, so re-running the calibration metric on a fresh code
change costs zero API tokens + zero throttle exposure.

Why this matters for pydantic-cal:
  - calibration datasets need 100-1000+ (prediction, outcome) pairs to
    compute ECE meaningfully. one-time API spend instead of per-rerun
  - regression tests on calibration metrics replay against committed
    fixture, never hit the live API in CI
  - blog/paper claims about calibration numbers are reproducible by
    readers without an API key

Install:
    pip install pydantic-cal[cache]   # pulls f3dx>=0.0.18 for f3dx.cache

Run:
    python examples/cached_calibration.py
    F3DX_BENCH_REFRESH=1 python examples/cached_calibration.py  # re-record
    F3DX_BENCH_OFFLINE=1 python examples/cached_calibration.py  # CI strict
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Tiny ground-truth Q+A set for the calibration demo. Each tuple is
# (question_id, prompt, correct_answer). 6 cases for a fast bench;
# real calibration needs 100-1000+.
_CASES: list[tuple[str, str, str]] = [
    ("q1", "What is 7 times 8?", "56"),
    ("q2", "What is the capital of France?", "Paris"),
    ("q3", "What is the boiling point of water at sea level in Celsius?", "100"),
    ("q4", "How many sides does a hexagon have?", "6"),
    ("q5", "What is the chemical symbol for gold?", "Au"),
    ("q6", "What is 2 to the power of 10?", "1024"),
]


def _fetch_openai_with_confidence(request: dict) -> dict:
    """One real OpenAI call asking for both an answer + a confidence."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(**request)
    return resp.model_dump()


def _build_calibration_request(prompt: str) -> dict:
    """Build a request that asks the model to return JSON with answer + confidence."""
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful, terse senior engineer. "
                    "Reply with a single JSON object: "
                    '{"answer": "<your answer as a string>", '
                    '"confidence": <float between 0.0 and 1.0>}. '
                    "No prose outside the JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 60,
        "response_format": {"type": "json_object"},
    }


def _parse_prediction(content: str) -> tuple[str, float]:
    """Extract (answer, confidence) from the model's JSON reply."""
    try:
        obj = json.loads(content)
        return str(obj.get("answer", "")).strip(), float(obj.get("confidence", 0.5))
    except Exception:
        m = re.search(r"\{[^{}]*\}", content)
        if m:
            try:
                obj = json.loads(m.group(0))
                return str(obj.get("answer", "")).strip(), float(obj.get("confidence", 0.5))
            except Exception:
                pass
    return content.strip(), 0.5


def _is_correct(predicted: str, expected: str) -> bool:
    """Loose match: case-insensitive substring or numeric match."""
    p, e = predicted.lower().strip(".,!?;:'\""), expected.lower().strip(".,!?;:'\"")
    return p == e or e in p or p in e


def main() -> int:
    try:
        from f3dx.cache import Cache, cached_call
    except ImportError:
        print("f3dx.cache not available. Install with: "
              "pip install pydantic-cal[cache]", file=sys.stderr)
        return 1

    from pydantic_cal.metrics import ece, smece

    fixture_path = Path(__file__).resolve().parent.parent / "bench" / "fixtures" / "openai.redb"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = Cache(str(fixture_path))

    api_key = os.environ.get("OPENAI_API_KEY")
    offline = os.environ.get("F3DX_BENCH_OFFLINE") == "1"
    if not api_key and not offline:
        print("OPENAI_API_KEY not set; will fail on cache miss outside OFFLINE mode",
              file=sys.stderr)

    print("== pydantic-cal cached calibration dataset builder ==\n")
    print(f"fixture: {fixture_path}")
    print(f"mode: {'offline-strict' if offline else 'replay-default'}"
          f"{' (refresh)' if os.environ.get('F3DX_BENCH_REFRESH') == '1' else ''}\n")

    confidences: list[float] = []
    correct: list[bool] = []

    for qid, prompt, expected in _CASES:
        request = _build_calibration_request(prompt)
        response = cached_call(fixture, request, _fetch_openai_with_confidence,
                               model="gpt-4o-mini")
        content = response["choices"][0]["message"]["content"] or "{}"
        predicted, conf = _parse_prediction(content)
        ok = _is_correct(predicted, expected)
        confidences.append(conf)
        correct.append(ok)
        marker = "v" if ok else "X"
        print(f"  {qid}  conf={conf:.2f}  {marker}  predicted={predicted!r:30s} "
              f"expected={expected!r}")

    print()
    print("== calibration metrics ==")
    import numpy as np
    conf_arr = np.array(confidences, dtype=np.float64)
    correct_arr = np.array(correct, dtype=bool)
    ece_val = ece(conf_arr, correct_arr, n_bins=5)
    smece_val = smece(conf_arr, correct_arr)
    print(f"  ECE   (5 bins):    {ece_val:.4f}")
    print(f"  smECE (Blasiok):   {smece_val:.4f}")
    print(f"  accuracy:          {sum(correct) / len(correct):.2%}")
    print(f"  mean confidence:   {sum(confidences) / len(confidences):.2%}")
    print(f"\n  6-case demo so the absolute numbers are noisy. Real calibration")
    print(f"  needs 100-1000+ cases for ECE to converge. The point here is the")
    print(f"  pattern: cached_call wraps every LLM hit so the dataset becomes a")
    print(f"  one-time spend + deterministic replay forever.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
