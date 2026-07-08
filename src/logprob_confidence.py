"""
logprob_confidence.py
---------------------
Converts raw per-token log probabilities (from OpenAI-compatible APIs or
llama-cpp-python) into a normalised [0, 1] confidence score.

Design rationale
----------------
- Log probabilities are in (-inf, 0].  A token with logprob = 0 means the
  model was 100% sure it was the right token.  A logprob of -5 means the
  model put only ~0.7% probability on that token.
- We take the MEAN log probability over all completion tokens. This gives
  a per-token geometric mean probability, which is a good proxy for how
  "surprised" the model was by its own answer.
- We then map that mean onto [0, 1] with a calibrated sigmoid-like rescale
  based on empirical ranges:
    mean_logprob ≈  0.0  →  extremely confident  →  confidence = 1.0
    mean_logprob ≈ -1.0  →  moderately confident →  confidence ≈ 0.78
    mean_logprob ≈ -2.0  →  uncertain             →  confidence ≈ 0.50
    mean_logprob ≈ -4.0  →  very uncertain         →  confidence ≈ 0.18
    mean_logprob ≤ -6.0  →  almost random          →  confidence ≈ 0.0

  Formula:  confidence = sigmoid((mean_logprob + 2.0) * 1.2)
  where sigmoid(x) = 1 / (1 + exp(-x)).
  This centres the sigmoid at mean_logprob = -2.0 with mild steepness.

Graceful fallback
-----------------
- If logprobs is None (server doesn't support it, or call failed), the
  function returns None, and the caller should fall back to the LLM-as-judge
  verifier.
- If logprobs is an empty list, returns None.
"""

import math
from typing import List, Optional


# Calibration constants (tune here if needed)
_CENTRE = -2.0   # the mean_logprob that maps to confidence = 0.5
_SLOPE  =  1.2   # steepness of the sigmoid


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def logprobs_to_confidence(logprobs: Optional[List[float]]) -> Optional[float]:
    """
    Convert a list of per-token log probabilities to a [0, 1] confidence score.

    Returns None if logprobs is None or empty (caller should fall back to
    the LLM-as-judge verifier).
    """
    if not logprobs:
        return None

    # Filter out any -inf values (padding tokens, etc.)
    valid = [lp for lp in logprobs if math.isfinite(lp)]
    if not valid:
        return None

    mean_lp = sum(valid) / len(valid)
    return round(_sigmoid((mean_lp - _CENTRE) * _SLOPE), 4)


def mean_logprob(logprobs: Optional[List[float]]) -> Optional[float]:
    """Convenience: just compute the mean log prob without converting."""
    if not logprobs:
        return None
    valid = [lp for lp in logprobs if math.isfinite(lp)]
    return sum(valid) / len(valid) if valid else None
