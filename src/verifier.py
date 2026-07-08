"""
Self-verification, entirely local and free. Two confidence signals:

  PRIMARY (when available): Log-probability confidence
    Uses the per-token log probabilities returned by the local model.
    This is much more reliable than LLM-as-judge because it measures the
    model's actual internal certainty, not its ability to self-critique.
    Supported by: llama-cpp-python (always), LM Studio (depends on version),
                  Fireworks AI (logprobs=True flag).
    Runtime-detected: if the server/library doesn't support it, we fall back.

  FALLBACK: Entailment-style self-check (LLM-as-judge)
    Ask the SAME local model whether its own answer addresses the query.
    Weaker but always available.

  SUPPLEMENTARY: Self-consistency
    If multiple local samples were generated, blend in their pairwise agreement.
    Applied on top of whichever primary signal is used.

Follows the AutoMix / FrugalGPT pattern — no labeled data required.
"""
import re
from difflib import SequenceMatcher
from typing import List, Optional

from .logprob_confidence import logprobs_to_confidence

VERIFY_PROMPT_TEMPLATE = """You are a strict grader. Judge whether the ANSWER correctly and \
completely addresses the QUERY. Reply with only a single line in the format:
SCORE: <a number from 0 to 10>
where 0 means completely wrong or irrelevant and 10 means fully correct and complete.

QUERY: {query}

ANSWER: {answer}

SCORE:"""


def _extract_score(text: str) -> float:
    match = re.search(r"SCORE:\s*([0-9]+(\.[0-9]+)?)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"([0-9]+(\.[0-9]+)?)", text)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val / 10.0))
        except ValueError:
            return 0.5
    return 0.5  # unknown → neutral; whether this escalates depends on your threshold


def self_consistency_score(samples: List[str]) -> float:
    """Average pairwise similarity across local samples. 1.0 = perfect agreement."""
    if len(samples) < 2:
        return 1.0
    scores = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            scores.append(SequenceMatcher(None, samples[i], samples[j]).ratio())
    return sum(scores) / len(scores) if scores else 1.0


class Verifier:
    def __init__(self, local_model, consistency_weight: float = 0.3):
        self.local_model = local_model
        self.consistency_weight = consistency_weight

    def score(
        self,
        query: str,
        answer: str,
        samples: Optional[List[str]] = None,
        logprobs: Optional[List[float]] = None,
    ) -> float:
        """
        Compute a confidence score in [0, 1].

        Priority:
          1. Log-probability confidence (if logprobs provided and non-empty)
          2. LLM-as-judge (generate_with_logprobs fallback, then plain generate)
          3. Self-consistency blend (applied on top of whichever signal fires)

        Parameters
        ----------
        query    : the original user query
        answer   : the local model's normalised answer
        samples  : all local samples (for self-consistency; optional)
        logprobs : per-token log probs of the answer (from local model; optional)
        """
        # ── Signal 1: Log-probability confidence ─────────────────────────────
        lp_conf = logprobs_to_confidence(logprobs) if logprobs else None

        if lp_conf is None:
            # Try fetching logprobs inline via generate_with_logprobs
            try:
                results = self.local_model.generate_with_logprobs(
                    VERIFY_PROMPT_TEMPLATE.format(query=query, answer=answer),
                    n_samples=1,
                )
                verify_text, verify_lp = results[0]
                # Use logprobs of the *grader output* as a signal of grader certainty
                lp_conf = logprobs_to_confidence(verify_lp) if verify_lp else None
                # Also extract the numeric score from the grader text as fallback
                judge_score = _extract_score(verify_text)
            except Exception:
                verify_text = ""
                judge_score = 0.5
                lp_conf = None
        else:
            # We already have logprobs for the answer — no need to call grader
            judge_score = None
            verify_text = ""

        # ── Signal 2: LLM-as-judge (fallback when no logprobs) ───────────────
        if lp_conf is None and not verify_text:
            verify_text = self.local_model.generate(
                VERIFY_PROMPT_TEMPLATE.format(query=query, answer=answer),
                n_samples=1,
            )[0]
            judge_score = _extract_score(verify_text)

        # Choose primary signal
        primary = lp_conf if lp_conf is not None else (judge_score or 0.5)

        # ── Signal 3: Self-consistency blend ──────────────────────────────────
        if samples and len(samples) > 1:
            consistency = self_consistency_score(samples)
            return round(
                (1 - self.consistency_weight) * primary
                + self.consistency_weight * consistency,
                4,
            )

        return round(primary, 4)
