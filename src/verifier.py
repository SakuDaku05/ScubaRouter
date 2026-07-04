"""
Self-verification, entirely local and free. Two signals blended into a
single confidence score in [0, 1]:

  1. Entailment-style self-check: ask the SAME local model whether its own
     answer actually addresses the query correctly.
  2. Self-consistency: if multiple local samples were generated, how much
     do they agree with each other?

Both are training-free (no labeled data needed) -- this follows the
AutoMix / FrugalGPT pattern rather than a trained classifier, which is
the right choice when the task distribution isn't known until kickoff.
"""
import re
from difflib import SequenceMatcher
from typing import List, Optional

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
    return 0.5  # unknown -> neutral; whether this escalates depends on your threshold


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
    def __init__(self, local_model, consistency_weight: float = 0.4):
        self.local_model = local_model
        self.consistency_weight = consistency_weight

    def score(self, query: str, answer: str, samples: Optional[List[str]] = None) -> float:
        prompt = VERIFY_PROMPT_TEMPLATE.format(query=query, answer=answer)
        verify_out = self.local_model.generate(prompt, n_samples=1)[0]
        entailment_score = _extract_score(verify_out)

        if samples and len(samples) > 1:
            consistency = self_consistency_score(samples)
            return (1 - self.consistency_weight) * entailment_score + self.consistency_weight * consistency

        return entailment_score
