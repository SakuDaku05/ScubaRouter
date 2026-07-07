"""
Format Validators — deterministic, zero-cost checks applied immediately
after local generation, before the LLM self-grader runs.

If the answer structurally cannot be correct for its task type
(e.g. a math question answered with a paragraph, or a code task
answered with plain prose), we force confidence to 0.0 and skip
the expensive LLM verification call entirely. The router then
sees confidence=0.0 < threshold and escalates to the remote model.

This directly solves the "overconfident local model" problem:
instead of asking the 8B model to grade its own homework, we apply
cheap deterministic rules first, before giving the model a chance
to grade itself.

Each validator returns a (passed: bool, reason: str) tuple.
  passed=True  → format looks OK, proceed to LLM verifier normally
  passed=False → format is wrong, force escalation immediately
"""
import re
import ast
from typing import Tuple


# ── helpers ────────────────────────────────────────────────────────────────

def _extract_numbers(text: str) -> list:
    """Pull all numeric values (int or float) from a string."""
    return re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)


def _is_mostly_prose(text: str, max_words_per_number: int = 30) -> bool:
    """True if there are too many words relative to numbers — signals a
    wordy paragraph answer when a concise numeric answer was expected."""
    numbers = _extract_numbers(text)
    word_count = len(text.split())
    if not numbers:
        return True  # no numbers at all in a math answer → definitely prose
    # if there are many words per number found, it looks like a prose explanation
    return (word_count / max(len(numbers), 1)) > max_words_per_number


# ── per-type validators ─────────────────────────────────────────────────────

def validate_math(query: str, answer: str) -> Tuple[bool, str]:
    """
    A math answer must contain at least one number.
    If it is a wall of text with no clear numeric result, it fails.
    """
    answer = answer.strip()
    if not answer:
        return False, "empty_answer"

    numbers = _extract_numbers(answer)
    if not numbers:
        return False, "no_number_found_in_math_answer"

    if _is_mostly_prose(answer):
        return False, "math_answer_is_prose_paragraph"

    return True, "ok"


def validate_code(query: str, answer: str) -> Tuple[bool, str]:
    """
    A code answer should contain at least one Python code indicator.
    Also try to parse a code block if present.
    """
    answer = answer.strip()
    if not answer:
        return False, "empty_answer"

    # Extract code from markdown fences if present
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", answer, re.DOTALL)
    code_to_check = fenced[0].strip() if fenced else answer

    CODE_INDICATORS = [
        r"\bdef\b", r"\bclass\b", r"\blambda\b", r"\breturn\b",
        r"\bfor\b.+\bin\b", r"\bif\b.+:", r"\bimport\b", r"\basync\b",
    ]
    for pattern in CODE_INDICATORS:
        if re.search(pattern, code_to_check):
            return True, "ok"

    # Last resort: try to parse as Python AST
    try:
        ast.parse(code_to_check)
        return True, "ok"
    except SyntaxError:
        pass

    return False, "no_code_structure_found"


def validate_summarization(query: str, answer: str) -> Tuple[bool, str]:
    """
    A summarization answer must be non-empty and reasonably shorter than
    the source text (summaries shouldn't be longer than what they summarize).
    """
    answer = answer.strip()
    if not answer:
        return False, "empty_answer"

    # Estimate the source text length from the query
    # (the query usually contains the text to be summarized)
    source_word_count = len(query.split())
    answer_word_count = len(answer.split())

    # If the summary is longer than the source, something went wrong
    if answer_word_count > source_word_count * 1.2:
        return False, "summary_longer_than_source"

    return True, "ok"


def validate_translation(query: str, answer: str) -> Tuple[bool, str]:
    """
    A translation answer must be non-empty and meaningfully different
    from the original input (not just echoing it back).
    """
    answer = answer.strip()
    if not answer:
        return False, "empty_answer"

    # Extract the text inside quotes from the query (the source string)
    quoted = re.findall(r"['\"](.+?)['\"]", query)
    source_text = quoted[-1].strip().lower() if quoted else ""

    if source_text and source_text in answer.lower():
        return False, "translation_echoes_source_verbatim"

    return True, "ok"


def validate_qa(query: str, answer: str) -> Tuple[bool, str]:
    """
    QA answers just need to be non-empty and not just repeat
    the question back verbatim.
    """
    answer = answer.strip()
    if not answer:
        return False, "empty_answer"

    # If the answer is just the query repeated, something is wrong
    if query.strip().lower() in answer.lower() and len(answer.split()) < 10:
        return False, "answer_is_just_the_question"

    return True, "ok"


def validate_sentiment(query: str, answer: str) -> Tuple[bool, str]:
    """
    Sentiment answers must contain a clear label (positive/negative/neutral)
    and be non-empty.
    """
    answer = answer.strip()
    if not answer:
        return False, "empty_answer"
    LABELS = ["positive", "negative", "neutral", "mixed"]
    lower = answer.lower()
    if not any(label in lower for label in LABELS):
        return False, "no_sentiment_label_found"
    return True, "ok"


def validate_ner(query: str, answer: str) -> Tuple[bool, str]:
    """
    NER answers must be non-empty and contain at least one entity-like token
    (capitalized word or entity label pattern).
    """
    answer = answer.strip()
    if not answer:
        return False, "empty_answer"
    # Accept if there's at least one capitalized word or entity bracket
    if re.search(r"[A-Z][a-z]+|\b(PERSON|ORG|LOC|DATE|GPE|FAC)\b|\[.*?\]", answer):
        return True, "ok"
    return False, "no_entities_found"


def validate_logical_reasoning(query: str, answer: str) -> Tuple[bool, str]:
    """
    Logical reasoning answers should be non-empty and not just
    a repetition of the question.
    """
    answer = answer.strip()
    if not answer:
        return False, "empty_answer"
    if len(answer.split()) < 2:
        return False, "answer_too_short"
    return True, "ok"


# ── dispatcher ────────────────────────────────────────────────────────────

VALIDATORS = {
    "math":              validate_math,
    "code":              validate_code,
    "code_debug":        validate_code,   # alias for hackathon category name
    "code_generation":   validate_code,   # alias
    "summarization":     validate_summarization,
    "translation":       validate_translation,
    "qa":                validate_qa,
    "factual_knowledge": validate_qa,     # same rules as QA
    "sentiment":         validate_sentiment,
    "ner":               validate_ner,
    "logical_reasoning": validate_logical_reasoning,
}


def validate(task_type: str, query: str, answer: str) -> Tuple[bool, str]:
    """
    Main entry point. Returns (passed, reason).
    If no validator exists for the task type, returns (True, 'no_validator')
    — we don't penalise unknown types.
    """
    validator = VALIDATORS.get(task_type)
    if validator is None:
        return True, "no_validator_for_type"
    return validator(query, answer)
