"""
Answer Normalizer — strips conversational preamble and markdown formatting
from the local model's raw output BEFORE it reaches the format validator
and verifier.

This prevents false escalations caused by patterns like:
  "Sure! Here is the answer: 42"   → normalizes to "42"
  "```python\ndef foo(): ...\n```"  → normalizes to "def foo(): ..."
  "Great question! The capital is Paris." → "Paris"

All normalization is deterministic regex — zero latency, zero cost.
"""
import re


# ── Preamble patterns to strip ───────────────────────────────────────────────
# Ordered: most specific first. Each pattern covers the start of the string.
_PREAMBLE_PATTERNS = [
    # "Sure! Here is the answer:" / "Here's the answer:"
    r"^(?:sure[!,.]?\s*)?here(?:'s| is)(?: the)?(?: (?:answer|translation|solution|result|code|output|summary))?\s*[:\-]\s*",
    # "Of course! " / "Absolutely! " / "Certainly! "
    r"^(?:of course|absolutely|certainly|definitely)[!,.]?\s*",
    # "Great question! "
    r"^(?:great|good|excellent|interesting|nice)\s+(?:question|query)[!,.]?\s*",
    # "The answer is:" / "The result is:"
    r"^the\s+(?:answer|result|solution|translation|output)\s+(?:is|would be|to this is)\s*[:\-]?\s*",
    # "In (language), the translation is:"
    r"^in\s+\w+(?:\s+\w+)?\s*,\s*(?:the\s+)?translation\s+(?:is|would be)\s*[:\-]?\s*",
    # "Let me (calculate/translate/solve)..."
    r"^let me\s+(?:calculate|translate|solve|explain|answer|help)\s+(?:that|this|you)?\s*[:\-]?\s*",
    # "Based on the (question/context/information), "
    r"^based on (?:the\s+)?(?:question|context|information|query|prompt)[,.]?\s*",
    # "To answer (your question), "
    r"^to\s+(?:answer|solve|address)\s+(?:your\s+)?(?:question|query)[,.]?\s*",
]

_PREAMBLE_RE = re.compile(
    "|".join(f"(?:{p})" for p in _PREAMBLE_PATTERNS),
    re.IGNORECASE,
)

# ── Markdown fence stripper ───────────────────────────────────────────────────
_FENCE_RE = re.compile(r"^```(?:\w+)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    """If the entire answer is wrapped in a markdown code fence, unwrap it."""
    m = _FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


def _strip_preamble(text: str) -> str:
    """Iteratively strip leading conversational preamble patterns."""
    prev = None
    while prev != text:
        prev = text
        text = _PREAMBLE_RE.sub("", text, count=1).strip()
    return text


def _strip_trailing_meta(text: str) -> str:
    """
    Strip trailing sentences that are meta-commentary rather than the answer.
    e.g. "...Let me know if you need more help!" or "I hope this helps!"
    """
    _TRAILING = re.compile(
        r"\s*(?:let me know|i hope|feel free|please (?:let|feel)|if you have)[^.!?]*[.!?]?\s*$",
        re.IGNORECASE,
    )
    return _TRAILING.sub("", text).strip()


def normalize(raw_answer: str) -> str:
    """
    Main entry point.
    Returns a cleaned answer string ready for format_validator and verifier.
    The original answer is preserved if nothing matches.
    """
    if not raw_answer or not raw_answer.strip():
        return raw_answer

    text = raw_answer.strip()
    text = _strip_fences(text)
    text = _strip_preamble(text)
    text = _strip_trailing_meta(text)

    # Fall back to original if we accidentally stripped everything
    return text if text.strip() else raw_answer.strip()
