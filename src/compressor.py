"""
Prompt compression applied ONLY on the escalation path, since every token
sent to Fireworks counts against the score. Two stages:

  1. Deterministic strip -- free, instant, no model call.
  2. Local-model summarization for long context -- still free, uses local
     compute to shrink what gets sent remotely.
"""
import re

_BLANK_LINES = re.compile(r"\n{3,}")
_REPEATED_SPACES = re.compile(r"[ \t]{2,}")

MAX_CONTEXT_CHARS_BEFORE_SUMMARY = 1500

SUMMARIZE_PROMPT_TEMPLATE = """Summarize the following context in the fewest words possible \
while keeping every fact needed to answer the query below. Do not add commentary.

QUERY: {query}

CONTEXT:
{context}

SUMMARY:"""


def deterministic_strip(text: str) -> str:
    cleaned = text.strip()
    cleaned = _BLANK_LINES.sub("\n", cleaned)
    cleaned = _REPEATED_SPACES.sub(" ", cleaned)
    return cleaned


class Compressor:
    def __init__(self, local_model):
        self.local_model = local_model

    def compress(self, query: str, context: str = "") -> str:
        query = deterministic_strip(query)
        context = deterministic_strip(context) if context else ""

        if len(context) > MAX_CONTEXT_CHARS_BEFORE_SUMMARY:
            summary_prompt = SUMMARIZE_PROMPT_TEMPLATE.format(query=query, context=context)
            context = self.local_model.generate(summary_prompt, n_samples=1)[0]

        if context:
            return f"{query}\n\nRelevant context:\n{context}"
        return query
