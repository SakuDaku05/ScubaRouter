"""
Prompt compression applied ONLY on the escalation path, since every token
sent to the remote API counts against the score.

Updates:
1. We no longer send the local model's failed answer as context (it wastes tokens
   and can confuse the remote model).
2. We append a strict instruction to prevent the 70B model from outputting
   long explanations. We want a 1-token output if possible.
"""
import re

_BLANK_LINES = re.compile(r"\n{3,}")
_REPEATED_SPACES = re.compile(r"[ \t]{2,}")

STRICT_OUTPUT_INSTRUCTION = (
    "Output ONLY the final answer. Do not include any explanations, "
    "reasoning, formatting, or conversational preamble."
)

def deterministic_strip(text: str) -> str:
    cleaned = text.strip()
    cleaned = _BLANK_LINES.sub("\n", cleaned)
    cleaned = _REPEATED_SPACES.sub(" ", cleaned)
    return cleaned

class Compressor:
    def __init__(self, local_model):
        # Local model no longer needed for summarization, but kept for API compatibility
        self.local_model = local_model

    def compress(self, query: str, context: str = "") -> str:
        # We ignore 'context' intentionally now, as sending the local model's
        # wrong answer just wastes remote input tokens and anchors the remote model to wrong logic.
        
        cleaned_query = deterministic_strip(query)
        
        # Append strict instruction to save output tokens
        return f"{cleaned_query}\n\n{STRICT_OUTPUT_INSTRUCTION}"
