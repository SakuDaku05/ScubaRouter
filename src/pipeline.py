"""
Main orchestration: the cascade described in the architecture spec.
handle_query() is the single entry point the eval harness calls.

Pipeline order (all steps except escalation are FREE — zero scored tokens):
  1. Auto-classify task type from prompt keywords (FREE, deterministic)
  2. Local model generates an answer (FREE — local tokens not scored)
  3. Answer normalizer strips preamble/fences (FREE, pure regex)
  4. Format validator checks structural correctness (FREE, deterministic)
     → if fails: force confidence=0.0, skip verifier, escalate immediately
  5. Supra / MF pre-check: obviously easy? skip verifier, confidence=1.0 (FREE)
  6. Self-verifier: local model grades its own answer (FREE)
  7. Decision gate: confidence vs. per-type/difficulty threshold (FREE)
     → if escalate: compress + call remote model (PAID — tokens recorded by proxy)
       - Use fast_model for marginally-failed queries
       - Use strong_model for clearly-hard queries (math, code, logic)

Token-saving rules applied on escalation path:
  - Prompt is stripped/compressed before sending
  - max_tokens is capped per task type
  - Output-only instruction appended to prevent verbose explanations
"""
import re
from typing import Optional

from .verifier import Verifier
from .router import Router
from .compressor import Compressor
from .logger import Logger
from .format_validator import validate as format_validate
from .answer_normalizer import normalize as normalize_answer


# ── Task-type auto-classifier ─────────────────────────────────────────────────
# Maps prompt patterns to hackathon task categories.
# These drive per-type routing thresholds WITHOUT needing labeled task_type input.

_TASK_CLASSIFIERS = [
    # Code generation / debugging — check before math to avoid false positives
    ("code",    re.compile(
        r"\b(write|implement|create|code|function|class|def |program|script|"
        r"debug|fix (the |this )?bug|what('s| is) wrong|correct (the )?code)\b",
        re.IGNORECASE)),
    # Mathematical reasoning
    ("math",    re.compile(
        r"\b(calculat|comput|solv|evaluat|integrat|differentiat|derivative|"
        r"percent|probability|equation|arithmetic|algebra|proof|sum of|"
        r"how (many|much|far|long)|what is \d|\d\s*[\+\-\*\/\^]\s*\d)\b",
        re.IGNORECASE)),
    # Sentiment classification
    ("sentiment", re.compile(
        r"\b(sentiment|positive|negative|neutral|tone|emotion|feel|opinion|"
        r"review|label (the |this )?(sentiment|text)|classify (the |this )?(review|text))\b",
        re.IGNORECASE)),
    # Named entity recognition
    ("ner",     re.compile(
        r"\b(extract|identify|list|find|name).{0,40}(entit|person|organization|"
        r"location|date|company|place|people|named)\b",
        re.IGNORECASE)),
    # Text summarization
    ("summarization", re.compile(
        r"\b(summarize|summarise|summary|condense|shorten|main (idea|point|argument)|"
        r"in (one|1|two|2) sentence|key takeaway|tldr|tl;dr)\b",
        re.IGNORECASE)),
    # Logical / deductive reasoning
    ("logical_reasoning", re.compile(
        r"\b(logic|deduc|infer|puzzle|constraint|if .+ then|who (is|must|can)|"
        r"which (one|person|option)|only if|cannot both|must be (true|false)|"
        r"conclude|valid argument)\b",
        re.IGNORECASE)),
    # Factual knowledge / QA (broad catch-all, low priority)
    ("factual_knowledge", re.compile(
        r"\b(what (is|are|was|were)|who (is|was)|when (did|was)|where (is|was)|"
        r"why (does|did|is)|how does|explain|define|describe|tell me about)\b",
        re.IGNORECASE)),
]

# Per-task max_tokens for remote calls — cap output to minimize scored tokens
_MAX_TOKENS_BY_TYPE = {
    "math":              64,   # just the number + brief working
    "sentiment":         32,   # one label + one sentence justification
    "ner":              128,   # entity list
    "summarization":    128,   # summary paragraph
    "code":             512,   # full function
    "logical_reasoning": 128,  # short answer + single-line justification
    "factual_knowledge":  96,  # a concise factual answer
    "qa":                 96,
    "default":           256,
}

# Hard-escalate these task types — local models are reliably weak here.
_ALWAYS_ESCALATE_TYPES = {"logical_reasoning"}


def classify_task_type(query: str, given_type: Optional[str] = None) -> str:
    """
    If a task_type is provided by the caller, trust it.
    Otherwise, use regex heuristics to infer from the prompt.
    Falls back to 'qa' if nothing matches.
    """
    if given_type:
        return given_type
    for label, pattern in _TASK_CLASSIFIERS:
        if pattern.search(query):
            return label
    return "qa"


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RoutingPipeline:
    def __init__(self, local_model, remote_model, config: dict, logger: Optional[Logger] = None):
        self.local_model = local_model
        self.remote_model = remote_model
        self.verifier = Verifier(local_model)
        routing_cfg = config["routing"]
        self.router = Router(
            threshold=routing_cfg["verification_threshold"],
            static_escalate_task_types=routing_cfg.get("static_escalate_task_types", []),
            use_mf_precheck=routing_cfg.get("use_mf_precheck", False),
            use_supra=routing_cfg.get("use_supra", False),
            supra_complexity_at=routing_cfg.get("supra_complexity_at", 3),
            per_type_thresholds=routing_cfg.get("per_type_thresholds", {}),
            per_difficulty_thresholds=routing_cfg.get("per_difficulty_thresholds", {}),
        )
        self.compressor = Compressor(local_model)
        self.n_consistency_samples = routing_cfg.get("self_consistency_samples", 1)
        self.logger = logger or Logger()

    def handle_query(
        self,
        query: str,
        task_type: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> dict:
        # ── Step 1: Auto-classify task type (FREE) ────────────────────────
        task_type = classify_task_type(query, given_type=task_type)

        # ── Step 2: Local model generates answer (FREE) ───────────────────
        samples = self.local_model.generate(query, n_samples=self.n_consistency_samples)
        raw_answer = samples[0] if samples else ""

        # ── Step 3: Normalize the answer (FREE, pure regex) ───────────────
        answer = normalize_answer(raw_answer)

        # ── Step 4: Format validation (FREE, deterministic) ───────────────
        format_ok, format_reason = format_validate(task_type or "", query, answer)
        format_forced_escalation = False
        skip_verification = False
        from .supra_precheck import PreCheckResult as _PCR
        precheck_result = _PCR(looks_easy=False, score=0.0, source="not_run")

        if not format_ok:
            confidence = 0.0
            format_forced_escalation = True
        elif task_type in _ALWAYS_ESCALATE_TYPES:
            # Hard-escalate: local models can't reliably solve these
            confidence = 0.0
            format_forced_escalation = False
        else:
            # ── Step 5: Supra / MF pre-check (FREE) ───────────────────────
            precheck_result = self.router.precheck_query(query)
            skip_verification = precheck_result.looks_easy

            if skip_verification:
                confidence = 1.0
            elif precheck_result.score == 0.0 and "strict" not in precheck_result.source:
                confidence = 0.0
            else:
                # ── Step 6: Self-verifier (FREE, local model) ─────────────
                confidence = self.verifier.score(query, answer, samples=samples)

        # ── Step 7: Decision gate (FREE) ──────────────────────────────────
        decision = self.router.decide(query, task_type, confidence, difficulty=difficulty)

        remote_tokens = 0
        final_answer = answer

        if decision.route == "escalate":
            compressed_prompt = self.compressor.compress(query)
            max_tok = _MAX_TOKENS_BY_TYPE.get(task_type, _MAX_TOKENS_BY_TYPE["default"])

            # Use the fast (smaller/cheaper) model if confidence was moderate;
            # only pull in the strong model for definitely-hard queries.
            use_fast = (confidence > 0.3) and not format_forced_escalation
            remote_answer, remote_tokens = self.remote_model.generate(
                compressed_prompt,
                use_fast_model=use_fast,
                max_tokens=max_tok,
            )
            final_answer = remote_answer or answer

        record = {
            "query": query,
            "task_type": task_type,
            "difficulty": difficulty,
            "raw_local_answer": raw_answer,
            "local_answer": answer,
            "confidence": confidence,
            "effective_threshold": decision.effective_threshold,
            "format_ok": format_ok,
            "format_reason": format_reason,
            "format_forced_escalation": format_forced_escalation,
            "route": decision.route,
            "route_reason": decision.reason,
            "skipped_verification": skip_verification,
            "remote_tokens_used": remote_tokens,
            "final_answer": final_answer,
            "supra_source": precheck_result.source if not format_forced_escalation else "format_failed",
            "supra_complexity": getattr(precheck_result, "complexity", 0),
            "supra_domain": getattr(precheck_result, "domain", ""),
            "supra_is_math": getattr(precheck_result, "is_math", False),
            "supra_is_code": getattr(precheck_result, "is_code", False),
        }
        self.logger.log(record)
        return record
