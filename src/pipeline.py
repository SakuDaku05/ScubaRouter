"""
Main orchestration: the cascade described in the architecture spec.
handle_query() is the single entry point the eval harness (and, later,
the real hackathon task runner) should call.

Pipeline order:
  1. Local model generates an answer (FREE)
  2. Answer normalizer strips preamble/fences (FREE, pure regex)
  3. Format validator checks structural correctness (FREE, deterministic)
     → if fails: force confidence=0.0, skip verifier, escalate immediately
  4. MF pre-check: obviously easy? skip verifier, confidence=1.0 (FREE)
  5. Self-verifier: LLM grades its own answer (FREE, uses local model)
  6. Decision gate: confidence vs. per-type/difficulty threshold (FREE)
     → if escalate: compress + call remote model (PAID, scored)
"""
from typing import Optional

from .verifier import Verifier
from .router import Router
from .compressor import Compressor
from .logger import Logger
from .format_validator import validate as format_validate
from .answer_normalizer import normalize as normalize_answer


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
        # ── Step 1: Local model always runs first (FREE) ──────────────────
        samples = self.local_model.generate(query, n_samples=self.n_consistency_samples)
        raw_answer = samples[0] if samples else ""

        # ── Step 2: Normalize the answer (FREE, pure regex) ───────────────
        # Strips "Sure! Here is...", markdown fences, trailing meta-commentary.
        # This prevents false format-validation failures caused by preamble.
        answer = normalize_answer(raw_answer)

        # ── Step 3: Format validation (FREE, deterministic) ───────────────
        format_ok, format_reason = format_validate(task_type or "", query, answer)
        format_forced_escalation = False
        skip_verification = False

        if not format_ok:
            # Structural format failure — don't waste a verifier call on a
            # broken answer; force escalation immediately.
            confidence = 0.0
            format_forced_escalation = True
        else:
            # ── Step 4: Fast pre-check (FREE) ─────────────────────────────
            skip_verification = self.router.should_skip_verification(query)
            if skip_verification:
                confidence = 1.0
            else:
                # ── Step 5: Self-verifier (FREE, local model) ─────────────
                confidence = self.verifier.score(query, answer, samples=samples)

        # ── Step 6: Decision gate (FREE) ──────────────────────────────────
        decision = self.router.decide(query, task_type, confidence, difficulty=difficulty)

        remote_tokens = 0
        final_answer = answer

        if decision.route == "escalate":
            compressed_prompt = self.compressor.compress(query, context=answer)
            remote_answer, remote_tokens = self.remote_model.generate(compressed_prompt)
            final_answer = remote_answer or answer  # fall back to local if remote fails

        record = {
            "query": query,
            "task_type": task_type,
            "difficulty": difficulty,
            "raw_local_answer": raw_answer,
            "local_answer": answer,          # normalized
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
        }
        self.logger.log(record)
        return record
