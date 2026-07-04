"""
Main orchestration: the cascade described in the architecture spec.
handle_query() is the single entry point the eval harness (and, later,
the real hackathon task runner) should call.
"""
from typing import Optional

from .verifier import Verifier
from .router import Router
from .compressor import Compressor
from .logger import Logger
from .format_validator import validate as format_validate


class RoutingPipeline:
    def __init__(self, local_model, remote_model, config: dict, logger: Optional[Logger] = None):
        self.local_model = local_model
        self.remote_model = remote_model
        self.verifier = Verifier(local_model)
        self.router = Router(
            threshold=config["routing"]["verification_threshold"],
            static_escalate_task_types=config["routing"].get("static_escalate_task_types", []),
            use_mf_precheck=config["routing"].get("use_mf_precheck", False),
        )
        self.compressor = Compressor(local_model)
        self.n_consistency_samples = config["routing"].get("self_consistency_samples", 1)
        self.logger = logger or Logger()

    def handle_query(self, query: str, task_type: Optional[str] = None) -> dict:
        # 1. Always try locally first -- free, no matter what.
        samples = self.local_model.generate(query, n_samples=self.n_consistency_samples)
        answer = samples[0] if samples else ""

        # 2. Format validation — deterministic, zero-cost, runs before the LLM
        #    self-grader. If the answer is structurally wrong for the task type
        #    (e.g. a paragraph instead of a number for math), we force
        #    confidence=0.0 immediately and skip the expensive verifier call.
        format_ok, format_reason = format_validate(task_type or "", query, answer)
        format_forced_escalation = False

        if not format_ok:
            # Structural format failure — don't bother asking the model to
            # grade its own broken answer; escalate straight away.
            confidence = 0.0
            format_forced_escalation = True
            skip_verification = False
        else:
            # 3. Fast pre-check: skip LLM verification for obviously-easy queries.
            skip_verification = self.router.should_skip_verification(query)
            if skip_verification:
                confidence = 1.0
            else:
                confidence = self.verifier.score(query, answer, samples=samples)

        # 4. Decision gate -- the only place that can trigger a paid call.
        decision = self.router.decide(query, task_type, confidence)

        remote_tokens = 0
        final_answer = answer

        if decision.route == "escalate":
            compressed_prompt = self.compressor.compress(query, context=answer)
            remote_answer, remote_tokens = self.remote_model.generate(compressed_prompt)
            final_answer = remote_answer or answer  # fall back to local if remote fails

        record = {
            "query": query,
            "task_type": task_type,
            "local_answer": answer,
            "confidence": confidence,
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
