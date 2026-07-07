"""
Decision gate: the only component that decides local vs. escalate.
Always runs AFTER the local model has already produced an answer --
never routes a query to remote without first trying it for free.

Now supports per-task-type and per-difficulty confidence thresholds,
so different categories get tuned independently without touching code.

Pre-check priority:
  1. SupraPreCheck (Supra-Router-51M, 51M params) — if use_supra=True
  2. MatrixFactorizationPreCheck (RouteLLM MF)    — if use_mf_precheck=True
  3. Strict mode: always force verifier             — default fallback
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from .mf_router_strict import MatrixFactorizationPreCheck


@dataclass
class RoutingDecision:
    route: str            # "local" or "escalate"
    reason: str
    confidence: float
    effective_threshold: float   # the threshold that was actually applied


class Router:
    def __init__(
        self,
        threshold: float,
        static_escalate_task_types: Optional[List[str]] = None,
        use_mf_precheck: bool = False,
        use_supra: bool = False,
        supra_complexity_at: int = 3,
        per_type_thresholds: Optional[Dict[str, float]] = None,
        per_difficulty_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.global_threshold = threshold
        self.static_escalate_task_types = set(static_escalate_task_types or [])
        self.per_type_thresholds = per_type_thresholds or {}
        self.per_difficulty_thresholds = per_difficulty_thresholds or {}

        # Pre-check selection: Supra > MF > strict
        if use_supra:
            from .supra_precheck import SupraPreCheck
            self.precheck = SupraPreCheck(complexity_escalate_at=supra_complexity_at)
        else:
            self.precheck = MatrixFactorizationPreCheck(use_pretrained=use_mf_precheck)

    def _effective_threshold(
        self, task_type: Optional[str], difficulty: Optional[str]
    ) -> tuple:
        """
        Priority: difficulty override > task-type threshold > global threshold.
        Returns (threshold, reason_label).
        """
        if difficulty and difficulty in self.per_difficulty_thresholds:
            return self.per_difficulty_thresholds[difficulty], f"per_difficulty[{difficulty}]"
        if task_type and task_type in self.per_type_thresholds:
            return self.per_type_thresholds[task_type], f"per_type[{task_type}]"
        return self.global_threshold, "global"

    def precheck_query(self, query: str):
        """Run the active pre-checker and return the full result object.
        Callers can use result.looks_easy, result.score, and result.source."""
        return self.precheck.check(query)

    def should_skip_verification(self, query: str) -> bool:
        """Convenience wrapper kept for backward compatibility."""
        return self.precheck_query(query).looks_easy

    def decide(
        self,
        query: str,
        task_type: Optional[str],
        confidence: float,
        difficulty: Optional[str] = None,
    ) -> RoutingDecision:
        # Hard static override — always escalate this task type
        if task_type and task_type in self.static_escalate_task_types:
            t, _ = self._effective_threshold(task_type, difficulty)
            return RoutingDecision(
                route="escalate", reason="static_override",
                confidence=confidence, effective_threshold=t,
            )

        threshold, threshold_reason = self._effective_threshold(task_type, difficulty)

        if confidence >= threshold:
            return RoutingDecision(
                route="local",
                reason=f"confidence_above_{threshold_reason}_threshold",
                confidence=confidence,
                effective_threshold=threshold,
            )

        return RoutingDecision(
            route="escalate",
            reason=f"confidence_below_{threshold_reason}_threshold",
            confidence=confidence,
            effective_threshold=threshold,
        )
