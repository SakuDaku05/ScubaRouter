"""
Decision gate: the only component that decides local vs. escalate.
Always runs AFTER the local model has already produced an answer --
never routes a query to remote without first trying it for free.

Now supports per-task-type and per-difficulty confidence thresholds,
so different categories get tuned independently without touching code.
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
        per_type_thresholds: Optional[Dict[str, float]] = None,
        per_difficulty_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.global_threshold = threshold
        self.static_escalate_task_types = set(static_escalate_task_types or [])
        self.precheck = MatrixFactorizationPreCheck(use_pretrained=use_mf_precheck)
        self.per_type_thresholds = per_type_thresholds or {}
        self.per_difficulty_thresholds = per_difficulty_thresholds or {}

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

    def should_skip_verification(self, query: str) -> bool:
        """Fast pre-check: if obviously easy, skip the extra local verification
        call to save latency. Never skips straight to remote."""
        result = self.precheck.check(query)
        return result.looks_easy

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
