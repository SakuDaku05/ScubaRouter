"""
Decision gate: the only component that decides local vs. escalate.
Always runs AFTER the local model has already produced an answer --
never routes a query to remote without first trying it for free.
"""
from dataclasses import dataclass
from typing import List, Optional

from .mf_router_strict import MatrixFactorizationPreCheck


@dataclass
class RoutingDecision:
    route: str            # "local" or "escalate"
    reason: str
    confidence: float


class Router:
    def __init__(self, threshold: float, static_escalate_task_types: Optional[List[str]] = None,
                 use_mf_precheck: bool = False):
        self.threshold = threshold
        self.static_escalate_task_types = set(static_escalate_task_types or [])
        self.precheck = MatrixFactorizationPreCheck(use_pretrained=use_mf_precheck)

    def should_skip_verification(self, query: str) -> bool:
        """Fast pre-check: if this looks obviously easy, skip the extra
        local verification call to save latency. Never used to skip
        straight to remote -- only to shortcut to a confident 'local' route."""
        result = self.precheck.check(query)
        return result.looks_easy

    def decide(self, query: str, task_type: Optional[str], confidence: float) -> RoutingDecision:
        if task_type and task_type in self.static_escalate_task_types:
            return RoutingDecision(route="escalate", reason="static_override", confidence=confidence)

        if confidence >= self.threshold:
            return RoutingDecision(route="local", reason="confidence_above_threshold", confidence=confidence)

        return RoutingDecision(route="escalate", reason="confidence_below_threshold", confidence=confidence)
