"""
Strict version of the pre-check.
This completely disables the heuristic fallback. If RouteLLM isn't installed
or enabled, it will always return looks_easy=False, forcing the pipeline
to actually run the self-verification step on every single query.
"""
from dataclasses import dataclass

@dataclass
class PreCheckResult:
    looks_easy: bool
    score: float
    source: str


class MatrixFactorizationPreCheck:
    def __init__(self, use_pretrained: bool = False, threshold: float = 0.7):
        self.threshold = threshold
        self.use_pretrained = use_pretrained
        self._router = None
        if use_pretrained:
            try:
                # Optional dependency
                from routellm.controller import Controller  # type: ignore
                self._router = Controller(routers=["mf"])
            except Exception:
                self._router = None

    def check(self, query: str) -> PreCheckResult:
        # If RouteLLM is actually installed and enabled, use it
        if self._router is not None:
            try:
                win_rate = self._router.mf.calculate_strong_win_rate(query)
                easy_score = 1.0 - win_rate
                return PreCheckResult(looks_easy=easy_score > self.threshold,
                                       score=easy_score, source="routellm-mf")
            except Exception:
                pass  
        
        # STRICT MODE: No heuristic fallback! 
        # Always force the verifier to run.
        return PreCheckResult(looks_easy=False, score=0.0, source="strict_no_skip")
