"""
Optional fast pre-check layer, inspired by RouteLLM's matrix-factorization
router (https://github.com/lm-sys/routellm).

Important: this does NOT decide escalation on its own. The project's core
design principle is that the local model is free, so no query should ever
be sent straight to remote without being tried locally first. This module
only decides whether to SKIP the (still-free) self-verification step for
queries that look obviously easy -- saving a little local latency, not
score, since local compute is free either way.

If the `routellm` package and a compatible pretrained checkpoint are
available, this will try to use it. Otherwise it falls back to a cheap
heuristic (length + keyword complexity) so the pipeline never breaks.
"""
from dataclasses import dataclass


@dataclass
class PreCheckResult:
    looks_easy: bool
    score: float
    source: str


_COMPLEXITY_MARKERS = [
    "prove", "derive", "optimi", "algorithm", "multi-step", "step by step",
    "code", "debug", "regex", "integral", "derivative", "theorem",
]


def _heuristic_precheck(query: str) -> PreCheckResult:
    q = query.lower()
    length_penalty = min(len(query) / 400.0, 1.0)  # longer -> more likely complex
    marker_hits = sum(1 for m in _COMPLEXITY_MARKERS if m in q)
    complexity = min(1.0, length_penalty * 0.5 + marker_hits * 0.2)
    easy_score = 1.0 - complexity
    return PreCheckResult(looks_easy=easy_score > 0.7, score=easy_score, source="heuristic")


class MatrixFactorizationPreCheck:
    def __init__(self, use_pretrained: bool = False, threshold: float = 0.7):
        self.threshold = threshold
        self.use_pretrained = use_pretrained
        self._router = None
        if use_pretrained:
            try:
                # Optional dependency -- only imported if explicitly enabled
                # via routing.use_mf_precheck: true in config/models.yaml.
                from routellm.controller import Controller  # type: ignore
                self._router = Controller(routers=["mf"])
            except Exception:
                self._router = None  # silently fall back to heuristic

    def check(self, query: str) -> PreCheckResult:
        if self._router is not None:
            try:
                win_rate = self._router.mf.calculate_strong_win_rate(query)
                easy_score = 1.0 - win_rate  # low predicted need for strong model = easy
                return PreCheckResult(looks_easy=easy_score > self.threshold,
                                       score=easy_score, source="routellm-mf")
            except Exception:
                pass  # fall through to heuristic on any runtime error
        return _heuristic_precheck(query)
