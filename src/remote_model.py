"""
Remote model wrapper — Fireworks AI.

On evaluation day, the harness injects three env vars:
  FIREWORKS_API_KEY   — use this, NOT your own key
  FIREWORKS_BASE_URL  — all calls must go through here (the judging proxy)
  ALLOWED_MODELS      — comma-separated list of permitted model IDs

Model selection strategy (token-efficiency):
  We pick TWO models from ALLOWED_MODELS:
    • strong_model  — the largest/most-capable one (for hard tasks)
    • fast_model    — the smallest one (for ambiguous tasks that barely failed local)

  Heuristic: sort by name length descending. Larger names usually
  contain "70b", "72b", "405b" etc. If only one model is allowed, use it for both.

Every token spent here is COUNTED by the judging proxy. Keep prompts compressed.
"""
import os
import logging
from typing import Optional, Tuple

from .model_client import OpenAICompatibleClient, MockClient

logger = logging.getLogger(__name__)


def _select_models(allowed_str: str) -> tuple[str, str]:
    """
    Given the ALLOWED_MODELS comma-separated string, return
    (strong_model_id, fast_model_id).

    Priority: prefer models containing "70b", "72b", "405b" as strong,
    and "7b", "8b", "3b", "1b" as fast. Falls back to list order.
    """
    models = [m.strip() for m in allowed_str.split(",") if m.strip()]
    if not models:
        raise RuntimeError("ALLOWED_MODELS is empty — cannot select a model.")

    LARGE_KEYWORDS  = ["405b", "141b", "72b", "70b", "65b", "34b", "32b"]
    SMALL_KEYWORDS  = ["1b", "3b", "7b", "8b", "mini", "small", "tiny"]

    def _score(model_id: str, keywords: list) -> int:
        lower = model_id.lower()
        for i, kw in enumerate(keywords):
            if kw in lower:
                return len(keywords) - i  # higher score = better match
        return 0

    strong = max(models, key=lambda m: _score(m, LARGE_KEYWORDS))
    fast   = min(models, key=lambda m: _score(m, SMALL_KEYWORDS) * -1 or len(m))

    # If only one model, use it for both roles
    if len(models) == 1:
        fast = strong

    logger.info(f"[RemoteModel] strong={strong}  fast={fast}")
    return strong, fast


class RemoteModel:
    """
    Two-tier remote model:
      generate()        — uses strong_model (hard tasks)
      generate_fast()   — uses fast_model (marginally-failed local tasks)
    """

    def __init__(self, config: dict):
        self.config = config

        if config.get("use_mock", True):
            self._strong_client = MockClient(label="remote-strong")
            self._fast_client   = MockClient(label="remote-fast")
            self.strong_model   = "mock-strong"
            self.fast_model     = "mock-fast"
            return

        # ── Fireworks env vars injected by harness ──────────────────────────
        api_key  = os.environ.get("FIREWORKS_API_KEY")
        base_url = os.environ.get("FIREWORKS_BASE_URL")
        allowed  = os.environ.get("ALLOWED_MODELS", "")

        # ── Local dev fallback (Groq / whatever is in config) ──────────────
        if not api_key:
            api_key = os.environ.get(config.get("api_key_env", ""), "")
        if not base_url:
            base_url = config.get("base_url", "")
        if not allowed:
            # Use config model for both roles during local dev
            fallback_model = config.get("model", "")
            allowed = fallback_model

        if not api_key:
            raise RuntimeError(
                "No API key found. Set FIREWORKS_API_KEY (harness) or the "
                f"env var named in config ({config.get('api_key_env')})."
            )
        if not base_url:
            raise RuntimeError("No base_url found. Set FIREWORKS_BASE_URL.")

        self.strong_model, self.fast_model = _select_models(allowed)
        timeout = config.get("timeout", 25)  # keep under 30s hard limit

        self._strong_client = OpenAICompatibleClient(
            base_url=base_url, api_key=api_key,
            model=self.strong_model, timeout=timeout,
        )
        self._fast_client = OpenAICompatibleClient(
            base_url=base_url, api_key=api_key,
            model=self.fast_model, timeout=timeout,
        )

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        use_fast_model: bool = False,
        max_tokens: int = 512,
    ) -> Tuple[str, int]:
        """
        Returns (answer_text, tokens_used). tokens_used is what gets scored.
        use_fast_model=True to use the smaller, cheaper model.
        max_tokens limits output length — fewer completion tokens = lower score.
        """
        client = self._fast_client if use_fast_model else self._strong_client
        results = client.generate(
            prompt, n=1, system=system, max_tokens=max_tokens,
        )
        r = results[0]
        return r.text, r.total_tokens
