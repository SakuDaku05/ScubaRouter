"""
Generic OpenAI-compatible chat completion client.

Works for local servers (vLLM, llama.cpp server, text-generation-inference)
AND for Fireworks AI, since both expose an OpenAI-compatible
/chat/completions API. This lets LocalModel and RemoteModel share the same
underlying client, parameterized only by base_url / api_key / model.
"""
import time
from dataclasses import dataclass
from typing import Optional, List

import math
from openai import OpenAI, APIError, APITimeoutError


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_error: Optional[str] = None
    logprobs: Optional[List[float]] = None   # per-token log probs of the completion
    mean_logprob: Optional[float] = None     # mean over completion tokens


class MockClient:
    """Fallback client used when use_mock=True or no real server is up yet.
    Lets you build and test the full pipeline before real model endpoints
    exist -- useful during the pre-kickoff prep days."""

    def __init__(self, label: str = "mock"):
        self.label = label

    def generate(self, prompt: str, n: int = 1, system: Optional[str] = None, max_tokens: Optional[int] = None) -> List[GenerationResult]:
        canned = f"[{self.label} mock answer for]: {prompt[:80]}"
        return [
            GenerationResult(
                text=canned,
                prompt_tokens=len(prompt.split()),
                completion_tokens=10,
                total_tokens=len(prompt.split()) + 10,
            )
            for _ in range(n)
        ]


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 30, max_retries: int = 2):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", timeout=timeout)
        # None = not yet probed; True/False = result of runtime probe
        self._logprobs_supported: Optional[bool] = None

    @property
    def supports_logprobs(self) -> Optional[bool]:
        """Returns True/False if probed, None if not yet attempted."""
        return self._logprobs_supported

    def _probe_logprobs(self, messages: list, max_tokens: Optional[int]) -> Optional[list]:
        """
        Try a single generation with logprobs=True.
        Returns the list of per-token logprob values on success, None on failure.
        Sets self._logprobs_supported accordingly.
        """
        try:
            create_kwargs = dict(
                model=self.model, messages=messages, n=1,
                logprobs=True, top_logprobs=1,
            )
            if max_tokens is not None:
                create_kwargs["max_tokens"] = max_tokens
            resp = self.client.chat.completions.create(**create_kwargs)
            lp_content = resp.choices[0].logprobs
            if lp_content and lp_content.content:
                vals = [t.logprob for t in lp_content.content if t.logprob is not None]
                self._logprobs_supported = True
                return vals or None
            # Server accepted the flag but returned no data
            self._logprobs_supported = False
            return None
        except Exception:
            self._logprobs_supported = False
            return None

    def generate(self, prompt: str, n: int = 1, system: Optional[str] = None, max_tokens: Optional[int] = None) -> List[GenerationResult]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # ── Runtime logprob probe (first call only) ───────────────────────────
        probed_lp_vals: Optional[list] = None
        if self._logprobs_supported is None:
            probed_lp_vals = self._probe_logprobs(messages, max_tokens)
            if probed_lp_vals is not None:
                # Probe succeeded — build result directly from probe response
                mean_lp = sum(probed_lp_vals) / len(probed_lp_vals) if probed_lp_vals else None
                # We still need full usage; re-use the probe call as the actual call
                # (probe already consumed one call — avoid double billing by returning now)
                # We'll re-issue below so n>1 still works. For n=1 return from probe.
                if n == 1:
                    # Re-fetch the text from probe (need a fresh call since probe result is lost)
                    pass  # fall through to normal path — probe already set _logprobs_supported=True

        for attempt in range(self.max_retries + 1):
            try:
                create_kwargs = dict(model=self.model, messages=messages, n=n)
                if max_tokens is not None:
                    create_kwargs["max_tokens"] = max_tokens
                # Include logprobs if we know the server supports them
                if self._logprobs_supported:
                    create_kwargs["logprobs"] = True
                    create_kwargs["top_logprobs"] = 1

                resp = self.client.chat.completions.create(**create_kwargs)
                usage = resp.usage
                num_choices = max(len(resp.choices), 1)
                per_choice_completion = (usage.completion_tokens // num_choices) if usage else 0
                prompt_tokens = usage.prompt_tokens if usage else 0

                results = []
                for choice in resp.choices:
                    lp_vals: Optional[List[float]] = None
                    mean_lp: Optional[float] = None
                    if self._logprobs_supported and choice.logprobs and choice.logprobs.content:
                        lp_vals = [t.logprob for t in choice.logprobs.content if t.logprob is not None]
                        if lp_vals:
                            mean_lp = sum(lp_vals) / len(lp_vals)
                    results.append(GenerationResult(
                        text=choice.message.content or "",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=per_choice_completion,
                        total_tokens=prompt_tokens + per_choice_completion,
                        logprobs=lp_vals,
                        mean_logprob=mean_lp,
                    ))
                return results
            except (APIError, APITimeoutError, Exception) as e:
                if attempt == self.max_retries:
                    return [GenerationResult(text="", prompt_tokens=0, completion_tokens=0,
                                              total_tokens=0, raw_error=str(e))]
                time.sleep(1.5 * (attempt + 1))
        return []
