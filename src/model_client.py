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

from openai import OpenAI, APIError, APITimeoutError


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_error: Optional[str] = None


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

    def generate(self, prompt: str, n: int = 1, system: Optional[str] = None, max_tokens: Optional[int] = None) -> List[GenerationResult]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries + 1):
            try:
                create_kwargs = dict(model=self.model, messages=messages, n=n)
                if max_tokens is not None:
                    create_kwargs["max_tokens"] = max_tokens
                resp = self.client.chat.completions.create(**create_kwargs)
                usage = resp.usage
                num_choices = max(len(resp.choices), 1)
                # Some servers report one usage total for all n choices;
                # split completion tokens evenly as an estimate.
                per_choice_completion = (usage.completion_tokens // num_choices) if usage else 0
                prompt_tokens = usage.prompt_tokens if usage else 0

                results = []
                for choice in resp.choices:
                    results.append(GenerationResult(
                        text=choice.message.content or "",
                        prompt_tokens=prompt_tokens,
                        completion_tokens=per_choice_completion,
                        total_tokens=prompt_tokens + per_choice_completion,
                    ))
                return results
            except (APIError, APITimeoutError, Exception) as e:
                if attempt == self.max_retries:
                    return [GenerationResult(text="", prompt_tokens=0, completion_tokens=0,
                                              total_tokens=0, raw_error=str(e))]
                time.sleep(1.5 * (attempt + 1))
        return []
