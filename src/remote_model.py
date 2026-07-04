"""
Remote model wrapper (Fireworks AI). Every token spent here counts
against the score, so this is only ever called on the escalation path,
and only with an already-compressed prompt.
"""
import os
from typing import Optional, Tuple

from .model_client import OpenAICompatibleClient, MockClient


class RemoteModel:
    def __init__(self, config: dict):
        self.config = config
        if config.get("use_mock", True):
            self.client = MockClient(label="remote")
        else:
            api_key = os.environ.get(config["api_key_env"])
            if not api_key:
                raise RuntimeError(
                    f"Missing {config['api_key_env']} environment variable for the Fireworks API"
                )
            self.client = OpenAICompatibleClient(
                base_url=config["base_url"],
                api_key=api_key,
                model=config["model"],
                timeout=config.get("timeout", 60),
            )

    def generate(self, prompt: str, system: Optional[str] = None) -> Tuple[str, int]:
        """Returns (answer_text, tokens_used). tokens_used is what gets scored."""
        results = self.client.generate(prompt, n=1, system=system)
        r = results[0]
        return r.text, r.total_tokens
