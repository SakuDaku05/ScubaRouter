"""
Local model wrapper. Every token generated here scores ZERO in the
hackathon, so this is always the first call for every query.
"""
import os
from typing import List, Optional

from .model_client import OpenAICompatibleClient, MockClient


class LocalModel:
    def __init__(self, config: dict):
        self.config = config
        if config.get("use_mock", True):
            self.client = MockClient(label="local")
        else:
            api_key = os.environ.get(config["api_key_env"], "not-needed")
            self.client = OpenAICompatibleClient(
                base_url=config["base_url"],
                api_key=api_key,
                model=config["model"],
                timeout=config.get("timeout", 30),
            )

    def generate(self, prompt: str, n_samples: int = 1, system: Optional[str] = None) -> List[str]:
        """Returns a list of answer strings. Cost is always 0 for scoring."""
        results = self.client.generate(prompt, n=n_samples, system=system)
        return [r.text for r in results]
