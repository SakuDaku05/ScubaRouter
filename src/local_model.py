import os
import logging
from typing import List, Optional, Tuple

from .model_client import OpenAICompatibleClient, MockClient
from .logprob_confidence import logprobs_to_confidence

logger = logging.getLogger(__name__)

class LocalModel:
    def __init__(self, config: dict):
        self.config = config
        
        if config.get("use_mock", False):
            self.client = MockClient(label="local")
            self.use_api = True
            return

        # Check if we are running with a local GGUF file (e.g. inside Docker)
        # We copied the model into the 'models' directory in the project root.
        self.gguf_path = os.environ.get(
            "LOCAL_GGUF_PATH", 
            "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        )
        
        if os.path.exists(self.gguf_path):
            logger.info(f"[LocalModel] Loading GGUF directly via llama-cpp-python: {self.gguf_path}")
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=self.gguf_path,
                    n_ctx=4096,
                    n_threads=int(os.environ.get("OMP_NUM_THREADS", "8")),
                    n_gpu_layers=0,  # CPU only in the judging VM
                    verbose=False
                )
                self.use_api = False
            except (ImportError, ValueError) as e:
                logger.warning(f"[LocalModel] Could not load GGUF via llama-cpp-python ({e}). Falling back to LM Studio API.")
                self._fallback_to_api(config)
        else:
            logger.info(f"[LocalModel] GGUF not found at {self.gguf_path}. Using API fallback (LM Studio).")
            self._fallback_to_api(config)

    def _fallback_to_api(self, config: dict):
        api_key = os.environ.get(config.get("api_key_env", ""), "not-needed")
        from .model_client import OpenAICompatibleClient
        self.client = OpenAICompatibleClient(
            base_url=config.get("base_url", "http://127.0.0.1:1234/v1"),
            api_key=api_key,
            model=config.get("model", "qwen/qwen3-4b-2507"),
            timeout=config.get("timeout", 30),
        )
        self.use_api = True

    @property
    def supports_logprobs(self) -> bool:
        """True if this client can return log probabilities."""
        if not self.use_api:
            return True   # llama_cpp always supports logprobs
        if hasattr(self, 'client') and isinstance(self.client, OpenAICompatibleClient):
            # None = not yet probed; treat as unknown (return False for safety)
            return bool(self.client._logprobs_supported)
        return False

    def generate(self, prompt: str, n_samples: int = 1) -> List[str]:
        if self.use_api:
            results = self.client.generate(prompt, n=n_samples)
            return [r.text for r in results]
        else:
            results = []
            max_tokens = self.config.get("max_tokens", 256)
            for _ in range(n_samples):
                out = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    echo=False,
                    logprobs=True,   # llama_cpp always supports this
                )
                results.append(out["choices"][0]["text"].strip())
            return results

    def generate_with_logprobs(
        self, prompt: str, n_samples: int = 1
    ) -> List[Tuple[str, Optional[List[float]]]]:
        """
        Like generate(), but also returns per-token log probabilities.
        Returns a list of (text, logprobs_or_None) tuples.
        Falls back gracefully: logprobs element is None when not supported.
        """
        if self.use_api:
            results = self.client.generate(prompt, n=n_samples)
            return [(r.text, r.logprobs) for r in results]
        else:
            output = []
            max_tokens = self.config.get("max_tokens", 256)
            for _ in range(n_samples):
                out = self.llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    echo=False,
                    logprobs=True,
                )
                choice = out["choices"][0]
                text = choice["text"].strip()
                # llama_cpp stores logprobs in choice["logprobs"]["token_logprobs"]
                lp_data = choice.get("logprobs") or {}
                lp_vals = lp_data.get("token_logprobs") or []
                # Filter out None / 0.0 first-token sentinel
                lp_vals = [v for v in lp_vals if v is not None]
                output.append((text, lp_vals if lp_vals else None))
            return output
