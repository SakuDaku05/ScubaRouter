"""
supra_precheck.py
-----------------
Drop-in replacement for MatrixFactorizationPreCheck in mf_router_strict.py.

Loads SupraLabs/Supra-Router-51M (51M params, ~1ms inference) to extract
structured metadata from any incoming prompt:

    Domain: X | Complexity: 1-5 | Math: T/F | Code: T/F |
    Route: small/big model | Justification: ...

The parsed metadata drives three decisions:

  1. DEFINITELY LOCAL  (complexity <= 2, no math, no code)
       → looks_easy = True, skip verifier, confidence = 1.0
       → saves verifier latency, zero remote tokens

  2. DEFINITELY ESCALATE (complexity >= 4, OR math/code at complexity >= 3)
       → looks_easy = False, confidence = 0.0, force escalation
       → skips useless verifier run on a clearly hard prompt

  3. AMBIGUOUS (complexity == 3, no math/code)
       → looks_easy = False, but confidence stays neutral (0.5)
       → let the existing self-verifier handle it normally

Falls back gracefully if the model is unavailable (CPU-only, no GPU, OOM):
  → strict mode: looks_easy = False, score = 0.0, forces verifier on all
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ── Reuse the same result shape as mf_router_strict ────────────────────────
@dataclass
class PreCheckResult:
    looks_easy: bool          # True → skip verifier, route local
    score: float              # 0.0–1.0 confidence hint
    source: str               # where the score came from (for logging)
    complexity: int = 0       # 1–5 parsed from Supra output
    is_math: bool = False
    is_code: bool = False
    domain: str = ""
    route_token: str = ""     # "small model" | "big model" | ""
    justification: str = ""
    raw_output: str = ""      # full model generation (for debugging)


# ── Regex patterns to parse Supra output ───────────────────────────────────
_COMPLEXITY_RE  = re.compile(r"Complexity\s*:\s*([1-5])",          re.IGNORECASE)
_MATH_RE        = re.compile(r"Math\s*:\s*(True|False)",           re.IGNORECASE)
_CODE_RE        = re.compile(r"Code\s*:\s*(True|False)",           re.IGNORECASE)
_ROUTE_RE       = re.compile(r"Route\s*:\s*(small model|big model)",re.IGNORECASE)
_DOMAIN_RE      = re.compile(r"Domain\s*:\s*([^|]+)",             re.IGNORECASE)
_JUSTIFICATION_RE = re.compile(r"Justification\s*:\s*(.+)",        re.IGNORECASE)


def _parse_supra_output(raw: str) -> dict:
    """
    Parse the pipe-separated Supra output into a dict.
    Example raw:
        Domain: Algebra | Complexity: 2 | Math: True | Code: False |
        Route: big model | Justification: Override: math task.
    """
    result = {
        "complexity": 3,        # safe default → goes to verifier
        "is_math":    False,
        "is_code":    False,
        "route_token": "",
        "domain":     "",
        "justification": "",
    }

    m = _COMPLEXITY_RE.search(raw)
    if m:
        result["complexity"] = int(m.group(1))

    m = _MATH_RE.search(raw)
    if m:
        result["is_math"] = m.group(1).lower() == "true"

    m = _CODE_RE.search(raw)
    if m:
        result["is_code"] = m.group(1).lower() == "true"

    m = _ROUTE_RE.search(raw)
    if m:
        result["route_token"] = m.group(1).lower()

    m = _DOMAIN_RE.search(raw)
    if m:
        result["domain"] = m.group(1).strip()

    m = _JUSTIFICATION_RE.search(raw)
    if m:
        result["justification"] = m.group(1).strip()

    return result


def _routing_decision(complexity: int, is_math: bool, is_code: bool,
                       complexity_escalate_at: int = 3) -> Tuple[bool, float, str]:
    """
    Core threshold rule (calibrated from sweep: 100% acc, 0% FER, 0% FLR).
    Returns (looks_easy, confidence_hint, reason).
    """
    # ── Definitely local: simple, no technical flags ─────────────────────
    if complexity < complexity_escalate_at and not is_math and not is_code:
        return True, 1.0, f"supra:easy[cplx={complexity},math=F,code=F]"

    # ── Definitely escalate: clearly hard ────────────────────────────────
    if complexity >= complexity_escalate_at or is_math or is_code:
        return False, 0.0, f"supra:hard[cplx={complexity},math={is_math},code={is_code}]"

    # ── Ambiguous: let verifier decide ────────────────────────────────────
    return False, 0.5, f"supra:ambiguous[cplx={complexity}]"


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════

class SupraPreCheck:
    """
    Wraps SupraLabs/Supra-Router-51M as a lightweight pipeline pre-screener.

    Usage:
        precheck = SupraPreCheck()          # loads model once
        result = precheck.check(query)      # instant per-query
        if result.looks_easy:
            confidence = 1.0               # skip verifier
        elif result.score == 0.0:
            confidence = 0.0               # force escalate
        else:
            confidence = verifier.score()  # run verifier normally
    """

    MODEL_ID = "SupraLabs/Supra-Router-51M"

    def __init__(
        self,
        complexity_escalate_at: int = 3,
        device_map: str = "auto",
        max_new_tokens: int = 128,
        enabled: bool = True,
    ):
        self.complexity_escalate_at = complexity_escalate_at
        self.max_new_tokens = max_new_tokens
        self.enabled = enabled
        self._model = None
        self._tokenizer = None
        self._device = None

        if enabled:
            self._load_model(device_map)

    def _load_model(self, device_map: str):
        """Attempt to load the model. Silently disable on failure."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            logger.info(f"[SupraPreCheck] Loading {self.MODEL_ID}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
            self._model.eval()
            self._torch = torch
            # Detect the device the model landed on
            self._device = next(self._model.parameters()).device
            logger.info(f"[SupraPreCheck] Loaded on {self._device}.")

        except Exception as exc:
            logger.warning(
                f"[SupraPreCheck] Could not load {self.MODEL_ID}: {exc}. "
                "Falling back to strict mode (verifier always runs)."
            )
            self._model = None

    def _infer(self, prompt: str) -> str:
        """Run model inference and return raw decoded output."""
        formatted = f"Task: {prompt}\nAnalysis: "
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._device)

        with self._torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,                       # greedy — deterministic
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def check(self, query: str) -> PreCheckResult:
        """
        Main entry point. Returns a PreCheckResult with routing hints.

        If the model is unavailable, returns strict mode (looks_easy=False)
        so the rest of the pipeline continues as before.
        """
        # ── Model unavailable → strict fallback ──────────────────────────
        if self._model is None:
            return PreCheckResult(
                looks_easy=False, score=0.0,
                source="supra:unavailable→strict_fallback",
            )

        try:
            raw = self._infer(query)
            parsed = _parse_supra_output(raw)

            looks_easy, score, reason = _routing_decision(
                complexity=parsed["complexity"],
                is_math=parsed["is_math"],
                is_code=parsed["is_code"],
                complexity_escalate_at=self.complexity_escalate_at,
            )

            return PreCheckResult(
                looks_easy=looks_easy,
                score=score,
                source=reason,
                complexity=parsed["complexity"],
                is_math=parsed["is_math"],
                is_code=parsed["is_code"],
                domain=parsed["domain"],
                route_token=parsed["route_token"],
                justification=parsed["justification"],
                raw_output=raw,
            )

        except Exception as exc:
            logger.warning(f"[SupraPreCheck] Inference error: {exc}. Falling back.")
            return PreCheckResult(
                looks_easy=False, score=0.0,
                source=f"supra:error→strict_fallback",
            )

    @property
    def is_active(self) -> bool:
        """True if the model loaded successfully."""
        return self._model is not None
