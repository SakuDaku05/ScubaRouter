import pytest
from src.format_validator import validate as format_validate
from src.verifier import Verifier
from src.router import Router, RoutingDecision
from src.compressor import Compressor

# --- format_validator tests ---
def test_format_validator_math():
    # Valid math: contains numbers, not too much prose
    assert format_validate("math", "Q", "42")[0] == True
    assert format_validate("math", "Q", "The answer is 42.5")[0] == True
    
    # Invalid math: pure prose, no numbers
    assert format_validate("math", "Q", "The answer is clearly stated in the text.")[0] == False
    # Invalid math: too wordy (prose threshold > 30 words per number)
    assert format_validate("math", "Q", "To solve this problem we must first take the derivative of the function using the power rule, then we set it to zero, and we keep doing a lot of other things that take up many words and explain every single step in painstaking detail. The final answer is 5.")[0] == False

def test_format_validator_code():
    # Valid code
    assert format_validate("code", "Q", "def solve(): return 1")[0] == True
    assert format_validate("code", "Q", "```python\nfor i in range(5):\n```")[0] == True
    assert format_validate("code", "Q", "x = [i for i in range(10)]")[0] == True # parseable
    
    # Invalid code
    assert format_validate("code", "Q", "I don't know the answer")[0] == False

def test_format_validator_translation():
    # Valid translation
    assert format_validate("translation", "Translate 'hello'", "bonjour")[0] == True
    
    # Invalid translation (echoing source)
    assert format_validate("translation", "Translate 'hello'", "The translation is hello")[0] == False

from src.verifier import Verifier, _extract_score

# --- verifier._extract_score tests ---
def test_verifier_extract_score():
    # Normal case
    assert _extract_score("SCORE: 8") == 0.8
    assert _extract_score("The SCORE: 10/10") == 1.0
    
    # Malformed cases (graceful fallback, returns 0.5)
    assert _extract_score("SCORE: eleven") == 0.5
    assert _extract_score("SCORE: 999") == 1.0 # Out of range > 10 is capped to 1.0 by max/min
    assert _extract_score("I give it a zero") == 0.5 # Missing SCORE:

# --- router.decide tests ---
def test_router_decide():
    router = Router(threshold=0.7, static_escalate_task_types=["always_escalate"])
    
    # Confidence exactly at threshold -> local
    decision = router.decide(query="Q", task_type="qa", confidence=0.7)
    assert decision.route == "local"
    assert decision.effective_threshold == 0.7
    
    # Confidence below threshold -> escalate
    decision = router.decide(query="Q", task_type="qa", confidence=0.69)
    assert decision.route == "escalate"
    
    # Static override -> escalate even if confidence is 1.0
    decision = router.decide(query="Q", task_type="always_escalate", confidence=1.0)
    assert decision.route == "escalate"
    assert decision.reason == "static_override"

# --- compressor tests ---
def test_compressor():
    compressor = Compressor(None)
    
    # Deterministic strip checks
    query = "  Query \n\n\n with   spaces  "
    compressed = compressor.compress(query)
    assert "Query \n with spaces\n\nOutput ONLY the final answer" in compressed
