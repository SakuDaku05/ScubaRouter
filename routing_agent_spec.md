# Hybrid Token-Efficient Routing Agent — Architecture & Spec

**Track:** AMD Act II Hackathon — Track 1, Hybrid Token-Efficient Routing Agent
**Kickoff:** July 6, 9:45 PM IST (Introduction to the Challenge) · **Submission deadline:** July 11, 9:30 PM IST
**Judging:** total remote token count + output accuracy (local tokens score zero)

---

## 1. Design principle

Local inference is free on the scoreboard. Remote (Fireworks) inference costs tokens. Accuracy must stay above threshold. Therefore the system should:

1. Run every query on the local model first, at no cost.
2. Decide — using signals generated entirely locally — whether the local answer is trustworthy enough to return as-is.
3. Escalate to Fireworks only when there's real risk the local answer is wrong, and when it does escalate, send the smallest possible prompt.
4. Log everything, so the escalation threshold can be recalibrated fast once real tasks are visible, and so a learned classifier can be trained later if time allows.

This is the LLM cascade pattern (FrugalGPT, AutoMix), not a pre-classification router (RouteLLM/HybridLLM) — the latter needs labeled preference data for your specific task distribution, which doesn't exist until kickoff. Cascading needs no training data and adapts per-query at inference time.

---

## 2. System architecture

```
Query
  │
  ▼
┌─────────────────────┐
│ Local model          │  Runs on every query. Zero cost.
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│ Self-verification     │  Local, cheap. Estimates confidence
└─────────┬────────────┘  in the local answer.
          ▼
┌─────────────────────┐
│ Decision gate          │  confidence vs. calibrated threshold
└──────┬─────────┬─────┘  (+ static override rules)
       │         │
  confident   uncertain
       │         │
       ▼         ▼
 Return local  Compress context
 answer        (strip boilerplate,
 (0 tokens)     local-model summarize)
                    │
                    ▼
              Fireworks remote
              (paid, escalation only)
                    │
       └─────┬──────┘
             ▼
        Logger / eval store
   (route, confidence, tokens, correctness)
```

### Components

| Component | Role | Cost |
|---|---|---|
| Query ingestion | Normalizes incoming task into a common internal format | free |
| Local model runner | Generates first-pass answer for every query | free |
| Self-verification | Entailment-style check: does the answer follow from / correctly address the query? Optionally combined with self-consistency (2-3 local samples) or structural/format validation | free |
| Static override rules | Hardcoded escalation for task types known in advance to be local-model blind spots (e.g. multi-step math if the local model is small) | free |
| Decision gate | Combines verification score + override rules against a calibrated threshold | free |
| Prompt compressor | Deterministic trim (strip boilerplate/redundant instructions/unused few-shot) + local-model extractive summarization of long context, applied only on the escalation path | free (uses local compute) |
| Fireworks remote client | Sends compressed prompt, returns answer | **paid — this is the only thing that costs points** |
| Logger | Records every (query, local answer, verification score, route taken, remote tokens if any, correctness once scoreable) | free |
| Eval harness | Offline scoring: accuracy + total remote token count across a batch, used to calibrate the threshold before submitting | free |
| Classifier (phase 2, optional) | Trained on logged data if there's slack time; used as a fast pre-filter in front of the cascade | free to run, needs upfront labeled data to train |

---

## 3. Interfaces (pseudocode contracts)

Keep every model swappable behind these interfaces — you don't know the real models until kickoff.

```python
class LocalModel:
    def generate(self, prompt: str, n_samples: int = 1) -> list[str]:
        """Runs locally. Returns n_samples completions. Always free."""

class RemoteModel:
    def generate(self, prompt: str) -> tuple[str, int]:
        """Calls Fireworks API. Returns (answer, tokens_used)."""

class Verifier:
    def score(self, query: str, answer: str, context: str = "") -> float:
        """Returns confidence in [0, 1] that `answer` is correct for `query`.
        Implemented as a local-model entailment/self-check prompt,
        optionally blended with self-consistency agreement rate."""

class Router:
    def decide(self, query: str, task_type: str, confidence: float) -> str:
        """Returns 'local' or 'escalate'.
        Applies static override rules first, then threshold check."""

class Compressor:
    def compress(self, prompt: str, context: str = "") -> str:
        """Deterministic strip + optional local-model summarization.
        Only called on the escalate path."""

class Logger:
    def log(self, record: dict) -> None:
        """Appends a structured record — see schema in Section 4."""
```

### Main loop

```python
def handle_query(query, task_type):
    local_answers = local_model.generate(query, n_samples=N_SAMPLES)
    answer = pick_best(local_answers)  # majority vote / first sample
    confidence = verifier.score(query, answer)

    route = router.decide(query, task_type, confidence)

    if route == "local":
        result = answer
        tokens_spent = 0
    else:
        compressed_prompt = compressor.compress(query, context=answer)
        result, tokens_spent = remote_model.generate(compressed_prompt)

    logger.log({
        "query": query,
        "task_type": task_type,
        "local_answer": answer,
        "confidence": confidence,
        "route": route,
        "remote_tokens": tokens_spent,
        "final_answer": result,
    })
    return result
```

---

## 4. Logging schema

Every record should support both day-3 threshold calibration and a possible future classifier. Keep it flat and simple:

```json
{
  "timestamp": "...",
  "query_id": "...",
  "task_type": "string, e.g. qa | math | code | summarization",
  "query_text": "string (or hash if large)",
  "local_answer": "string",
  "verification_score": 0.0,
  "self_consistency_agreement": 0.0,
  "route": "local | escalate",
  "remote_tokens_used": 0,
  "remote_answer": "string or null",
  "final_answer": "string",
  "correct": "true | false | null (fill in once ground truth is known)",
  "latency_ms": 0
}
```

This is your single source of truth for: threshold tuning, post-hoc accuracy/cost analysis, and classifier training data if you get to Phase 2.

---

## 5. Decision gate logic

```python
def decide(query, task_type, confidence):
    if task_type in STATIC_ESCALATE_TYPES:      # known local blind spots
        return "escalate"
    if confidence >= THRESHOLD:
        return "local"
    return "escalate"
```

- `THRESHOLD` is a single float, tuned on your validation set on day 3-4 by sweeping values and plotting accuracy vs. remote-token-spend, picking the knee of the curve.
- `STATIC_ESCALATE_TYPES` is populated only if you discover, empirically, that the local model reliably fails a category — don't guess this in advance, derive it from logged data.
- Bias the threshold slightly conservative early (favor escalation) since underestimating accuracy risk costs more than a few extra remote tokens; tighten it once you trust your verification signal.

---

## 6. Fallback and failure handling

Non-negotiable — a crash on a task scores worse than an unnecessary escalation:

- Local model errors, empty output, or timeout → auto-escalate.
- Fireworks API errors, timeout, or rate limit → retry once with backoff, then return the local answer anyway (best-effort) rather than failing the task.
- Malformed/unparseable output for structured tasks → auto-escalate.
- Every external call wrapped in try/except; nothing should propagate an unhandled exception up to the scoring harness.

---

## 7. Build timeline

**Day 1 (Jul 6 kickoff → next morning)**
- Wire up local model + Fireworks client against real, revealed models.
- Ship the cascade end-to-end (steps 2-4 above) as a working baseline — this is your safety-net submission.
- Turn on logging immediately.

**Day 2-3**
- Accumulate labeled data from cascade runs + any sample tasks you can generate matching the real distribution.
- Build/refine self-verification prompt and self-consistency check.
- (Optional, time permitting) Train the lightweight embedding-based classifier on logged data.

**Day 4**
- Build eval harness comparing cascade-only vs. classifier-assisted vs. hybrid.
- Sweep and lock the threshold on held-out data.
- Stress-test edge cases and failure handling.

**Day 5 (buffer before Jul 11, 9:30 PM)**
- Final integration test as close to the standardized scoring environment as possible.
- Freeze and submit early — don't wait for the deadline in case the standardized environment surfaces surprises.

---

## 8. Evaluation methodology (pre-submission)

Build a local eval script that, given a batch of (query, expected_answer) pairs:

1. Runs the full pipeline.
2. Scores accuracy (exact match / rubric / LLM-judge using the **local** model to stay free).
3. Sums total remote tokens spent.
4. Reports accuracy vs. remote-token-spend, broken down by task type.

Run this after every meaningful change to the verifier or threshold — it's your only signal that you haven't traded away accuracy for cost.

---

## 9. Suggested repo structure

```
routing-agent/
├── config/
│   └── models.yaml          # swappable local/remote model config
├── src/
│   ├── local_model.py
│   ├── remote_model.py
│   ├── verifier.py
│   ├── router.py
│   ├── compressor.py
│   ├── logger.py
│   └── pipeline.py           # handle_query() orchestration
├── eval/
│   ├── eval_harness.py
│   ├── sample_tasks.jsonl    # synthetic tasks for pre-kickoff testing
│   └── calibrate_threshold.py
├── classifier/               # phase 2, optional
│   ├── train.py
│   └── features.py
└── logs/
    └── run_logs.jsonl
```

---

## 10. Explicit non-goals (for this 5-day scope)

- No N-way routing across many models — you have exactly two tiers.
- No pretrained RouteLLM-style classifier as the primary mechanism — insufficient labeled data for your specific model pair and task distribution.
- No heavy third-party compression libraries (e.g. Headroom) as a load-bearing dependency — reimplement the useful idea (deterministic strip + local-model summarization) instead, to avoid new dependency/environment risk on the standardized scoring machine.
- No multi-agent orchestration, RL-trained routers, or latency-based SLO tiering — the judging criteria is tokens + accuracy only.

---

## 11. Open decisions to finalize once models/tasks are revealed

- Exact self-verification prompt template (depends on task types).
- Number of self-consistency samples (tradeoff: more samples = better signal, still free, but adds local latency).
- Initial threshold value before any real data exists (start conservative, e.g. escalate on any confidence below ~0.7, then recalibrate).
- Whether task types can be reliably identified up front (needed for static override rules) or must be inferred from the query itself.
