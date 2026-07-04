# ScubaRouter — Team Onboarding & Research Guide

> **Hackathon:** AMD Developer Hackathon Act II — Track 1: Token-Efficient Routing Agent  
> **Deadline:** July 11, 2026, 9:30 PM IST  
> **Repo:** https://github.com/SakuDaku05/ScubaRouter  
> **Scoring:** Minimize `remote_tokens_used` while keeping accuracy above threshold

---

## 1. The Core Problem

We have two models available:

| Model | Cost | When to use |
|---|---|---|
| **Local small model** (e.g. Llama 3.1 8B) | **FREE** — tokens not counted | First attempt on every query |
| **Remote large model** (Fireworks AI, e.g. Llama 3.3 70B) | **PAID** — every token counts against our score | Only when local model isn't confident |

**The goal:** Answer as many queries correctly as possible using only the free local model. Only call the expensive remote model when you genuinely need it.

This is the **LLM Cascade pattern** — the same approach used in:
- [FrugalGPT (Chen et al., 2023)](https://arxiv.org/abs/2305.05176)
- [AutoMix (Madaan et al., 2023)](https://arxiv.org/abs/2310.12963)
- [RouteLLM (Ong et al., 2024)](https://github.com/lm-sys/routellm)

---

## 2. How the Pipeline Works

```
Every incoming query goes through this exact sequence:

Query
  │
  ▼
┌─────────────────────────┐
│   Local model (8B)       │  FREE. Always runs first.
│   Generates an answer    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Format Validator        │  FREE. Deterministic rules.
│  (src/format_validator)  │  e.g. "Is a math answer a number?"
└────────────┬────────────┘
             │ Format wrong?  ──────► confidence = 0.0
             │ Format OK?
             ▼
┌─────────────────────────┐
│  MF Pre-check            │  FREE. Fast heuristic.
│  (src/mf_router_strict)  │  "Is this query obviously easy?"
└────────────┬────────────┘
             │ Looks easy?  ──────► skip verifier, confidence = 1.0
             │ Uncertain?
             ▼
┌─────────────────────────┐
│  Self-Verifier           │  FREE. Local model grades itself.
│  (src/verifier.py)       │  Returns confidence score 0.0 → 1.0
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Router (Decision Gate)  │  FREE. The only gatekeeper.
│  (src/router.py)         │  confidence >= threshold? → LOCAL
└────────────┬────────────┘  confidence < threshold?  → ESCALATE
             │
    ┌────────┴────────┐
    ▼                 ▼
LOCAL (free)     ESCALATE
Return answer    │
                 ▼
         ┌────────────────┐
         │  Compressor     │  FREE. Trims prompt before sending.
         │  (src/compressor│  Strips boilerplate, summarizes context.
         └───────┬────────┘
                 ▼
         ┌────────────────┐
         │  Remote model   │  PAID ← this is the hackathon score
         │  (70B via       │  Every token sent + received = cost
         │   Fireworks AI) │
         └────────────────┘
```

---

## 3. Key Files

```
routing-agent/
├── config/
│   └── models.yaml          ← CHANGE THIS at kickoff (model names, threshold)
├── src/
│   ├── pipeline.py           ← The main orchestrator. Start reading here.
│   ├── format_validator.py   ← NEW: deterministic checks before LLM verifier
│   ├── verifier.py           ← Local model grades its own answer (self-check)
│   ├── router.py             ← Decides local vs. escalate
│   ├── compressor.py         ← Shrinks prompts before remote calls
│   ├── mf_router_strict.py   ← Strict pre-checker (disables heuristic skip)
│   ├── mf_router.py          ← Original pre-checker (with heuristic fallback)
│   ├── model_client.py       ← OpenAI-compatible API client (works for Groq + Fireworks)
│   ├── local_model.py        ← Wraps the local model client
│   ├── remote_model.py       ← Wraps the remote model client
│   ├── logger.py             ← JSONL logging of every routing decision
│   └── config.py             ← YAML + .env loader
├── eval/
│   ├── eval_harness.py       ← Run this to test accuracy + token spend
│   └── sample_tasks.jsonl    ← 51 labeled test tasks (easy → very_hard)
├── main.py                   ← CLI entry point
├── requirements.txt
└── .env.example              ← Copy to .env and fill in API keys
```

---

## 4. Quick Start (Testing Today)

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Set up your `.env` file**
```bash
cp .env.example .env
# Fill in your GROQ_API key (get free key at console.groq.com)
```

**Step 3: Verify `config/models.yaml`** is set to:
```yaml
local:
  base_url: "https://api.groq.com/openai/v1"
  model: "llama-3.1-8b-instant"   # Small, free-tier model

remote:
  base_url: "https://api.groq.com/openai/v1"
  model: "llama-3.3-70b-versatile" # Large, expensive model

routing:
  use_mock: false
  verification_threshold: 0.7
```

**Step 4: Run a single query**
```bash
python main.py "What is the capital of France?"
```

**Step 5: Run the full evaluation**
```bash
python eval/eval_harness.py                     # All 51 tasks
python eval/eval_harness.py --random 15         # Quick 15-task sample
python eval/eval_harness.py --difficulty hard   # Only hard tasks
python eval/eval_harness.py --type math         # Only math tasks
```

---

## 5. What We've Built So Far

### ✅ Core Cascade Pipeline
The basic `local → verify → route → (compress) → remote` flow is complete and working end-to-end. Currently using Groq API for both models in testing.

### ✅ Format Validators (`src/format_validator.py`)
Deterministic, zero-cost checks that fire immediately after local generation — **before** the LLM self-grader. If the answer format is structurally wrong, we force `confidence = 0.0` and escalate immediately without waiting for an overconfident self-grade.

| Task Type | What gets checked |
|---|---|
| `math` | Does the answer contain a number? Is it a paragraph instead of a value? |
| `code` | Does it contain `def`, `class`, `return`, `lambda`, or parseable Python? |
| `summarization` | Is it shorter than the source? Is it non-empty? |
| `translation` | Is it non-empty? Does it echo the source verbatim? |
| `qa` | Is it non-empty? Is it just the question repeated? |

### ✅ Strict MF Pre-checker (`src/mf_router_strict.py`)
The original `mf_router.py` heuristic was skipping verification for short queries (marking them "obviously easy"), which caused the router to never escalate even when the local model was wrong. The strict version forces the verifier to run on every query unless RouteLLM is explicitly installed.

> **Note:** The original `mf_router.py` is kept intact — we swap between them by changing the import in `src/router.py`.

### ✅ Eval Harness (`eval/eval_harness.py`)
Full offline evaluation with:
- Per-task routing decision and token count
- Per-difficulty breakdown (easy/normal/hard/very_hard)
- Per-task-type accuracy bars
- Failed task inspector showing what keywords the model missed
- Calibration advice (raise/lower threshold)
- `--random N`, `--seed`, `--difficulty`, `--type` filters

### ✅ Test Task Set (`eval/sample_tasks.jsonl`)
51 labeled tasks with multi-keyword matching (any keyword from the list counts as correct). Covers:
- QA (factual, CS concepts, theory)
- Math (arithmetic → calculus)
- Code (Python functions → async generators)
- Summarization (short → dense technical texts)
- Translation (French, Spanish, German, Japanese)

---

## 6. Current Eval Results (as of Jul 4)

**Full run — 51 tasks, strict verifier on:**

| Metric | Value |
|---|---|
| Accuracy | ~94% |
| Local (free) | ~48/51 |
| Escalated (paid) | ~3/51 |
| Remote tokens used | ~483 |

**What still fails:**
- Translation tasks (local 8B model struggles with non-European languages)
- Very hard math (model gives correct working but wrong final decimal)
- Some very_hard QA where keyword matching is still too strict

---

## 7. Critical Things to Do at Kickoff (July 6)

> **Kickoff is July 6, 9:45 PM IST. The actual local + remote model names are revealed then.**

1. **Update `config/models.yaml`:**
   - Set `local.base_url` to the local model server URL (AMD Developer Cloud)
   - Set `local.model` to the revealed local model name
   - Set `remote.base_url` to `https://api.fireworks.ai/inference/v1`
   - Set `remote.model` to the revealed Fireworks model name
   - Set `remote.api_key_env` to `FIREWORKS_API_KEY`
   - Add `FIREWORKS_API_KEY=your_key` to `.env`

2. **Run `python eval/eval_harness.py` immediately** on any sample tasks provided

3. **Turn on logging** — `logs/run_log.jsonl` records every decision. Start accumulating data immediately.

4. **DO NOT change any code** until you have a working baseline submission. Code correctness first, optimizations second.

---

## 8. Calibrating the Threshold

`routing.verification_threshold` in `config/models.yaml` is the single most important tuning knob.

- **Too high (e.g. 0.95):** Almost everything escalates → high accuracy but maximum token spend
- **Too low (e.g. 0.40):** Almost nothing escalates → low token spend but accuracy may drop

**How to calibrate (Day 3-4):**
```bash
# Run the eval at different thresholds and record results
# Then pick the "knee" — the point where accuracy stops improving but tokens keep rising

# Example sweep (do manually or write a loop):
# threshold=0.60 → accuracy=91%, tokens=120
# threshold=0.70 → accuracy=94%, tokens=250
# threshold=0.80 → accuracy=95%, tokens=580
# threshold=0.85 → accuracy=95%, tokens=900
# → Pick 0.70 (knee of the curve)
```

---

## 9. What "Token Efficient" Actually Means for Scoring

The judges count **only remote tokens** (tokens sent to and received from the Fireworks API).

```
Score = total_remote_tokens_used
        (lower is better, subject to accuracy threshold)
```

This means:
- Running the local model 1000 times → costs NOTHING on the scoreboard
- Running the self-verifier (which calls the local model) → costs NOTHING
- Compressing a prompt using the local model → costs NOTHING
- One single call to Fireworks for 400 tokens → costs 400 points

**Strategy:** Be aggressive about local routing. Even if local accuracy on a category is 85%, that's better than escalating everything and spending tokens.

---

## 10. Key Research Papers & Resources

| Resource | What to Read |
|---|---|
| [FrugalGPT (2023)](https://arxiv.org/abs/2305.05176) | The paper that formalized LLM cascading for cost reduction. Core theory behind this project. |
| [AutoMix (2023)](https://arxiv.org/abs/2310.12963) | Adds POMDP-based routing and few-shot self-verification to cascading. The verifier design is inspired by this. |
| [RouteLLM (2024)](https://github.com/lm-sys/routellm) | Open-source framework with matrix factorization + BERT classifiers for routing. The `mf_router.py` optionally uses this. |
| [Self-Consistency (Wang et al., 2022)](https://arxiv.org/abs/2203.11171) | The paper behind sampling multiple answers and picking by majority. Used in `verifier.py`. |
| [AMD Developer Cloud Docs](https://www.amd.com/en/developer/resources/developer-cloud.html) | Where the actual local model server will run at kickoff |
| [Fireworks AI API Docs](https://readme.fireworks.ai/) | Remote model API reference |
| [Groq API Docs](https://console.groq.com/docs/) | Current test API (swap with Fireworks at kickoff) |

---

## 11. Ideas for Further Improvement (If Time Allows)

These are ranked by feasibility within the 5-day window:

| Idea | Effort | Expected Gain |
|---|---|---|
| **Exact-match cache** — return cached answer for identical repeated queries | 30 min | Low (benchmarks rarely repeat) |
| **Per-task-type thresholds** — different confidence cutoff for math vs QA | 1 hour | Medium |
| **Instruction stripping in compressor** — remove verbose prompt boilerplate before remote call | 1 hour | Medium — saves 30-60 tokens per escalation |
| **Log-probability confidence** — use model token probabilities instead of LLM self-grading | 2 hours | High (much better signal) |
| **Semantic cache** — embed queries, return cached answers for near-duplicate inputs | 4 hours | High if eval has similar queries |
| **Lightweight classifier on logged data** — train logistic regression on (query_embedding → local/escalate) | 1 day | High if enough data accumulates |

---

## 12. File to Change at Kickoff vs. Never Touch

| File | Status |
|---|---|
| `config/models.yaml` | ✅ **CHANGE THIS** — model names, endpoint URLs, threshold |
| `.env` | ✅ **CHANGE THIS** — add FIREWORKS_API_KEY |
| `src/pipeline.py` | ⚠️ Only touch if adding a major new component |
| `src/format_validator.py` | ⚠️ May need to add task-type rules once we see the real eval set |
| `eval/sample_tasks.jsonl` | ⚠️ Replace with real tasks once revealed |
| `src/verifier.py` | ⚠️ May need to tune prompt once we know the task types |
| `src/model_client.py` | 🔒 Do not touch — well-tested |
| `src/compressor.py` | 🔒 Do not touch unless adding instruction stripping |
| `src/router.py` | 🔒 Do not touch — the logic is correct |
| `main.py` | 🔒 Do not touch |
