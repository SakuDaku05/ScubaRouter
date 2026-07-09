# Hybrid token-efficient routing agent — Phase 1

Implements the cascade architecture from the project spec: every query
runs on the local model first (free), gets a training-free self-verification
check, and only escalates to Fireworks (paid) when confidence is genuinely
low. See `routing_agent_spec.md` for the full design rationale.

## File structure

```
routing-agent/
├── config/
│   └── models.yaml          # local/remote endpoints + routing thresholds
├── src/
│   ├── config.py             # YAML + .env loader
│   ├── model_client.py       # generic OpenAI-compatible client (+ mock mode)
│   ├── local_model.py        # free, always called first
│   ├── remote_model.py       # paid, escalation only
│   ├── verifier.py           # self-verification (entailment + self-consistency)
│   ├── mf_router.py          # optional RouteLLM-style pre-check (skip verification only)
│   ├── router.py             # the only component that decides local vs. escalate
│   ├── compressor.py         # trims/summarizes prompts before any remote call
│   ├── logger.py             # JSONL logging for calibration + future classifier data
│   └── pipeline.py           # orchestrates the full cascade
├── eval/
│   ├── sample_tasks.jsonl    # synthetic tasks for pre-kickoff testing
│   └── eval_harness.py       # accuracy + remote-token report
├── classifier/               # phase 2 (optional), empty for now
├── logs/                     # run_log.jsonl gets written here
├── main.py                   # CLI entry point
├── requirements.txt
└── .env.example
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in FIREWORKS_API_KEY once you have it
```

## Running before kickoff (mock mode)

`config/models.yaml` ships with `use_mock: true`, so the entire pipeline
runs end-to-end right now using canned responses — no real model server
needed yet. This is how you test the harness, logging, and eval script
before real models are revealed.

```bash
python main.py "What is the capital of France?"
python eval/eval_harness.py
```

## Switching to real models at kickoff

1. Point `config/models.yaml` → `local.base_url` at your local server
   (e.g. a vLLM or llama.cpp server exposing an OpenAI-compatible API on
   `http://localhost:8000/v1`), and set `local.model` to the real model name.
2. Set `remote.model` to the Fireworks model name revealed at kickoff.
3. Put your real Fireworks key in `.env` as `FIREWORKS_API_KEY`.
4. Flip `routing.use_mock` to `false`.
5. Re-run `python eval/eval_harness.py` to sanity check before submitting.

No code changes needed for any of this — only `config/models.yaml` and `.env`.

## Docker Deployment (Hackathon Submission)

When submitting for the hackathon, your pipeline must run entirely within a Docker container. 

To build the image (Note: this takes a few minutes the first time to download PyTorch and compile dependencies):
```bash
docker build -t routing-agent .
```

To test the container locally exactly as the evaluation harness will:
```bash
# Create mock input/output directories
mkdir -p input output
cp eval/sample_tasks.jsonl input/tasks.json

# Run the container (it automatically executes run.py)
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  -e FIREWORKS_API_KEY="your-api-key" \
  routing-agent
```
The results will be written to `output/results.json`.

## Calibrating the threshold

`routing.verification_threshold` in `config/models.yaml` controls how
confident the local answer must be before it's returned as-is. Lower =
fewer remote tokens but more risk; higher = safer but more escalation.

Once you have real tasks:
1. Run the eval harness across a labeled batch.
2. Sweep the threshold (e.g. 0.5 to 0.9 in steps of 0.05).
3. Plot accuracy vs. total remote tokens.
4. Pick the value at the knee of the curve, biased slightly conservative
   if you're unsure about the local model's reliability on unseen task types.

## Optional: matrix-factorization pre-check

`mf_router.py` can use RouteLLM's pretrained matrix-factorization router
as a fast pre-check to skip the verification step for obviously-easy
queries. This is a latency optimization only — it never skips straight to
remote, since the local model is free and should always be tried first.

To enable:
```bash
pip install routellm
```
Then set `routing.use_mf_precheck: true` in `config/models.yaml`. If the
package or a compatible checkpoint isn't available, it silently falls
back to a lightweight heuristic (query length + complexity keywords) so
the pipeline never breaks.

## Notes on scoring

- Only `remote_tokens_used` (logged per query) counts against your score.
- Local generation, verification, self-consistency checks, and compression
  all run on the free local model and cost nothing on the scoreboard.
- `logs/run_log.jsonl` accumulates every decision made — this is your
  data source for both threshold calibration and a possible phase-2
  classifier (see `classifier/` and the spec doc's Section 3, "Phase 2").
