"""
run.py — Hackathon submission entry point.

Contract (from Participant Guide):
  INPUT:  /input/tasks.json  — list of {task_id, prompt}
  OUTPUT: /output/results.json — list of {task_id, answer}
  EXIT:   0 on success, non-zero on failure

The pipeline:
  1. Load all tasks from /input/tasks.json
  2. For each task: run through RoutingPipeline.handle_query()
     - Local model (Phi-3-mini in container) tries first — FREE
     - If uncertain, escalates to Fireworks via FIREWORKS_BASE_URL — COUNTED
  3. Write results to /output/results.json
  4. Exit 0

Env vars injected by harness:
  FIREWORKS_API_KEY   — required for remote calls
  FIREWORKS_BASE_URL  — all API calls must go through here
  ALLOWED_MODELS      — comma-separated model IDs, read at runtime
"""
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from src.config import load_config
from src.local_model import LocalModel
from src.remote_model import RemoteModel
from src.pipeline import RoutingPipeline
from src.logger import Logger

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # keep stdout clean for debug; results go to file
)
log = logging.getLogger("run")

INPUT_PATH  = Path(os.environ.get("INPUT_PATH",  "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/models.yaml")


def build_pipeline() -> RoutingPipeline:
    config = load_config(CONFIG_PATH)
    use_mock = config["routing"].get("use_mock", False)

    local_config  = {**config["local"],  "use_mock": use_mock}
    remote_config = {**config["remote"], "use_mock": use_mock}

    local_model  = LocalModel(local_config)
    remote_model = RemoteModel(remote_config)
    return RoutingPipeline(local_model, remote_model, config, logger=Logger())


def main() -> int:
    start = time.time()

    # ── Load tasks ────────────────────────────────────────────────────────────
    if not INPUT_PATH.exists():
        log.error(f"Input file not found: {INPUT_PATH}")
        return 1

    try:
        tasks = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse input JSON: {e}")
        return 1

    if not isinstance(tasks, list):
        log.error("Input JSON must be a list of task objects.")
        return 1

    log.info(f"Loaded {len(tasks)} tasks from {INPUT_PATH}")

    # ── Build pipeline ────────────────────────────────────────────────────────
    try:
        pipeline = build_pipeline()
    except Exception as e:
        log.error(f"Failed to build pipeline: {e}\n{traceback.format_exc()}")
        return 1

    # ── Process each task ─────────────────────────────────────────────────────
    results = []
    total_remote_tokens = 0

    for i, task in enumerate(tasks):
        task_id = task.get("task_id", f"unknown_{i}")
        prompt  = task.get("prompt", "")

        if not prompt:
            log.warning(f"Task {task_id} has empty prompt — skipping.")
            results.append({"task_id": task_id, "answer": ""})
            continue

        task_start = time.time()
        try:
            record = pipeline.handle_query(
                query=prompt,
                task_type=task.get("task_type"),   # usually None in submission
                difficulty=task.get("difficulty"), # usually None in submission
            )
            answer = record["final_answer"]
            route  = record["route"]
            tokens = record["remote_tokens_used"]
            total_remote_tokens += tokens

            elapsed = time.time() - task_start
            log.info(
                f"[{i+1}/{len(tasks)}] {task_id} | route={route} | "
                f"type={record['task_type']} | tokens={tokens} | {elapsed:.1f}s"
            )
        except Exception as e:
            log.error(f"Error processing task {task_id}: {e}\n{traceback.format_exc()}")
            answer = ""  # empty answer rather than crashing entire run

        results.append({"task_id": task_id, "answer": answer})

    # ── Write results ─────────────────────────────────────────────────────────
    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        log.error(f"Failed to write output: {e}")
        return 1

    elapsed_total = time.time() - start
    log.info(
        f"Done. {len(results)} results written to {OUTPUT_PATH}. "
        f"Total remote tokens: {total_remote_tokens}. "
        f"Total time: {elapsed_total:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
