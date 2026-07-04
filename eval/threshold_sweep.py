"""
Threshold Sweep — automatically runs the full eval at multiple threshold
values and prints a calibration table so you can pick the optimal setting.

Usage (from project root):
    python eval/threshold_sweep.py                       # global threshold sweep
    python eval/threshold_sweep.py --type math           # sweep for math only
    python eval/threshold_sweep.py --difficulty hard     # sweep on hard tasks
    python eval/threshold_sweep.py --random 20 --seed 1  # quick 20-task sweep
    python eval/threshold_sweep.py --min 0.50 --max 0.90 --step 0.05

The 'knee' is the threshold where accuracy stops improving but tokens
keep rising — that's your optimal setting. Update config/models.yaml
with that value before submission.
"""
import sys
import json
import random
import argparse
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.local_model import LocalModel
from src.remote_model import RemoteModel
from src.pipeline import RoutingPipeline
from src.logger import Logger

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def smart_match(answer: str, expected) -> bool:
    a = answer.strip().lower()
    keywords = [expected] if isinstance(expected, str) else list(expected)
    return any(k.strip().lower() in a for k in keywords)


def load_tasks(
    tasks_path: str,
    n_random: int = 0,
    seed: int = 42,
    filter_type: str = "",
    filter_difficulty: str = "",
) -> list:
    tasks = []
    with open(tasks_path) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if filter_type and t.get("task_type") != filter_type:
                    continue
                if filter_difficulty and t.get("difficulty") != filter_difficulty:
                    continue
                tasks.append(t)
    if n_random and n_random < len(tasks):
        random.seed(seed)
        tasks = random.sample(tasks, n_random)
    return tasks


def build_pipeline_with_threshold(config: dict, threshold: float) -> RoutingPipeline:
    """Build a fresh pipeline with a patched global threshold."""
    patched = json.loads(json.dumps(config))  # deep copy
    patched["routing"]["verification_threshold"] = threshold
    # Also patch all per-type and per-difficulty thresholds to scale proportionally
    base = config["routing"].get("verification_threshold", 0.70)
    delta = threshold - base
    for key in ["per_type_thresholds", "per_difficulty_thresholds"]:
        if key in patched["routing"]:
            for k in patched["routing"][key]:
                patched["routing"][key][k] = round(
                    min(0.99, max(0.10, config["routing"][key][k] + delta)), 3
                )

    use_mock = patched["routing"].get("use_mock", True)
    local_model = LocalModel({**patched["local"], "use_mock": use_mock})
    remote_model = RemoteModel({**patched["remote"], "use_mock": use_mock})
    return RoutingPipeline(local_model, remote_model, patched, logger=Logger(enabled=False))


def sweep(
    tasks_path: str,
    thresholds: list,
    n_random: int,
    seed: int,
    filter_type: str,
    filter_difficulty: str,
):
    print(f"\n{BOLD}Loading tasks...{RESET}")
    tasks = load_tasks(tasks_path, n_random, seed, filter_type, filter_difficulty)
    print(f"Running sweep on {len(tasks)} tasks across {len(thresholds)} thresholds.\n")

    config = load_config()

    results = []
    for threshold in thresholds:
        pipeline = build_pipeline_with_threshold(config, threshold)

        correct = total = escalations = remote_tokens = 0
        for task in tasks:
            total += 1
            result = pipeline.handle_query(
                task["query"],
                task_type=task.get("task_type"),
                difficulty=task.get("difficulty"),
            )
            if smart_match(result["final_answer"], task["expected"]):
                correct += 1
            escalations      += 1 if result["route"] == "escalate" else 0
            remote_tokens    += result["remote_tokens_used"]

        accuracy = correct / total if total else 0
        results.append({
            "threshold":     threshold,
            "accuracy":      accuracy,
            "correct":       correct,
            "total":         total,
            "escalations":   escalations,
            "remote_tokens": remote_tokens,
        })
        status = f"{GREEN}✓{RESET}" if accuracy >= 0.90 else (f"{YELLOW}~{RESET}" if accuracy >= 0.75 else f"{RED}✗{RESET}")
        print(f"  threshold={threshold:.2f}  acc={accuracy:.1%}  esc={escalations}/{total}  tokens={remote_tokens:>5}  {status}")

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{BOLD}{'─'*72}{RESET}")
    print(f"{BOLD}  CALIBRATION SUMMARY{RESET}")
    print(f"{BOLD}{'─'*72}{RESET}")
    print(f"  {'THRESHOLD':>10}  {'ACCURACY':>9}  {'ESCALATED':>10}  {'REMOTE TOKENS':>14}  NOTE")
    print(f"  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*13}  {'─'*25}")

    # Find knee: highest accuracy with fewest tokens
    max_acc = max(r["accuracy"] for r in results)
    knee = None
    for r in results:
        if r["accuracy"] >= max_acc * 0.99:  # within 1% of max accuracy
            if knee is None or r["remote_tokens"] < knee["remote_tokens"]:
                knee = r

    for r in results:
        acc_col = (f"{GREEN}{r['accuracy']:.1%}{RESET}" if r["accuracy"] >= 0.90
                   else f"{RED}{r['accuracy']:.1%}{RESET}")
        tok_col = (f"{YELLOW}{r['remote_tokens']:>13}{RESET}" if r["remote_tokens"] > 0
                   else f"{GREEN}{r['remote_tokens']:>13}{RESET}")
        note = f"{CYAN}{BOLD}<< RECOMMENDED KNEE{RESET}" if r is knee else ""
        print(f"  {r['threshold']:>10.2f}  {acc_col}  {r['escalations']:>5}/{r['total']:<3}  {tok_col}  {note}")

    print(f"\n{BOLD}  Recommendation:{RESET}")
    if knee:
        print(f"  Set verification_threshold: {CYAN}{BOLD}{knee['threshold']:.2f}{RESET} in config/models.yaml")
        print(f"  Expected: accuracy={knee['accuracy']:.1%}, remote_tokens={knee['remote_tokens']}\n")
    print(f"{'─'*72}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold sweep for routing calibration")
    parser.add_argument("--tasks",       default="eval/sample_tasks.jsonl")
    parser.add_argument("--random",      type=int, default=0,    metavar="N")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--type",        default="", dest="task_type")
    parser.add_argument("--difficulty",  default="")
    parser.add_argument("--min",         type=float, default=0.50)
    parser.add_argument("--max",         type=float, default=0.92)
    parser.add_argument("--step",        type=float, default=0.05)
    args = parser.parse_args()

    thresholds = []
    t = args.min
    while t <= args.max + 1e-9:
        thresholds.append(round(t, 2))
        t += args.step

    sweep(
        tasks_path=args.tasks,
        thresholds=thresholds,
        n_random=args.random,
        seed=args.seed,
        filter_type=args.task_type,
        filter_difficulty=args.difficulty,
    )
