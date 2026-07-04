"""
Eval harness: runs the pipeline over a labeled batch and reports:
  - Per-task accuracy and routing decision (with difficulty shown)
  - Per-task-type AND per-difficulty accuracy + remote token breakdown
  - Cost model: local tokens = $0, remote tokens = your score
  - Random sampling mode to stress-test a subset

Usage (from the project root):
    python eval/eval_harness.py                          # full run
    python eval/eval_harness.py --random 15             # 15 random tasks
    python eval/eval_harness.py --difficulty very_hard  # filter by difficulty
    python eval/eval_harness.py --type math             # filter by task type
    python eval/eval_harness.py --random 20 --seed 99   # reproducible random run
"""
import json
import sys
import random
import argparse
from pathlib import Path
from collections import defaultdict

# Fix Windows terminal encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).resolve().parents[1]))
from main import build_pipeline  # noqa: E402

# ── ANSI colours ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

DIFFICULTY_ORDER = ["easy", "normal", "hard", "very_hard"]
DIFFICULTY_COLOR = {
    "easy":      "\033[92m",   # green
    "normal":    "\033[96m",   # cyan
    "hard":      "\033[93m",   # yellow
    "very_hard": "\033[91m",   # red
}


# ── helpers ─────────────────────────────────────────────────────────────────

def simple_match(final_answer: str, expected: str) -> bool:
    """Case-insensitive keyword-in-answer check."""
    return expected.strip().lower() in final_answer.strip().lower()


def bar(value: float, width: int = 20, fill: str = "#", empty: str = ".") -> str:
    filled = int(round(value * width))
    return fill * filled + empty * (width - filled)


def sep(char: str = "-", width: int = 78) -> str:
    return char * width


def diff_label(d: str) -> str:
    color = DIFFICULTY_COLOR.get(d, "")
    return f"{color}{d:<10}{RESET}"


def route_label(route: str) -> str:
    if route == "local":
        return f"{GREEN}LOCAL  (free){RESET}"
    return f"{YELLOW}REMOTE (paid){RESET}"


# ── main eval function ───────────────────────────────────────────────────────

def run_eval(
    tasks_path: str = "eval/sample_tasks.jsonl",
    n_random: int = 0,
    seed: int = 42,
    filter_difficulty: str = "",
    filter_type: str = "",
):
    pipeline = build_pipeline()

    # ── load + filter tasks ─────────────────────────────────────────────────
    all_tasks = []
    with open(tasks_path) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if filter_difficulty and t.get("difficulty") != filter_difficulty:
                    continue
                if filter_type and t.get("task_type") != filter_type:
                    continue
                all_tasks.append(t)

    if not all_tasks:
        print(f"{RED}No tasks matched your filters.{RESET}")
        return

    # ── random sampling ─────────────────────────────────────────────────────
    if n_random and n_random < len(all_tasks):
        random.seed(seed)
        tasks = random.sample(all_tasks, n_random)
        mode_label = f"RANDOM SAMPLE ({n_random}/{len(all_tasks)} tasks, seed={seed})"
    else:
        tasks = all_tasks
        mode_label = f"FULL RUN ({len(tasks)} tasks)"

    # sort for display: easy → normal → hard → very_hard
    tasks.sort(key=lambda t: DIFFICULTY_ORDER.index(t.get("difficulty", "normal"))
               if t.get("difficulty") in DIFFICULTY_ORDER else 99)

    # ── accumulators ────────────────────────────────────────────────────────
    total            = len(tasks)
    correct          = 0
    total_remote_tok = 0
    escalations      = 0
    results          = []

    by_type: dict = defaultdict(lambda: {"total": 0, "correct": 0, "remote_tokens": 0, "escalations": 0})
    by_diff: dict = defaultdict(lambda: {"total": 0, "correct": 0, "remote_tokens": 0, "escalations": 0})

    # ── header ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{sep('=')}{RESET}")
    print(f"{BOLD}  ROUTING AGENT EVAL  |  {mode_label}{RESET}")
    print(f"{BOLD}{sep('=')}{RESET}")
    print(
        f"  {'#':<4} {'TYPE':<15} {'DIFF':<12} {'ROUTE':<20} "
        f"{'TOK':>5}  {'OK':>3}  QUERY"
    )
    print(sep())

    # ── run each task ────────────────────────────────────────────────────────
    for i, task in enumerate(tasks, 1):
        query      = task["query"]
        expected   = task["expected"]
        task_type  = task.get("task_type", "unknown")
        difficulty = task.get("difficulty", "normal")

        result = pipeline.handle_query(query, task_type=task_type)

        ok         = simple_match(result["final_answer"], expected)
        remote_tok = result["remote_tokens_used"]
        route      = result["route"]
        confidence = result["confidence"]

        # store for later
        results.append({**task, "ok": ok, "remote_tokens": remote_tok,
                         "route": route, "confidence": confidence,
                         "final_answer": result["final_answer"]})

        correct          += ok
        total_remote_tok += remote_tok
        escalations      += (1 if route == "escalate" else 0)
        bt = by_type[task_type]
        bt["total"] += 1; bt["correct"] += ok
        bt["remote_tokens"] += remote_tok
        bt["escalations"] += (1 if route == "escalate" else 0)
        bd = by_diff[difficulty]
        bd["total"] += 1; bd["correct"] += ok
        bd["remote_tokens"] += remote_tok
        bd["escalations"] += (1 if route == "escalate" else 0)

        ok_str    = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        tok_str   = f"{YELLOW}{remote_tok:>5}{RESET}" if remote_tok > 0 else f"{DIM}{remote_tok:>5}{RESET}"
        q_short   = (query[:42] + "…") if len(query) > 42 else query

        print(
            f"  {i:<4} {task_type:<15} {diff_label(difficulty):<12} "
            f"{route_label(route):<20} {tok_str}  {ok_str}   {q_short}"
        )

    # ── per-difficulty breakdown ─────────────────────────────────────────────
    print(f"\n{BOLD}{sep()}{RESET}")
    print(f"{BOLD}  BREAKDOWN BY DIFFICULTY{RESET}")
    print(sep())
    print(f"  {'DIFFICULTY':<12} {'ACC':>8}  {'ESCALATED':>12}  {'REMOTE TOK':>12}  ACCURACY BAR")
    print(f"  {'-'*12}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*22}")

    for diff in DIFFICULTY_ORDER:
        if diff not in by_diff:
            continue
        s   = by_diff[diff]
        t, c, e, rt = s["total"], s["correct"], s["escalations"], s["remote_tokens"]
        acc = c / t
        dc  = DIFFICULTY_COLOR.get(diff, "")
        acc_color = GREEN if acc >= 0.75 else RED
        print(
            f"  {dc}{diff:<12}{RESET} "
            f"{acc_color}{c}/{t} ({acc:.0%}){RESET}  "
            f"{e:>6}/{t}      "
            f"{rt:>10}    "
            f"{dc}[{bar(acc)}]{RESET}"
        )

    # ── per-type breakdown ───────────────────────────────────────────────────
    print(f"\n{BOLD}{sep()}{RESET}")
    print(f"{BOLD}  BREAKDOWN BY TASK TYPE{RESET}")
    print(sep())
    print(f"  {'TYPE':<15} {'ACC':>8}  {'ESCALATED':>12}  {'REMOTE TOK':>12}  ACCURACY BAR")
    print(f"  {'-'*15}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*22}")

    for ttype, s in sorted(by_type.items()):
        t, c, e, rt = s["total"], s["correct"], s["escalations"], s["remote_tokens"]
        acc = c / t
        acc_color = GREEN if acc >= 0.75 else RED
        print(
            f"  {ttype:<15} "
            f"{acc_color}{c}/{t} ({acc:.0%}){RESET}  "
            f"{e:>6}/{t}      "
            f"{rt:>10}    "
            f"[{bar(acc)}]"
        )

    # ── failures list ────────────────────────────────────────────────────────
    failures = [r for r in results if not r["ok"]]
    if failures:
        print(f"\n{BOLD}{sep()}{RESET}")
        print(f"{BOLD}  {RED}FAILED TASKS ({len(failures)}){RESET}")
        print(sep())
        for r in failures:
            print(f"  [{r.get('difficulty','?'):>9}] [{r.get('task_type','?'):<14}] Q: {r['query'][:60]}")
            print(f"  {DIM}Expected: '{r['expected']}'   Got: '{r['final_answer'][:80]}'{RESET}")
            print()
    else:
        print(f"\n  {GREEN}{BOLD}No failures! All tasks answered correctly.{RESET}")

    # ── overall results ──────────────────────────────────────────────────────
    accuracy     = correct / total if total else 0
    local_routed = total - escalations
    acc_color    = GREEN if accuracy >= 0.75 else RED

    print(f"\n{BOLD}{sep('=')}{RESET}")
    print(f"{BOLD}  OVERALL RESULTS{RESET}")
    print(f"{BOLD}{sep('=')}{RESET}")
    print(f"  Tasks run:            {total}")
    print(f"  Accuracy:             {acc_color}{BOLD}{correct}/{total}  ({accuracy:.1%}){RESET}")
    print(f"  Local  (FREE):        {GREEN}{local_routed}/{total}  — zero tokens{RESET}")
    print(f"  Remote (PAID):        {YELLOW}{escalations}/{total}  — tokens scored{RESET}")
    print(f"\n  {BOLD}Remote tokens used:   {YELLOW}{total_remote_tok}{RESET}  {BOLD}<-- HACKATHON SCORE (lower = better){RESET}")
    print(f"  Local  tokens used:   {GREEN}[not counted — infinite free budget]{RESET}")

    # ── calibration advice ───────────────────────────────────────────────────
    efficiency = (local_routed / total * 100) if total else 0
    print(f"\n{BOLD}  CALIBRATION ADVICE{RESET}")
    print(sep())
    print(f"  Local routing rate:   {efficiency:.1f}%")
    print(f"  Avg remote tok/esc:   {(total_remote_tok / escalations):.0f}" if escalations else "  Avg remote tok/esc:   N/A (no escalations)")

    if accuracy < 0.70:
        print(f"\n  {RED}⚠  ACCURACY DANGER: below 70%.{RESET}")
        print(f"     → Raise verification_threshold in config/models.yaml (try +0.05)")
        print(f"     → Check if very_hard tasks need static_escalate override")
    elif accuracy >= 0.90 and escalations > 0:
        print(f"\n  {YELLOW}~  Accuracy strong but some escalations happened.{RESET}")
        print(f"     → Try lowering verification_threshold slightly to save more tokens")
    elif accuracy >= 0.90 and escalations == 0:
        print(f"\n  {GREEN}✓  Perfect: high accuracy + zero remote spend.{RESET}")
        print(f"     → Try harder tasks to find the real escalation boundary")
    else:
        print(f"\n  {YELLOW}~  Acceptable. Sweep threshold 0.60-0.85 in 0.05 steps for the knee.{RESET}")

    print(f"{BOLD}{sep('=')}{RESET}\n")

    return {
        "total": total,
        "accuracy": accuracy,
        "escalations": escalations,
        "total_remote_tokens": total_remote_tok,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Routing agent evaluation harness")
    parser.add_argument("--tasks",       default="eval/sample_tasks.jsonl",
                        help="Path to .jsonl task file")
    parser.add_argument("--random",      type=int, default=0, metavar="N",
                        help="Pick N tasks at random (0 = use all tasks)")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--difficulty",  default="",
                        choices=["", "easy", "normal", "hard", "very_hard"],
                        help="Only run tasks of this difficulty")
    parser.add_argument("--type",        default="", dest="task_type",
                        choices=["", "qa", "math", "code", "summarization", "translation"],
                        help="Only run tasks of this type")
    args = parser.parse_args()

    run_eval(
        tasks_path=args.tasks,
        n_random=args.random,
        seed=args.seed,
        filter_difficulty=args.difficulty,
        filter_type=args.task_type,
    )
