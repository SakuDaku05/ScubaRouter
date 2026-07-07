"""
benchmark_cpu.py
================
Benchmarks Gemma 4 E4B Q4_K_M GGUF on CPU-ONLY (n_gpu_layers=0).
This exactly simulates the hackathon judging VM environment.

Tests:
  1. Model load time  (limit: 60s)
  2. Per-query latency across all 8 hackathon task categories (limit: 30s each)
  3. Answer quality (keyword match vs expected)
  4. Routing recommendation per task type

Run: python benchmark_cpu.py
"""
import json
import time
import sys
from pathlib import Path

GGUF_PATH  = r"C:\Users\seema\.lmstudio\models\lmstudio-community\gemma-4-E4B-it-GGUF\gemma-4-E4B-it-Q4_K_M.gguf"
DATASET    = Path("eval/benchmark_dataset.json")
N_THREADS  = 8    # adjust if your CPU has more cores

# Max tokens per task type (what we send to remote model — kept low)
MAX_TOKENS = {
    "factual_knowledge": 80,
    "math":              32,
    "sentiment":         24,
    "ner":              100,
    "summarization":    100,
    "code":             300,
    "logical_reasoning": 100,
}
DEFAULT_MAX_TOKENS = 80

TIMEOUT_LIMIT = 30   # seconds per request (hackathon rule)
STARTUP_LIMIT = 60   # seconds for container start (hackathon rule)

COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "cyan":   "\033[96m",
}

def c(color, text):
    return f"{COLORS[color]}{text}{COLORS['reset']}"

def check_keywords(answer: str, keywords: list) -> bool:
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)

def main():
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed.")
        sys.exit(1)

    dataset = json.loads(DATASET.read_text())
    task_types = sorted(set(t["task_type"] for t in dataset))

    print(c("bold", "=" * 70))
    print(c("bold", f" CPU BENCHMARK — Gemma 4 E4B Q4_K_M"))
    print(c("bold", f" {len(dataset)} queries across {len(task_types)} categories"))
    print(c("bold", "=" * 70))
    print(f"\nGGUF: {GGUF_PATH}")
    print(f"Threads: {N_THREADS}  |  GPU layers: 0 (CPU-only)\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...", flush=True)
    t_load = time.time()
    llm = Llama(
        model_path=GGUF_PATH,
        n_ctx=2048,
        n_threads=N_THREADS,
        n_gpu_layers=0,    # ← FORCE CPU ONLY
        verbose=False,
    )
    load_time = time.time() - t_load
    load_ok = load_time < STARTUP_LIMIT
    status = c("green", "✅ OK") if load_ok else c("red", "❌ EXCEEDS 60s LIMIT")
    print(f"Model load: {load_time:.1f}s  {status}\n")

    # ── Run all queries ───────────────────────────────────────────────────────
    results_by_type = {tt: [] for tt in task_types}
    all_results = []

    for i, task in enumerate(dataset):
        tt      = task["task_type"]
        prompt  = task["prompt"]
        max_tok = MAX_TOKENS.get(tt, DEFAULT_MAX_TOKENS)
        expected= task.get("expected_keywords", [])

        t0 = time.time()
        out = llm(prompt, max_tokens=max_tok, temperature=0.0, echo=False)
        elapsed = time.time() - t0

        answer = out["choices"][0]["text"].strip()
        prompt_tok     = out["usage"]["prompt_tokens"]
        completion_tok = out["usage"]["completion_tokens"]
        tok_per_sec    = completion_tok / elapsed if elapsed > 0 else 0
        quality_ok     = check_keywords(answer, expected) if expected else None

        time_ok = elapsed < TIMEOUT_LIMIT
        t_icon  = c("green", "✅") if time_ok else c("red", "❌")
        q_icon  = (c("green", "✅") if quality_ok else c("red", "❌")) if quality_ok is not None else c("yellow", "?")

        rec = {
            "task_id": task["task_id"], "task_type": tt,
            "difficulty": task["difficulty"],
            "elapsed": elapsed, "time_ok": time_ok,
            "prompt_tok": prompt_tok, "completion_tok": completion_tok,
            "tok_per_sec": tok_per_sec, "quality_ok": quality_ok,
            "answer": answer,
        }
        results_by_type[tt].append(rec)
        all_results.append(rec)

        print(f"[{i+1:02d}/{len(dataset)}] {t_icon} {q_icon} "
              f"{tt:20s} {task['difficulty']:9s} "
              f"{elapsed:5.1f}s  {tok_per_sec:4.1f}tok/s  "
              f"| {answer[:50]!r}")

    # ── Per-type summary ──────────────────────────────────────────────────────
    print(f"\n{c('bold', '=' * 70)}")
    print(c("bold", " PER-CATEGORY SUMMARY"))
    print(c("bold", "=" * 70))
    print(f"{'Category':22s} {'Avg(s)':>7} {'Max(s)':>7} {'<30s':>6} {'Quality':>8} {'VERDICT':>12}")
    print("-" * 70)

    routing_decisions = {}
    for tt in task_types:
        recs       = results_by_type[tt]
        avg_t      = sum(r["elapsed"] for r in recs) / len(recs)
        max_t      = max(r["elapsed"] for r in recs)
        pass_time  = sum(1 for r in recs if r["time_ok"])
        q_recs     = [r for r in recs if r["quality_ok"] is not None]
        quality_pct= (sum(1 for r in q_recs if r["quality_ok"]) / len(q_recs) * 100) if q_recs else None

        # Routing verdict
        if max_t >= TIMEOUT_LIMIT:
            verdict = c("red", "ALWAYS REMOTE")
            routing_decisions[tt] = "always_remote"
        elif avg_t > 15:
            verdict = c("yellow", "PREFER REMOTE")
            routing_decisions[tt] = "prefer_remote"
        elif quality_pct is not None and quality_pct < 60:
            verdict = c("yellow", "PREFER REMOTE")
            routing_decisions[tt] = "prefer_remote"
        else:
            verdict = c("green", "LOCAL OK ✅")
            routing_decisions[tt] = "local_ok"

        q_str = f"{quality_pct:.0f}%" if quality_pct is not None else "N/A"
        print(f"{tt:22s} {avg_t:>7.1f} {max_t:>7.1f} {pass_time:>3}/{len(recs):>2}  {q_str:>7}  {verdict}")

    # ── Overall ───────────────────────────────────────────────────────────────
    print(f"\n{c('bold', '=' * 70)}")
    print(c("bold", " OVERALL RESULTS"))
    print(c("bold", "=" * 70))
    total_time    = sum(r["elapsed"] for r in all_results)
    failed_time   = [r for r in all_results if not r["time_ok"]]
    q_all         = [r for r in all_results if r["quality_ok"] is not None]
    quality_total = sum(1 for r in q_all if r["quality_ok"])

    print(f"Total queries:       {len(all_results)}")
    print(f"Total CPU time:      {total_time:.1f}s")
    print(f"Model load time:     {load_time:.1f}s")
    print(f"Timeout violations:  {len(failed_time)}  {c('red', str([r['task_id'] for r in failed_time])) if failed_time else c('green', 'none')}")
    print(f"Quality pass rate:   {quality_total}/{len(q_all)} ({quality_total/len(q_all)*100:.0f}%)" if q_all else "")

    print(f"\n{c('bold', ' ROUTING RECOMMENDATIONS FOR DOCKER CONTAINER:')}")
    print("-" * 70)
    for tt, decision in routing_decisions.items():
        if decision == "local_ok":
            print(f"  {c('green', '✅ LOCAL')}  {tt}")
        elif decision == "prefer_remote":
            print(f"  {c('yellow', '⚠ REMOTE')}  {tt}  (quality or latency concern)")
        else:
            print(f"  {c('red', '❌ REMOTE')}  {tt}  (exceeds 30s timeout)")

    print(f"\n{c('bold', ' FINAL VERDICT:')}")
    all_local = all(v == "local_ok" for v in routing_decisions.values())
    any_timeout = any(v == "always_remote" for v in routing_decisions.values())

    if all_local and not any_timeout:
        print(c("green", "  🟢 Gemma 4 E4B Q4_K_M on CPU handles ALL 8 categories locally!"))
        print("     → Bundle GGUF in Docker, use for both inference AND verification.")
        print("     → Fireworks only for low-confidence fallback. Minimum token spend.")
    elif not any_timeout:
        print(c("yellow", "  🟡 Gemma handles most categories but some prefer remote."))
        print("     → Use local for fast categories, Fireworks for slow/low-quality ones.")
    else:
        timed_out = [tt for tt, v in routing_decisions.items() if v == "always_remote"]
        print(c("red", f"  🔴 {timed_out} exceed 30s timeout on CPU."))
        print("     → Hard-escalate these types immediately. Already done in pipeline.py!")
        print(c("green", "  ✅ All other types: local inference works."))

    # Save results
    out_path = Path("eval/benchmark_results.json")
    out_path.write_text(json.dumps({
        "model": "gemma-4-E4B-it-Q4_K_M",
        "n_threads": N_THREADS,
        "load_time_s": round(load_time, 2),
        "routing_decisions": routing_decisions,
        "per_query": [{k: v for k, v in r.items() if k != "answer"} for r in all_results],
    }, indent=2))
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
