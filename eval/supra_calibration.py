"""
supra_calibration.py
--------------------
Evaluates TWO routing strategies side-by-side against the
SupraLabs/Prompt-Routing-Dataset (992 labeled examples):

  1. OUR ROUTER  — the existing complexity-based threshold logic
                   (Router.decide() via your config thresholds)
  2. SUPRA ROUTER — SupraLabs/Supra-Router-51M model inference
                    (loads locally via transformers, greedy decode)

Ground truth label: routing_choice == "small model" → local
                    routing_choice == "big model"   → escalate

Metrics reported per strategy:
  - Overall accuracy vs ground truth
  - Precision / Recall / F1 for "escalate" decisions
  - False Escalation Rate (FER): % of "small model" prompts we escalated
  - False Local Rate   (FLR): % of "big model" prompts we kept local
  - Confusion matrix
  - Per-complexity-score breakdown
  - Per-domain breakdown (top 10)
  - Threshold sweep for OUR ROUTER (finds the knee)
  - Agreement rate between the two systems

Usage (from project root):
    python eval/supra_calibration.py
    python eval/supra_calibration.py --no-supra        # skip model load
    python eval/supra_calibration.py --sample 200      # random 200 rows
    python eval/supra_calibration.py --sweep           # threshold sweep
    python eval/supra_calibration.py --save results.json
"""
import sys
import json
import argparse
import random
import re
from pathlib import Path
from collections import defaultdict

# ── Windows terminal fix ────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ── ANSI colours ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"
MAGENTA = "\033[95m"


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATASET LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_supra_dataset(sample: int = 0, seed: int = 42) -> list:
    """Download SupraLabs/Prompt-Routing-Dataset via HuggingFace datasets."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print(f"{RED}ERROR: `datasets` not installed. Run: pip install datasets{RESET}")
        sys.exit(1)

    print(f"{CYAN}Loading SupraLabs/Prompt-Routing-Dataset from HuggingFace...{RESET}")
    ds = load_dataset("SupraLabs/Prompt-Routing-Dataset", split="train")
    rows = list(ds)
    print(f"  Loaded {len(rows)} rows.")

    if sample and sample < len(rows):
        random.seed(seed)
        rows = random.sample(rows, sample)
        print(f"  Sampled {len(rows)} rows (seed={seed}).")

    return rows


# ═══════════════════════════════════════════════════════════════════════════
# 2. OUR ROUTER — uses the programmatic rule that mirrors our pipeline
#    (no local model needed; we replicate the threshold logic directly)
# ═══════════════════════════════════════════════════════════════════════════

def our_router_decision(row: dict, complexity_escalate_at: int = 3,
                         always_escalate_code: bool = True,
                         always_escalate_math: bool = True) -> str:
    """
    Mirrors the routing rule our pipeline would apply at the pre-check stage,
    using the ground-truth metadata fields from the dataset row.

    Returns "escalate" or "local".

    Configurable parameters let the calibration sweep find the optimal boundary:
      complexity_escalate_at : escalate if complexity_score >= this (default 3)
      always_escalate_code   : always escalate if coding_task == True
      always_escalate_math   : always escalate if math_task == True
    """
    complexity = int(row.get("complexity_score", 1))
    is_code    = bool(row.get("coding_task", False))
    is_math    = bool(row.get("math_task", False))

    if complexity >= complexity_escalate_at:
        return "escalate"
    if always_escalate_code and is_code:
        return "escalate"
    if always_escalate_math and is_math:
        return "escalate"
    return "local"


# ═══════════════════════════════════════════════════════════════════════════
# 3. SUPRA ROUTER — loads the actual 51M model and runs inference
# ═══════════════════════════════════════════════════════════════════════════

class SupraRouter:
    MODEL_ID = "SupraLabs/Supra-Router-51M"
    # Pattern to extract the Route token from the structured output
    ROUTE_RE = re.compile(r"Route:\s*(small model|big model)", re.IGNORECASE)

    def __init__(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            print(f"{RED}ERROR: transformers/torch not installed.{RESET}")
            print("Run: pip install transformers torch")
            sys.exit(1)

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"{CYAN}Loading {self.MODEL_ID} (51M params)...{RESET}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.torch = torch
        print(f"  {GREEN}Supra-Router loaded.{RESET}")

    def predict(self, prompt: str) -> tuple:
        """
        Returns (decision: str, raw_output: str)
        decision is "local" or "escalate"
        """
        formatted = f"Task: {prompt}\nAnalysis: "
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        match = self.ROUTE_RE.search(raw)
        if match:
            route_token = match.group(1).lower()
            decision = "local" if route_token == "small model" else "escalate"
        else:
            # Fallback: if model didn't emit a parseable route, default to escalate
            decision = "escalate"

        return decision, raw

    def predict_batch(self, rows: list, verbose: bool = True) -> list:
        """Run inference on all rows, return list of (decision, raw_output)."""
        results = []
        total = len(rows)
        for i, row in enumerate(rows, 1):
            dec, raw = self.predict(row["prompt"])
            results.append((dec, raw))
            if verbose and i % 50 == 0:
                print(f"  Supra inference: {i}/{total}...")
        return results


# ═══════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════════════════

def ground_truth(row: dict) -> str:
    """Returns "local" or "escalate" from the dataset label."""
    rc = str(row.get("routing_choice", "big model")).lower().strip()
    return "local" if rc == "small model" else "escalate"


def compute_metrics(decisions: list, rows: list) -> dict:
    """
    decisions : list of "local"/"escalate" strings (one per row)
    rows      : original dataset rows (for ground truth)
    """
    tp = fp = tn = fn = 0   # escalate = positive class

    for dec, row in zip(decisions, rows):
        gt = ground_truth(row)
        pred_esc = dec == "escalate"
        true_esc = gt  == "escalate"

        if pred_esc and true_esc:   tp += 1
        elif pred_esc and not true_esc: fp += 1
        elif not pred_esc and not true_esc: tn += 1
        else:                           fn += 1

    total  = tp + fp + tn + fn
    acc    = (tp + tn) / total if total else 0
    prec   = tp / (tp + fp) if (tp + fp) else 0
    rec    = tp / (tp + fn) if (tp + fn) else 0
    f1     = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    fer    = fp / (fp + tn) if (fp + tn) else 0   # false escalation rate
    flr    = fn / (fn + tp) if (fn + tp) else 0   # false local rate

    return {
        "total": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "fer": fer, "flr": flr,
    }


def per_complexity_breakdown(decisions: list, rows: list) -> dict:
    """Accuracy per complexity_score (1-5)."""
    buckets = defaultdict(lambda: {"correct": 0, "total": 0, "escalated": 0})
    for dec, row in zip(decisions, rows):
        c = str(row.get("complexity_score", "?"))
        gt = ground_truth(row)
        buckets[c]["total"] += 1
        buckets[c]["correct"] += (dec == gt)
        buckets[c]["escalated"] += (dec == "escalate")
    return dict(buckets)


def per_domain_breakdown(decisions: list, rows: list, top_n: int = 10) -> list:
    """Accuracy per primary_domain, sorted by count desc."""
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for dec, row in zip(decisions, rows):
        d = row.get("primary_domain", "Unknown")
        gt = ground_truth(row)
        buckets[d]["total"] += 1
        buckets[d]["correct"] += (dec == gt)
    # Sort by total descending, take top N
    sorted_domains = sorted(buckets.items(), key=lambda x: x[1]["total"], reverse=True)
    return sorted_domains[:top_n]


# ═══════════════════════════════════════════════════════════════════════════
# 5. PRINTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def sep(char="─", width=78): return char * width

def bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)

def pct(v: float) -> str: return f"{v:.1%}"

def color_acc(v: float) -> str:
    c = GREEN if v >= 0.85 else (YELLOW if v >= 0.70 else RED)
    return f"{c}{pct(v)}{RESET}"


def print_metrics_block(label: str, m: dict, color: str = CYAN):
    print(f"\n{BOLD}{color}  {label}{RESET}")
    print(f"  {sep()}")
    print(f"  Accuracy  : {color_acc(m['accuracy'])}  ({m['tp']+m['tn']}/{m['total']})")
    print(f"  Precision : {color_acc(m['precision'])}  (of escalations, how many were right)")
    print(f"  Recall    : {color_acc(m['recall'])}  (of true big-model, how many caught)")
    print(f"  F1        : {color_acc(m['f1'])}")
    print(f"  {RED}FER (false escalations) : {pct(m['fer'])}  ← waste remote tokens{RESET}")
    print(f"  {YELLOW}FLR (false locals)      : {pct(m['flr'])}  ← accuracy risk{RESET}")
    print(f"\n  Confusion matrix (escalate = positive):")
    print(f"  {'':12} {'Pred:local':>12} {'Pred:escalate':>14}")
    print(f"  {'True:local':12} {GREEN}{m['tn']:>12}{RESET} {RED}{m['fp']:>14}{RESET}")
    print(f"  {'True:escalate':12} {RED}{m['fn']:>12}{RESET} {GREEN}{m['tp']:>14}{RESET}")


def print_complexity_table(label: str, breakdown: dict, color: str = CYAN):
    print(f"\n{BOLD}{color}  {label} — by Complexity Score{RESET}")
    print(f"  {sep()}")
    print(f"  {'Score':>6}  {'Correct/Total':>14}  {'Accuracy':>10}  {'Escalated':>10}  BAR")
    print(f"  {'─'*6}  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*20}")
    for score in ["1", "2", "3", "4", "5"]:
        if score not in breakdown:
            continue
        b = breakdown[score]
        acc = b["correct"] / b["total"] if b["total"] else 0
        print(
            f"  {score:>6}  {b['correct']:>6}/{b['total']:<7}  "
            f"{color_acc(acc):>10}  {b['escalated']:>10}  "
            f"{color}[{bar(acc)}]{RESET}"
        )


def print_domain_table(label: str, domains: list, color: str = CYAN):
    print(f"\n{BOLD}{color}  {label} — Top Domains{RESET}")
    print(f"  {sep()}")
    print(f"  {'Domain':<30} {'Correct/Total':>14}  {'Accuracy':>10}")
    print(f"  {'─'*30}  {'─'*14}  {'─'*10}")
    for domain, b in domains:
        acc = b["correct"] / b["total"] if b["total"] else 0
        short = domain[:28] if len(domain) > 28 else domain
        print(f"  {short:<30} {b['correct']:>6}/{b['total']:<7}  {color_acc(acc):>10}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. THRESHOLD SWEEP (Our Router only — sweeps complexity_escalate_at)
# ═══════════════════════════════════════════════════════════════════════════

def run_threshold_sweep(rows: list):
    print(f"\n{BOLD}{sep('═')}{RESET}")
    print(f"{BOLD}  THRESHOLD SWEEP — Our Router (complexity_escalate_at){RESET}")
    print(f"{BOLD}{sep('═')}{RESET}")
    print(f"  {'Escalate@':>10}  {'Code?':>6}  {'Math?':>6}  "
          f"{'Accuracy':>9}  {'FER':>7}  {'FLR':>7}  {'F1':>7}  NOTE")
    print(f"  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*20}")

    best = None
    configs = []
    for cplx_at in [2, 3, 4, 5]:
        for esc_code in [True, False]:
            for esc_math in [True, False]:
                decs = [our_router_decision(r, cplx_at, esc_code, esc_math) for r in rows]
                m = compute_metrics(decs, rows)
                configs.append({
                    "cplx_at": cplx_at, "esc_code": esc_code, "esc_math": esc_math,
                    **m,
                })
                if best is None or (m["f1"] > best["f1"]):
                    best = configs[-1]

    # Print sorted by F1
    for c in sorted(configs, key=lambda x: x["f1"], reverse=True)[:12]:
        note = f"{CYAN}{BOLD}<< BEST F1{RESET}" if c is best else ""
        print(
            f"  {c['cplx_at']:>10}  "
            f"{'Y' if c['esc_code'] else 'N':>6}  "
            f"{'Y' if c['esc_math'] else 'N':>6}  "
            f"{color_acc(c['accuracy']):>9}  "
            f"{pct(c['fer']):>7}  "
            f"{pct(c['flr']):>7}  "
            f"{color_acc(c['f1']):>7}  "
            f"{note}"
        )

    if best:
        print(f"\n  {BOLD}Recommended settings for supra_precheck.py:{RESET}")
        print(f"    complexity_escalate_at = {best['cplx_at']}")
        print(f"    always_escalate_code   = {best['esc_code']}")
        print(f"    always_escalate_math   = {best['esc_math']}")
        print(f"    → Accuracy={pct(best['accuracy'])}, F1={pct(best['f1'])}, "
              f"FER={pct(best['fer'])}, FLR={pct(best['flr'])}")

    return best


# ═══════════════════════════════════════════════════════════════════════════
# 7. AGREEMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def print_agreement(our_decs: list, supra_decs: list, rows: list):
    agree_total = agree_correct = disagree_our_right = disagree_supra_right = 0
    for od, sd, row in zip(our_decs, supra_decs, rows):
        gt = ground_truth(row)
        agree = od == sd
        if agree:
            agree_total += 1
            agree_correct += (od == gt)
        else:
            if od == gt:
                disagree_our_right += 1
            elif sd == gt:
                disagree_supra_right += 1

    total = len(rows)
    print(f"\n{BOLD}  AGREEMENT ANALYSIS{RESET}")
    print(f"  {sep()}")
    print(f"  Agree    : {agree_total}/{total} ({agree_total/total:.1%})")
    print(f"  Agree+correct : {agree_correct}/{agree_total} ({agree_correct/agree_total:.1%})" if agree_total else "")
    print(f"  Disagree : {total - agree_total}/{total}")
    print(f"    → Our router right  : {disagree_our_right}")
    print(f"    → Supra right       : {disagree_supra_right}")
    diff = total - agree_total - disagree_our_right - disagree_supra_right
    print(f"    → Both wrong        : {diff}")

    print(f"\n  {BOLD}Insight:{RESET}")
    if disagree_our_right > disagree_supra_right:
        print(f"  {GREEN}Our router wins on disagreements — trust our thresholds.{RESET}")
    elif disagree_supra_right > disagree_our_right:
        print(f"  {YELLOW}Supra wins on disagreements — consider deferring to Supra.{RESET}")
    else:
        print(f"  {CYAN}Tied on disagreements — combine both signals.{RESET}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Calibrate routing vs SupraLabs dataset")
    parser.add_argument("--sample",   type=int, default=0,
                        help="Random sample N rows (0 = all 992)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--no-supra", action="store_true",
                        help="Skip loading the Supra-Router-51M model")
    parser.add_argument("--sweep",    action="store_true",
                        help="Run threshold sweep for our router")
    parser.add_argument("--save",     default="",
                        help="Save JSON results to this path")
    parser.add_argument("--cplx-at", type=int, default=3,
                        help="complexity_escalate_at threshold for our router (default 3)")
    args = parser.parse_args()

    # ── Load dataset ────────────────────────────────────────────────────────
    rows = load_supra_dataset(sample=args.sample, seed=args.seed)

    # ── Our router decisions ────────────────────────────────────────────────
    print(f"\n{CYAN}Computing Our Router decisions...{RESET}")
    our_decs = [our_router_decision(r, args.cplx_at) for r in rows]
    our_m    = compute_metrics(our_decs, rows)
    our_cplx = per_complexity_breakdown(our_decs, rows)
    our_dom  = per_domain_breakdown(our_decs, rows)

    # ── Supra Router decisions ──────────────────────────────────────────────
    supra_decs = None
    supra_m    = None
    if not args.no_supra:
        supra = SupraRouter()
        print(f"\n{CYAN}Running Supra-Router-51M inference on {len(rows)} prompts...{RESET}")
        supra_results = supra.predict_batch(rows, verbose=True)
        supra_decs = [r[0] for r in supra_results]
        supra_m    = compute_metrics(supra_decs, rows)
        supra_cplx = per_complexity_breakdown(supra_decs, rows)
        supra_dom  = per_domain_breakdown(supra_decs, rows)

    # ── Ground truth distribution ───────────────────────────────────────────
    gt_labels = [ground_truth(r) for r in rows]
    n_local   = gt_labels.count("local")
    n_escalate = gt_labels.count("escalate")

    # ════════════════════════════════════════════════════════════════════════
    # REPORT
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n{BOLD}{sep('═')}{RESET}")
    print(f"{BOLD}  SUPRA CALIBRATION REPORT  |  n={len(rows)}{RESET}")
    print(f"{BOLD}{sep('═')}{RESET}")

    print(f"\n{BOLD}  DATASET DISTRIBUTION{RESET}")
    print(f"  {sep()}")
    print(f"  local    (small model) : {n_local:>5}  ({n_local/len(rows):.1%})")
    print(f"  escalate (big model)   : {n_escalate:>5}  ({n_escalate/len(rows):.1%})")

    # Complexity distribution
    cplx_counts = defaultdict(int)
    for r in rows:
        cplx_counts[str(r.get("complexity_score", "?"))] += 1
    print(f"\n  Complexity distribution:")
    for k in ["1", "2", "3", "4", "5"]:
        cnt = cplx_counts.get(k, 0)
        gt_local = sum(1 for r in rows if str(r.get("complexity_score")) == k
                       and ground_truth(r) == "local")
        print(f"    score={k}: {cnt:>4} rows  ({cnt/len(rows):.0%})  "
              f"→ GT local={gt_local}, GT escalate={cnt-gt_local}")

    # ── Our router report ──────────────────────────────────────────────────
    print_metrics_block(f"OUR ROUTER  (complexity_escalate_at={args.cplx_at})", our_m, GREEN)
    print_complexity_table("OUR ROUTER", our_cplx, GREEN)
    print_domain_table("OUR ROUTER", our_dom, GREEN)

    # ── Supra router report ────────────────────────────────────────────────
    if supra_decs is not None:
        print_metrics_block("SUPRA-ROUTER-51M", supra_m, MAGENTA)
        print_complexity_table("SUPRA-ROUTER-51M", supra_cplx, MAGENTA)
        print_domain_table("SUPRA-ROUTER-51M", supra_dom, MAGENTA)

        # ── Side-by-side summary ───────────────────────────────────────────
        print(f"\n{BOLD}{sep('═')}{RESET}")
        print(f"{BOLD}  SIDE-BY-SIDE SUMMARY{RESET}")
        print(f"{BOLD}{sep('═')}{RESET}")
        print(f"  {'Metric':<25} {'Our Router':>12}  {'Supra-51M':>12}  {'Winner':>10}")
        print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*10}")
        metrics_to_compare = [
            ("Accuracy",  "accuracy",  True),
            ("Precision", "precision", True),
            ("Recall",    "recall",    True),
            ("F1",        "f1",        True),
            ("FER ↓",     "fer",       False),   # lower is better
            ("FLR ↓",     "flr",       False),
        ]
        for label, key, higher_better in metrics_to_compare:
            ov = our_m[key]
            sv = supra_m[key]
            if higher_better:
                winner = f"{GREEN}Ours{RESET}" if ov > sv else (f"{MAGENTA}Supra{RESET}" if sv > ov else "Tie")
            else:
                winner = f"{GREEN}Ours{RESET}" if ov < sv else (f"{MAGENTA}Supra{RESET}" if sv < ov else "Tie")
            print(f"  {label:<25} {pct(ov):>12}  {pct(sv):>12}  {winner:>10}")

        print_agreement(our_decs, supra_decs, rows)

    # ── Threshold sweep ────────────────────────────────────────────────────
    if args.sweep:
        best_cfg = run_threshold_sweep(rows)

    # ── Save results ───────────────────────────────────────────────────────
    if args.save:
        output = {
            "n_rows": len(rows),
            "our_router": {**our_m, "cplx_at": args.cplx_at},
        }
        if supra_m:
            output["supra_router"] = supra_m
        Path(args.save).write_text(json.dumps(output, indent=2))
        print(f"\n  {GREEN}Results saved to {args.save}{RESET}")

    # ── Final recommendation ───────────────────────────────────────────────
    print(f"\n{BOLD}{sep('═')}{RESET}")
    print(f"{BOLD}  RECOMMENDATION FOR YOUR PIPELINE{RESET}")
    print(f"{BOLD}{sep('═')}{RESET}")
    print(f"""
  Based on this calibration:

  1. Use Our Router metadata fields (complexity, code, math) as a
     FIRST-PASS gate before running the self-verifier:
       complexity <= 2 AND NOT code AND NOT math → confidence = 1.0 (skip verifier)
       complexity >= 4 OR (code OR math at cplx>=3) → confidence = 0.0 (force escalate)
       else → run self-verifier as normal

  2. If Supra-Router-51M accuracy > Our Router accuracy:
       Load Supra-51M in the container as the pre-screener.
       Use its Route token only for the "definitely local" case (small model, cplx 1-2).
       For ambiguous cases, still run your own verifier.

  3. The FER metric matters most for your hackathon score:
       Lower FER = fewer wasted remote tokens = better leaderboard rank.
       Sacrifice some Recall (FLR) to reduce FER if accuracy stays above threshold.

  Run with --sweep to find the optimal complexity_escalate_at value.
""")
    print(f"{BOLD}{sep('═')}{RESET}\n")


if __name__ == "__main__":
    main()
