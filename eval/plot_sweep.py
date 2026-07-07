"""
plot_sweep.py
-------------
Visualises the cost-vs-accuracy trade-off of your routing thresholds
against the SupraLabs/Prompt-Routing-Dataset (992 labelled rows).

Produces four plots saved as PNGs in eval/plots/:

  1. accuracy_vs_fer.png      — Pareto frontier: Accuracy vs False Escalation Rate
  2. threshold_heatmap.png    — F1 score heatmap over (complexity_at x code/math flags)
  3. complexity_breakdown.png — Per complexity score: accuracy + escalation rate
  4. domain_accuracy.png      — Accuracy by prompt domain (top 15)

Usage (from project root):
    python eval/plot_sweep.py
    python eval/plot_sweep.py --sample 200   # quick run on 200 rows
    python eval/plot_sweep.py --out reports/ # save to custom directory
"""
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

# ── fix Windows terminal encoding ──────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib
    matplotlib.use("Agg")          # headless — safe on all OSes
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers (mirrors supra_calibration.py — kept here for independence)
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(sample: int = 0, seed: int = 42) -> list:
    from datasets import load_dataset as hf_load  # type: ignore
    print("Loading SupraLabs/Prompt-Routing-Dataset …")
    ds = hf_load("SupraLabs/Prompt-Routing-Dataset", split="train")
    rows = list(ds)
    print(f"  {len(rows)} rows loaded.")
    if sample and sample < len(rows):
        random.seed(seed)
        rows = random.sample(rows, sample)
        print(f"  Sampled {len(rows)} rows (seed={seed}).")
    return rows


def ground_truth(row: dict) -> str:
    rc = str(row.get("routing_choice", "big model")).lower().strip()
    return "local" if rc == "small model" else "escalate"


def our_decision(row: dict, cplx_at: int = 3,
                 esc_code: bool = True, esc_math: bool = True) -> str:
    c = int(row.get("complexity_score", 1))
    code = bool(row.get("coding_task", False))
    math = bool(row.get("math_task", False))
    if c >= cplx_at or (esc_code and code) or (esc_math and math):
        return "escalate"
    return "local"


def compute_metrics(decisions: list, rows: list) -> dict:
    tp = fp = tn = fn = 0
    for dec, row in zip(decisions, rows):
        gt = ground_truth(row)
        pred_esc = dec == "escalate"
        true_esc = gt == "escalate"
        if pred_esc and true_esc:         tp += 1
        elif pred_esc and not true_esc:   fp += 1
        elif not pred_esc and not true_esc: tn += 1
        else:                             fn += 1
    total = tp + fp + tn + fn
    acc  = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    fer  = fp / (fp + tn) if (fp + tn) else 0
    flr  = fn / (fn + tp) if (fn + tp) else 0
    return dict(total=total, tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=acc, precision=prec, recall=rec,
                f1=f1, fer=fer, flr=flr)


# ═══════════════════════════════════════════════════════════════════════════
# Plot 1 — Accuracy vs FER (Pareto frontier)
# ═══════════════════════════════════════════════════════════════════════════

def plot_accuracy_vs_fer(rows: list, out_dir: Path):
    """
    Each point = one (complexity_at, esc_code, esc_math) config.
    X axis = FER (False Escalation Rate) — wasted remote tokens.
    Y axis = Accuracy.
    The ideal corner is top-left (high accuracy, low FER).
    """
    configs = []
    for cplx_at in [2, 3, 4, 5]:
        for esc_code in [True, False]:
            for esc_math in [True, False]:
                decs = [our_decision(r, cplx_at, esc_code, esc_math) for r in rows]
                m = compute_metrics(decs, rows)
                configs.append({
                    "label": f"cplx≥{cplx_at} code={'Y' if esc_code else 'N'} math={'Y' if esc_math else 'N'}",
                    "cplx_at": cplx_at,
                    **m,
                })

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    cmap = plt.cm.plasma
    colors = [cmap(i / (len(configs) - 1)) for i in range(len(configs))]

    for cfg, color in zip(configs, colors):
        ax.scatter(cfg["fer"] * 100, cfg["accuracy"] * 100,
                   s=120, color=color, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(cfg["label"],
                    (cfg["fer"] * 100, cfg["accuracy"] * 100),
                    fontsize=7, color="#cccccc",
                    xytext=(5, 2), textcoords="offset points")

    # Pareto frontier
    sorted_cfg = sorted(configs, key=lambda x: x["fer"])
    pareto = []
    best_acc = -1
    for c in sorted_cfg:
        if c["accuracy"] > best_acc:
            pareto.append(c)
            best_acc = c["accuracy"]
    if len(pareto) > 1:
        px = [c["fer"] * 100 for c in pareto]
        py = [c["accuracy"] * 100 for c in pareto]
        ax.plot(px, py, "--", color="#00ff88", linewidth=1.5,
                label="Pareto frontier", zorder=2)

    # Ideal zone shading
    ax.axhspan(90, 102, xmin=0, xmax=0.1,
               facecolor="#00ff8820", edgecolor="none", label="Ideal zone")

    ax.set_xlim(-2, 102)
    ax.set_ylim(60, 103)
    ax.set_xlabel("False Escalation Rate — FER (%) ← lower = fewer wasted tokens",
                  color="#aaaaaa", fontsize=11)
    ax.set_ylabel("Accuracy (%)", color="#aaaaaa", fontsize=11)
    ax.set_title("Cost vs Accuracy Trade-off\n(Each point = one threshold config)",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.grid(True, color="#333333", linewidth=0.5)
    ax.legend(facecolor="#1a1d27", edgecolor="#444444",
              labelcolor="white", fontsize=9)

    path = out_dir / "accuracy_vs_fer.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 2 — F1 Heatmap over (complexity threshold x flag combos)
# ═══════════════════════════════════════════════════════════════════════════

def plot_f1_heatmap(rows: list, out_dir: Path):
    cplx_levels = [2, 3, 4, 5]
    flag_combos = [
        ("code+math", True,  True),
        ("code only", True,  False),
        ("math only", False, True),
        ("neither",   False, False),
    ]

    matrix = np.zeros((len(cplx_levels), len(flag_combos)))
    acc_matrix = np.zeros_like(matrix)

    for i, cplx_at in enumerate(cplx_levels):
        for j, (_, ec, em) in enumerate(flag_combos):
            decs = [our_decision(r, cplx_at, ec, em) for r in rows]
            m = compute_metrics(decs, rows)
            matrix[i, j]     = m["f1"] * 100
            acc_matrix[i, j] = m["accuracy"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")

    for ax, data, title, fmt in [
        (axes[0], matrix,     "F1 Score (%)",   ".1f"),
        (axes[1], acc_matrix, "Accuracy (%)", ".1f"),
    ]:
        ax.set_facecolor("#1a1d27")
        im = ax.imshow(data, cmap="plasma", vmin=60, vmax=102, aspect="auto")
        ax.set_xticks(range(len(flag_combos)))
        ax.set_xticklabels([f[0] for f in flag_combos], color="#cccccc", fontsize=10)
        ax.set_yticks(range(len(cplx_levels)))
        ax.set_yticklabels([f"cplx≥{c}" for c in cplx_levels], color="#cccccc", fontsize=10)
        ax.set_xlabel("Flag escalation rule", color="#aaaaaa")
        ax.set_ylabel("complexity_escalate_at", color="#aaaaaa")
        ax.set_title(title, color="white", fontweight="bold")
        for i in range(len(cplx_levels)):
            for j in range(len(flag_combos)):
                ax.text(j, i, f"{data[i, j]:{fmt}}",
                        ha="center", va="center",
                        color="white" if data[i, j] < 88 else "black",
                        fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color="#aaaaaa")

    fig.suptitle("Threshold Configuration Heatmap",
                 color="white", fontsize=14, fontweight="bold")
    path = out_dir / "threshold_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 3 — Per-complexity-score breakdown (accuracy + escalation rate)
# ═══════════════════════════════════════════════════════════════════════════

def plot_complexity_breakdown(rows: list, out_dir: Path, cplx_at: int = 3):
    scores = ["1", "2", "3", "4", "5"]
    decs = [our_decision(r, cplx_at) for r in rows]

    acc_vals, esc_vals, total_vals, gt_local_vals = [], [], [], []
    for s in scores:
        subset_rows = [r for r in rows if str(r.get("complexity_score")) == s]
        subset_decs = [d for d, r in zip(decs, rows) if str(r.get("complexity_score")) == s]
        if not subset_rows:
            acc_vals.append(0); esc_vals.append(0)
            total_vals.append(0); gt_local_vals.append(0)
            continue
        m = compute_metrics(subset_decs, subset_rows)
        acc_vals.append(m["accuracy"] * 100)
        esc_vals.append(m["tp"] + m["fp"])     # total escalations
        total_vals.append(m["total"])
        gt_local_vals.append(sum(1 for r in subset_rows if ground_truth(r) == "local"))

    x = np.arange(len(scores))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.patch.set_facecolor("#0f1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#aaaaaa")
        ax.grid(True, axis="y", color="#333333", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    # Top: accuracy bars
    bars1 = ax1.bar(x, acc_vals, color=["#00ff88" if v >= 90 else "#ffaa00" if v >= 70 else "#ff4444"
                                         for v in acc_vals], width=0.5, edgecolor="#222222")
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Accuracy (%)", color="#aaaaaa")
    ax1.set_title(f"Per Complexity Score (complexity_escalate_at={cplx_at})",
                  color="white", fontweight="bold")
    ax1.axhline(90, color="#00ff88", linestyle="--", linewidth=1, label="90% target")
    ax1.legend(facecolor="#1a1d27", edgecolor="#444444", labelcolor="white", fontsize=8)
    for bar, val in zip(bars1, acc_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", color="white", fontsize=10)

    # Bottom: stacked local vs escalated counts
    local_counts = [t - e for t, e in zip(total_vals, esc_vals)]
    ax2.bar(x, local_counts, label="Routed Local (free)",
            color="#00aaff", width=0.5, edgecolor="#222222")
    ax2.bar(x, esc_vals, bottom=local_counts, label="Escalated (paid)",
            color="#ff6600", width=0.5, edgecolor="#222222")
    ax2.set_ylabel("Query Count", color="#aaaaaa")
    ax2.set_xlabel("Complexity Score", color="#aaaaaa")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scores, color="#aaaaaa")
    ax2.legend(facecolor="#1a1d27", edgecolor="#444444", labelcolor="white", fontsize=9)

    fig.tight_layout()
    path = out_dir / "complexity_breakdown.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Plot 4 — Accuracy by domain (top 15)
# ═══════════════════════════════════════════════════════════════════════════

def plot_domain_accuracy(rows: list, out_dir: Path, cplx_at: int = 3):
    decs = [our_decision(r, cplx_at) for r in rows]

    domain_data = defaultdict(lambda: {"correct": 0, "total": 0, "local": 0})
    for dec, row in zip(decs, rows):
        d = row.get("primary_domain", "Unknown")
        gt = ground_truth(row)
        domain_data[d]["total"] += 1
        domain_data[d]["correct"] += (dec == gt)
        domain_data[d]["local"] += (dec == "local")

    # Top 15 by count
    top = sorted(domain_data.items(), key=lambda x: x[1]["total"], reverse=True)[:15]
    domains   = [t[0] for t in top]
    accs      = [t[1]["correct"] / t[1]["total"] * 100 for t in top]
    totals    = [t[1]["total"] for t in top]
    local_pct = [t[1]["local"] / t[1]["total"] * 100 for t in top]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    y = np.arange(len(domains))
    bars = ax.barh(y, accs,
                   color=["#00ff88" if a >= 90 else "#ffaa00" if a >= 70 else "#ff4444"
                           for a in accs],
                   height=0.6, edgecolor="#222222")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d} (n={t})" for d, t in zip(domains, totals)],
                       color="#cccccc", fontsize=9)
    ax.set_xlabel("Accuracy (%)", color="#aaaaaa")
    ax.set_title("Routing Accuracy by Prompt Domain (Top 15)",
                 color="white", fontweight="bold", fontsize=13)
    ax.axvline(90, color="#00ff88", linestyle="--", linewidth=1, label="90% target")
    ax.set_xlim(0, 110)
    ax.tick_params(colors="#aaaaaa")
    ax.legend(facecolor="#1a1d27", edgecolor="#444444", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.grid(True, axis="x", color="#333333", linewidth=0.5)

    for bar, val in zip(bars, accs):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", color="white", fontsize=9)

    fig.tight_layout()
    path = out_dir / "domain_accuracy.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot routing sweep graphs")
    parser.add_argument("--sample",  type=int, default=0,
                        help="Random sample N rows (0 = all 992)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--cplx-at", type=int, default=3,
                        help="complexity_escalate_at for breakdown plots (default 3)")
    parser.add_argument("--out",     default="eval/plots",
                        help="Output directory for PNG files")
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(sample=args.sample, seed=args.seed)

    print("\nGenerating plots …")
    plot_accuracy_vs_fer(rows, out_dir)
    plot_f1_heatmap(rows, out_dir)
    plot_complexity_breakdown(rows, out_dir, cplx_at=args.cplx_at)
    plot_domain_accuracy(rows, out_dir, cplx_at=args.cplx_at)

    print(f"\nAll plots saved to: {out_dir.resolve()}")
    print("Files:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
