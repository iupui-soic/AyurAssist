"""
Generate comparison reports for ablation study results.

Outputs:
  - comparison_table.md  — main results table
  - per_vignette.csv     — row-level detail
  - metrics.json         — machine-readable metrics
"""

import csv
import json
import os

from configs import SUMMARY_DIR


def generate_comparison_table(all_config_metrics, output_dir=SUMMARY_DIR):
    """Generate a markdown comparison table across all configs."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "comparison_table.md")

    lines = []
    lines.append("# Ablation Study Results\n")

    # Primary metrics
    lines.append("## Primary Metrics (over 80 vignettes)\n")
    lines.append("| Config | Dx Accuracy | Dx ROUGE-L | Dx Token-F1 | Tx Accuracy | Tx ROUGE-L | Tx Token-F1 |")
    lines.append("|--------|------------|-----------|-------------|------------|-----------|-------------|")

    for config_name, metrics in all_config_metrics.items():
        d = metrics["diagnosis"]
        t = metrics["treatment"]
        lines.append(
            f"| {config_name} | {d['accuracy']:.3f} ({d['n_correct']}/{d['n_scored']}) | "
            f"{d['rouge_l']:.3f} | {d['token_f1']:.3f} | "
            f"{t['accuracy']:.3f} ({t['n_correct']}/{t['n_scored']}) | "
            f"{t['rouge_l']:.3f} | {t['token_f1']:.3f} |"
        )

    # Term-level metrics
    lines.append("\n## Term-Level Metrics (micro-averaged over terms)\n")
    lines.append("| Config | N | Dx P | Dx R | Dx F1 | Tx P | Tx R | Tx F1 |")
    lines.append("|--------|---|------|------|-------|------|------|-------|")

    for config_name, metrics in all_config_metrics.items():
        d = metrics["diagnosis"]["term_level"]
        t = metrics["treatment"]["term_level"]
        n = metrics["diagnosis"]["n_scored"]
        lines.append(
            f"| {config_name} | {n} | {d['precision']:.3f} | {d['recall']:.3f} | "
            f"{d['f1']:.3f} | {t['precision']:.3f} | {t['recall']:.3f} | "
            f"{t['f1']:.3f} |"
        )

    # Tier breakdown
    lines.append("\n## Match Tier Breakdown (Diagnosis)\n")
    lines.append("| Config | ITA Match | Word Overlap | Fuzzy Match | No Match |")
    lines.append("|--------|-----------|--------------|-------------|----------|")

    for config_name, metrics in all_config_metrics.items():
        tb = metrics["diagnosis"]["term_level"]["tier_breakdown"]
        lines.append(
            f"| {config_name} | {tb.get('ita_match', 0)} | "
            f"{tb.get('word_overlap', 0)} | {tb.get('fuzzy_match', 0)} | "
            f"{tb.get('no_match', 0)} |"
        )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Comparison table: {filepath}")
    return filepath


def generate_per_vignette_csv(all_config_results, output_dir=SUMMARY_DIR):
    """Write per-vignette results for all configs to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "per_vignette.csv")

    rows = []
    for config_name, results in all_config_results.items():
        for idx, match_result, pipe_result in results:
            d = match_result["diagnosis"]
            t = match_result["treatment"]
            d_tm = d.get("text_metrics", {})
            t_tm = t.get("text_metrics", {})
            rows.append({
                "config": config_name,
                "vignette_index": idx,
                "diag_rouge_l": d_tm.get("rouge_l", {}).get("f1", 0),
                "diag_token_f1": d_tm.get("token_f1", {}).get("f1", 0),
                "diag_tp": d["tp"],
                "diag_fp": d["fp"],
                "diag_fn": d["fn"],
                "treat_rouge_l": t_tm.get("rouge_l", {}).get("f1", 0),
                "treat_token_f1": t_tm.get("token_f1", {}).get("f1", 0),
                "treat_tp": t["tp"],
                "treat_fp": t["fp"],
                "treat_fn": t["fn"],
                "diag_predicted": "; ".join(d["predicted_terms"]),
                "diag_gold": "; ".join(d["gold_terms"]),
                "treat_predicted": "; ".join(t["predicted_terms"]),
                "treat_gold": "; ".join(t["gold_terms"]),
                "elapsed_seconds": pipe_result.elapsed_seconds,
                "error": pipe_result.error or "",
            })

    if rows:
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    print(f"  Per-vignette CSV: {filepath}")
    return filepath


def generate_metrics_json(all_config_metrics, output_dir=SUMMARY_DIR):
    """Write metrics.json for machine-readable results."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "metrics.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(all_config_metrics, f, indent=2)

    print(f"  Metrics JSON: {filepath}")
    return filepath


def generate_all_reports(all_config_metrics, all_config_results):
    """Generate all report artifacts."""
    print("\nGenerating reports...")
    generate_comparison_table(all_config_metrics)
    generate_per_vignette_csv(all_config_results)
    generate_metrics_json(all_config_metrics)
    print("Done.\n")
