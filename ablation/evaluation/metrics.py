"""
Compute evaluation metrics for ablation study results.

Supports:
  - Accuracy: per-vignette binary (did any prediction match any gold term?)
  - ROUGE-L: macro-averaged over vignettes (LCS-based)
  - Token F1: macro-averaged over vignettes (bag-of-words overlap)
  - Term-level: micro-averaged P/R/F1 over matched terms
"""


def compute_metrics(vignette_results, field="diagnosis"):
    """Compute all metrics over vignette results for one field.

    Args:
        vignette_results: list of dicts from compute_vignette_match()
        field: "diagnosis" or "treatment"

    Returns: dict with accuracy, rouge_l, token_f1, and term_level metrics
    """
    # Term-level (micro-averaged)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    n_scored = 0
    tier_breakdown = {}

    # Per-vignette accuracy (binary: at least one TP)
    n_correct = 0

    # Text-level (macro-averaged per vignette)
    rouge_l_scores = []
    token_f1_scores = []

    for result in vignette_results:
        field_result = result[field]

        has_terms = field_result["predicted_terms"] or field_result["gold_terms"]
        if has_terms:
            n_scored += 1

        total_tp += field_result["tp"]
        total_fp += field_result["fp"]
        total_fn += field_result["fn"]

        # Accuracy: vignette is correct if at least one predicted term matched
        if has_terms and field_result["tp"] > 0:
            n_correct += 1

        for tier, count in field_result["tier_breakdown"].items():
            tier_breakdown[tier] = tier_breakdown.get(tier, 0) + count

        # Text metrics (present for every vignette)
        text_m = field_result.get("text_metrics", {})
        if text_m:
            rouge_l_scores.append(text_m["rouge_l"]["f1"])
            token_f1_scores.append(text_m["token_f1"]["f1"])

    # Term-level P/R/F1
    term_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    term_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    term_f1 = (2 * term_precision * term_recall / (term_precision + term_recall)
               if (term_precision + term_recall) > 0 else 0.0)

    # Macro-averaged text metrics
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
    avg_token_f1 = sum(token_f1_scores) / len(token_f1_scores) if token_f1_scores else 0.0

    # Accuracy
    accuracy = n_correct / n_scored if n_scored > 0 else 0.0

    return {
        "n_scored": n_scored,
        "accuracy": accuracy,
        "n_correct": n_correct,
        "rouge_l": avg_rouge_l,
        "token_f1": avg_token_f1,
        "term_level": {
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "precision": term_precision,
            "recall": term_recall,
            "f1": term_f1,
            "tier_breakdown": tier_breakdown,
        },
    }


def compute_all_metrics(vignette_results):
    """Compute metrics for both diagnosis and treatment.

    Returns dict with "diagnosis" and "treatment" metric dicts.
    """
    return {
        "diagnosis": compute_metrics(vignette_results, "diagnosis"),
        "treatment": compute_metrics(vignette_results, "treatment"),
    }
