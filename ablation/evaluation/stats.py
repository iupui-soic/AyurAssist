"""
Statistical significance testing for ablation study results.

Provides:
  - McNemar's test for paired binary correctness comparisons
  - Bootstrap confidence intervals for all metrics
"""

import math
import random

from evaluation.metrics import compute_metrics


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------
def mcnemar_test(results_a, results_b, field="diagnosis"):
    """McNemar's test comparing two configs on per-vignette binary correctness.

    Correctness is defined as tp > 0 for the given field.

    Args:
        results_a: list of vignette match dicts (from compute_vignette_match)
        results_b: list of vignette match dicts (same length, same order)
        field: "diagnosis" or "treatment"

    Returns: dict with b, c (discordant counts), chi2, p_value
    """
    assert len(results_a) == len(results_b), (
        f"Result lists must have same length: {len(results_a)} vs {len(results_b)}"
    )

    # b = A correct & B wrong; c = A wrong & B correct
    b = 0  # A correct, B wrong
    c = 0  # A wrong, B correct

    for ra, rb in zip(results_a, results_b):
        a_correct = ra[field]["tp"] > 0
        b_correct = rb[field]["tp"] > 0

        if a_correct and not b_correct:
            b += 1
        elif not a_correct and b_correct:
            c += 1

    # McNemar chi-squared (without continuity correction)
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}

    chi2 = (b - c) ** 2 / (b + c)

    # p-value from chi-squared distribution with 1 df
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = 1.0 - chi2_dist.cdf(chi2, df=1)
    except ImportError:
        # Manual approximation using complementary error function
        # For chi2 with 1 df, p = erfc(sqrt(chi2/2))
        p_value = math.erfc(math.sqrt(chi2 / 2))

    return {"b": b, "c": c, "chi2": chi2, "p_value": p_value}


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(vignette_results, field="diagnosis", metric_name="accuracy",
                 n_bootstrap=10000, seed=42):
    """Compute 95% bootstrap CI for a given metric.

    Resamples vignette indices with replacement and recomputes the metric
    on each bootstrap sample.

    Args:
        vignette_results: list of vignette match dicts
        field: "diagnosis" or "treatment"
        metric_name: key in compute_metrics() output (e.g. "accuracy", "rouge_l")
        n_bootstrap: number of bootstrap iterations
        seed: random seed for reproducibility

    Returns: dict with point_estimate, ci_lower (2.5th), ci_upper (97.5th)
    """
    rng = random.Random(seed)
    n = len(vignette_results)

    # Point estimate
    point = compute_metrics(vignette_results, field)[metric_name]

    # Bootstrap
    boot_values = []
    for _ in range(n_bootstrap):
        sample_indices = [rng.randint(0, n - 1) for _ in range(n)]
        sample = [vignette_results[i] for i in sample_indices]
        val = compute_metrics(sample, field)[metric_name]
        boot_values.append(val)

    boot_values.sort()
    ci_lower = boot_values[int(n_bootstrap * 0.025)]
    ci_upper = boot_values[int(n_bootstrap * 0.975)]

    return {
        "point_estimate": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def compute_all_stats(all_config_results):
    """Compute McNemar tests and bootstrap CIs for all configs.

    Args:
        all_config_results: dict mapping config_name -> list of
            (vignette_index, match_result, pipeline_result) tuples

    Returns: dict with "mcnemar" and "bootstrap_ci" sections
    """
    stats = {"mcnemar": {}, "bootstrap_ci": {}}

    # Extract match results per config (sorted by vignette index for alignment)
    config_match_results = {}
    for config_name, tuples in all_config_results.items():
        sorted_tuples = sorted(tuples, key=lambda t: t[0])
        config_match_results[config_name] = [t[1] for t in sorted_tuples]

    # McNemar: full_pipeline vs direct_llm (if both exist)
    if "full_pipeline" in config_match_results and "direct_llm" in config_match_results:
        full = config_match_results["full_pipeline"]
        direct = config_match_results["direct_llm"]
        if len(full) == len(direct):
            for field in ["diagnosis", "treatment"]:
                key = f"full_vs_direct_{field}"
                stats["mcnemar"][key] = mcnemar_test(full, direct, field)

    # Bootstrap CIs for all configs, fields, and key metrics
    metrics_to_ci = ["accuracy", "rouge_l", "token_f1"]
    for config_name, match_results in config_match_results.items():
        stats["bootstrap_ci"][config_name] = {}
        for field in ["diagnosis", "treatment"]:
            stats["bootstrap_ci"][config_name][field] = {}
            for metric in metrics_to_ci:
                ci = bootstrap_ci(match_results, field, metric)
                stats["bootstrap_ci"][config_name][field][metric] = ci

    return stats
