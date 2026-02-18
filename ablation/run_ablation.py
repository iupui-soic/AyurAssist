#!/usr/bin/env python3
"""
Ablation study orchestrator.

Runs 3 pipeline configurations against 80 gold-standard clinical vignettes
and produces comparison metrics.

Usage:
    python run_ablation.py                          # all configs, all vignettes
    python run_ablation.py --configs bridge --limit 5
    python run_ablation.py --configs full direct --limit 3
    python run_ablation.py --reeval                 # re-evaluate from cached checkpoints
"""

import argparse
import json
import os
import sys

# Ensure ablation/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gold_standard import load_vignettes, get_gold_terms
from evaluation.matching import compute_vignette_match
from evaluation.metrics import compute_all_metrics
from evaluation.report import generate_all_reports
from terminology_bridge import _get_ita_vocab
from pipelines.base import PipelineResult
from configs import RAW_RESPONSES_DIR


# Order matters: baseline first, then additions
CONFIG_MAP = {
    "direct": "direct_llm",         # Baseline: LLM alone
    "full": "full_pipeline",        # LLM + NER + ITA bridge
    "bridge": "bridge_only",        # NER + ITA only (no LLM)
}


def _get_pipeline(config_key):
    """Lazy-import and instantiate a pipeline by config key."""
    name = CONFIG_MAP[config_key]
    if name == "bridge_only":
        from pipelines.bridge_only import BridgeOnlyPipeline
        return BridgeOnlyPipeline()
    elif name == "full_pipeline":
        from pipelines.full_pipeline import FullPipeline
        return FullPipeline()
    elif name == "direct_llm":
        from pipelines.direct_llm import DirectLLMPipeline
        return DirectLLMPipeline()
    else:
        raise ValueError(f"Unknown config: {name}")


def _load_checkpoint(config_name):
    """Load cached pipeline results from JSONL checkpoint."""
    filepath = os.path.join(RAW_RESPONSES_DIR, f"{config_name}.jsonl")
    if not os.path.exists(filepath):
        return []
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                results.append(PipelineResult(**{
                    k: record[k]
                    for k in PipelineResult.__dataclass_fields__
                    if k in record
                }))
            except (json.JSONDecodeError, TypeError):
                continue
    return results


def _evaluate_config(config_name, pipeline_results, vignettes, ita_vocab):
    """Evaluate a config's pipeline results against gold standard."""
    vignette_match_results = []
    config_result_tuples = []

    for result in pipeline_results:
        v = vignettes[result.vignette_index]
        gold_diag = get_gold_terms(v, "diagnosis")
        gold_treat = get_gold_terms(v, "treatment")

        match_result = compute_vignette_match(
            result.predicted_diagnosis,
            result.predicted_treatment,
            gold_diag,
            gold_treat,
            ita_vocab,
        )
        vignette_match_results.append(match_result)
        config_result_tuples.append((result.vignette_index, match_result, result))

    metrics = compute_all_metrics(vignette_match_results)
    return metrics, config_result_tuples


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIG_MAP.keys()),
        default=list(CONFIG_MAP.keys()),
        help="Which configs to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max vignettes to process per config (default: all 80)",
    )
    parser.add_argument(
        "--reeval",
        action="store_true",
        help="Re-evaluate from cached checkpoints (no API calls)",
    )
    args = parser.parse_args()

    # Load gold standard
    print("Loading gold standard vignettes...")
    vignettes = load_vignettes()
    print(f"  {len(vignettes)} vignettes loaded")

    # Load ITA vocabulary for evaluation
    ita_vocab = _get_ita_vocab()
    print(f"  {len(ita_vocab.terms)} ITA terms loaded\n")

    all_config_metrics = {}
    all_config_results = {}

    for config_key in args.configs:
        config_name = CONFIG_MAP[config_key]
        print(f"\n{'#'*60}")
        print(f"  Running config: {config_name}")
        print(f"{'#'*60}")

        if args.reeval:
            # Load from checkpoints — no pipeline instantiation needed
            pipeline_results = _load_checkpoint(config_name)
            if not pipeline_results:
                print(f"  SKIPPING {config_name}: no checkpoint found")
                continue
            if args.limit:
                pipeline_results = pipeline_results[:args.limit]
            print(f"\n  Loaded {len(pipeline_results)} cached results")
        else:
            try:
                pipeline = _get_pipeline(config_key)
            except ValueError as e:
                print(f"  SKIPPING {config_name}: {e}")
                continue

            pipeline_results = pipeline.run_all(vignettes, limit=args.limit)

        # Evaluate
        metrics, config_result_tuples = _evaluate_config(
            config_name, pipeline_results, vignettes, ita_vocab
        )
        all_config_metrics[config_name] = metrics
        all_config_results[config_name] = config_result_tuples

        # Print summary
        d = metrics["diagnosis"]
        t = metrics["treatment"]
        print(f"\n  {config_name} Results (N={d['n_scored']}):")
        print(f"    Diagnosis — Acc: {d['accuracy']:.3f} ({d['n_correct']}/{d['n_scored']})  ROUGE-L: {d['rouge_l']:.3f}  Token-F1: {d['token_f1']:.3f}")
        print(f"    Treatment — Acc: {t['accuracy']:.3f} ({t['n_correct']}/{t['n_scored']})  ROUGE-L: {t['rouge_l']:.3f}  Token-F1: {t['token_f1']:.3f}")

    # Generate reports
    if all_config_metrics:
        generate_all_reports(all_config_metrics, all_config_results)

        # Print final comparison
        print("\n" + "=" * 70)
        print("  ABLATION STUDY COMPARISON")
        print("=" * 70)
        print(f"\n  {'Config':<20} {'Dx Acc':>7} {'Dx ROUGE-L':>10} {'Dx Tok-F1':>10} {'Tx Acc':>7} {'Tx ROUGE-L':>10} {'Tx Tok-F1':>10}")
        print(f"  {'-'*20} {'-'*7} {'-'*10} {'-'*10} {'-'*7} {'-'*10} {'-'*10}")
        for config_name, metrics in all_config_metrics.items():
            d = metrics["diagnosis"]
            t = metrics["treatment"]
            print(
                f"  {config_name:<20} "
                f"{d['accuracy']:>7.3f} {d['rouge_l']:>10.3f} {d['token_f1']:>10.3f} "
                f"{t['accuracy']:>7.3f} {t['rouge_l']:>10.3f} {t['token_f1']:>10.3f}"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
