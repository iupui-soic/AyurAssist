#!/usr/bin/env python3
"""Analyze accuracy from BhashaBench evaluation results."""

import json
import os
from collections import defaultdict

RESULTS_DIR = "results"

MODELS = {
    "AyurParam-2.9B": {
        "dir": "ayurparam-2.9b/bharatgenai__AyurParam",
        "english": "samples_bba_genai_English_2026-02-14T16-16-54.514934.jsonl",
        "hindi": "samples_bba_genai_Hindi_2026-02-14T16-16-54.514934.jsonl",
    },
    "GPT-OSS-120B": {
        "dir": "gpt-oss-120b/openai__gpt-oss-120b",
        "english": "samples_bba_genai_English_2026-02-14T14-04-45.839838.jsonl",
        "hindi": "samples_bba_genai_Hindi_2026-02-14T14-04-45.839838.jsonl",
    },
    "Qwen3-32B": {
        "dir": "qwen3-32b/qwen__qwen3-32b",
        "english": "samples_bba_genai_English_2026-02-14T13-46-57.613332.jsonl",
        "hindi": "samples_bba_genai_Hindi_2026-02-14T13-46-57.613332.jsonl",
    },
}


def load_samples(filepath):
    samples = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def compute_accuracy(samples):
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if s.get("exact_match", 0.0) == 1.0)
    return correct / len(samples)


def analyze_by_field(samples, field):
    groups = defaultdict(list)
    for s in samples:
        key = s.get("doc", {}).get(field, "Unknown")
        groups[key].append(s)
    results = {}
    for key, group in sorted(groups.items()):
        correct = sum(1 for s in group if s.get("exact_match", 0.0) == 1.0)
        results[key] = {
            "correct": correct,
            "total": len(group),
            "accuracy": correct / len(group) if group else 0.0,
        }
    return results


def print_separator(char="=", width=100):
    print(char * width)


def print_header(title):
    print()
    print_separator()
    print(f"  {title}")
    print_separator()


def main():
    all_data = {}

    # Load all samples
    for model_name, info in MODELS.items():
        all_data[model_name] = {}
        for lang in ["english", "hindi"]:
            filepath = os.path.join(RESULTS_DIR, info["dir"], info[lang])
            if os.path.exists(filepath):
                all_data[model_name][lang] = load_samples(filepath)
                print(f"Loaded {len(all_data[model_name][lang]):,} samples for {model_name} ({lang})")
            else:
                print(f"WARNING: File not found: {filepath}")
                all_data[model_name][lang] = []

    # ── Overall Accuracy ──
    print_header("OVERALL ACCURACY")
    print(f"{'Model':<20} {'English':>12} {'Hindi':>12} {'Combined':>12}")
    print("-" * 60)
    for model_name in MODELS:
        en_samples = all_data[model_name]["english"]
        hi_samples = all_data[model_name]["hindi"]
        en_acc = compute_accuracy(en_samples)
        hi_acc = compute_accuracy(hi_samples)
        combined = en_samples + hi_samples
        comb_acc = compute_accuracy(combined) if combined else 0.0
        print(f"{model_name:<20} {en_acc:>11.2%} {hi_acc:>11.2%} {comb_acc:>11.2%}")

    # ── Accuracy by Question Level ──
    print_header("ACCURACY BY QUESTION LEVEL")
    for lang in ["english", "hindi"]:
        print(f"\n  Language: {lang.upper()}")
        levels = set()
        level_data = {}
        for model_name in MODELS:
            level_data[model_name] = analyze_by_field(all_data[model_name][lang], "question_level")
            levels.update(level_data[model_name].keys())

        levels = sorted(levels)
        header = f"  {'Level':<15}"
        for model_name in MODELS:
            header += f" {model_name:>20}"
        print(header)
        print("  " + "-" * (15 + 21 * len(MODELS)))

        for level in levels:
            row = f"  {level:<15}"
            for model_name in MODELS:
                info = level_data[model_name].get(level, {"correct": 0, "total": 0, "accuracy": 0.0})
                row += f" {info['accuracy']:>11.2%} ({info['correct']:>4}/{info['total']:<4})"
            # Truncation-safe: just print accuracy
            pass

        # Re-print with cleaner formatting
        print(f"  {'Level':<12}", end="")
        for model_name in MODELS:
            print(f"  {model_name:>18}", end="")
        print()
        print("  " + "-" * (12 + 20 * len(MODELS)))
        for level in levels:
            print(f"  {level:<12}", end="")
            for model_name in MODELS:
                info = level_data[model_name].get(level, {"correct": 0, "total": 0, "accuracy": 0.0})
                print(f"  {info['accuracy']:>8.2%} ({info['correct']}/{info['total']})", end="")
            print()

    # ── Accuracy by Topic ──
    print_header("ACCURACY BY TOPIC")
    for lang in ["english", "hindi"]:
        print(f"\n  Language: {lang.upper()}")
        topics = set()
        topic_data = {}
        for model_name in MODELS:
            topic_data[model_name] = analyze_by_field(all_data[model_name][lang], "topic")
            topics.update(topic_data[model_name].keys())

        topics = sorted(topics)
        # Print header
        print(f"  {'Topic':<45}", end="")
        for model_name in MODELS:
            print(f"  {model_name:>16}", end="")
        print()
        print("  " + "-" * (45 + 18 * len(MODELS)))

        for topic in topics:
            display_topic = topic[:43] if len(topic) > 43 else topic
            print(f"  {display_topic:<45}", end="")
            for model_name in MODELS:
                info = topic_data[model_name].get(topic, {"correct": 0, "total": 0, "accuracy": 0.0})
                print(f"  {info['accuracy']:>7.2%} ({info['total']:>4})", end="")
            print()

    # ── Accuracy by Subject Domain ──
    print_header("ACCURACY BY SUBJECT DOMAIN")
    for lang in ["english", "hindi"]:
        print(f"\n  Language: {lang.upper()}")
        domains = set()
        domain_data = {}
        for model_name in MODELS:
            domain_data[model_name] = analyze_by_field(all_data[model_name][lang], "subject_domain")
            domains.update(domain_data[model_name].keys())

        domains = sorted(domains)
        print(f"  {'Subject Domain':<55}", end="")
        for model_name in MODELS:
            print(f"  {model_name:>16}", end="")
        print()
        print("  " + "-" * (55 + 18 * len(MODELS)))

        for domain in domains:
            display = domain[:53] if len(domain) > 53 else domain
            print(f"  {display:<55}", end="")
            for model_name in MODELS:
                info = domain_data[model_name].get(domain, {"correct": 0, "total": 0, "accuracy": 0.0})
                print(f"  {info['accuracy']:>7.2%} ({info['total']:>4})", end="")
            print()

    # ── Response Analysis: how often each model produces parseable answers ──
    print_header("RESPONSE PARSEABILITY (filtered_resps analysis)")
    for lang in ["english", "hindi"]:
        print(f"\n  Language: {lang.upper()}")
        print(f"  {'Model':<20} {'Valid A-D':>12} {'Empty/Other':>12} {'Total':>8} {'Parse Rate':>12}")
        print("  " + "-" * 68)
        for model_name in MODELS:
            samples = all_data[model_name][lang]
            valid = 0
            invalid = 0
            for s in samples:
                resp = s.get("filtered_resps", [""])[0] if s.get("filtered_resps") else ""
                if resp in ["A", "B", "C", "D"]:
                    valid += 1
                else:
                    invalid += 1
            total = valid + invalid
            rate = valid / total if total > 0 else 0.0
            print(f"  {model_name:<20} {valid:>12,} {invalid:>12,} {total:>8,} {rate:>11.2%}")

    # ── Answer Distribution ──
    print_header("ANSWER DISTRIBUTION (model predictions vs ground truth)")
    for lang in ["english", "hindi"]:
        print(f"\n  Language: {lang.upper()}")
        # Ground truth distribution
        samples_ref = None
        for model_name in MODELS:
            if all_data[model_name][lang]:
                samples_ref = all_data[model_name][lang]
                break
        if samples_ref:
            gt_dist = defaultdict(int)
            for s in samples_ref:
                gt_dist[s.get("target", "?")] += 1
            print(f"  Ground Truth: ", end="")
            for letter in ["A", "B", "C", "D"]:
                print(f"  {letter}={gt_dist.get(letter, 0):,}", end="")
            print()

        for model_name in MODELS:
            pred_dist = defaultdict(int)
            for s in all_data[model_name][lang]:
                resp = s.get("filtered_resps", [""])[0] if s.get("filtered_resps") else ""
                pred_dist[resp] += 1
            print(f"  {model_name:<20}", end="")
            for letter in ["A", "B", "C", "D"]:
                print(f"  {letter}={pred_dist.get(letter, 0):,}", end="")
            other = sum(v for k, v in pred_dist.items() if k not in ["A", "B", "C", "D"])
            if other:
                print(f"  Other={other:,}", end="")
            print()

    print()
    print_separator()
    print("  Analysis complete.")
    print_separator()


if __name__ == "__main__":
    main()