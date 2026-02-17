#!/usr/bin/env python
"""
Batch-test Qwen3-32B (via Groq API) on translating English Ayurvedic terms
to Sanskrit IAST. Compares against WHO-ITA ground truth.

Usage:
    set GROQ_API_KEY=your_key_here
    python test_qwen_translation.py [--limit N] [--batch-size N]
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata

try:
    from groq import Groq
except ImportError:
    print("Install groq SDK:  pip install groq")
    sys.exit(1)

MODEL = "qwen/qwen3-32b"
GROUND_TRUTH_CSV = "ita_terms.csv"
OUTPUT_CSV = "translation_results.csv"
RAW_RESPONSES_FILE = "raw_model_responses.jsonl"
DEFAULT_BATCH_SIZE = 50


# ── Normalisation helpers ────────────────────────────────────────────────

def strip_diacritics(text: str) -> str:
    """NFKD-decompose then drop combining marks → pure ASCII."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Lowercase ASCII with only alphanumerics and spaces."""
    t = strip_diacritics(text).lower().strip()
    return re.sub(r"[^a-z0-9 ]", "", t)


def parse_terms(field: str) -> list[str]:
    """Split a multi-term Sanskrit field into individual normalised tokens."""
    field = re.sub(r"\d+\.", "", field)          # drop "1.", "2." prefixes
    parts = re.split(r"[;,]", field)
    return [normalize(t) for t in parts if t.strip()]


def check_match(predicted: str, ground_truth: str) -> bool:
    """Any-one match: if ANY predicted sub-term matches ANY GT sub-term."""
    pred_terms = parse_terms(predicted)
    gt_terms = parse_terms(ground_truth)
    for p in pred_terms:
        for g in gt_terms:
            if p and g and (p == g or p in g or g in p):
                return True
    return False


# ── Prompt builder ───────────────────────────────────────────────────────

def build_prompt(batch_terms: list[tuple[str, str]]) -> str:
    lines = [f"{i+1}. [{ita_id}] {english}"
             for i, (ita_id, english) in enumerate(batch_terms)]
    return (
        "You are an expert in Ayurvedic terminology and Sanskrit.\n"
        "Translate each English Ayurvedic / medical term below into its "
        "Sanskrit equivalent using IAST (International Alphabet of Sanskrit "
        "Transliteration).\n\n"
        "Return ONLY a valid JSON array. Each element must have:\n"
        '  "n": <term number>,\n'
        '  "s": "<Sanskrit IAST>"\n\n'
        "If a term has multiple well-known Sanskrit equivalents, return the "
        "most common one.  Do NOT include explanations.\n\n"
        "Terms:\n" + "\n".join(lines)
    )


# ── JSON extraction ─────────────────────────────────────────────────────

def extract_json_array(text: str) -> list[dict]:
    """Robustly pull the first JSON array out of model output."""
    # Strip <think>…</think> block if present (Qwen3 reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return []


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch-test Qwen3-32B translation")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process first N terms (0 = all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Terms per API call (default {DEFAULT_BATCH_SIZE})")
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY environment variable first.")
        sys.exit(1)

    client = Groq(api_key=api_key)

    # Load ground truth (IAST with diacritics)
    rows: list[dict] = []
    with open(GROUND_TRUTH_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if args.limit:
        rows = rows[: args.limit]

    total = len(rows)
    batch_size = args.batch_size
    num_batches = (total + batch_size - 1) // batch_size

    print(f"Terms: {total}  |  Batch size: {batch_size}  |  Batches: {num_batches}")
    print(f"Model: {MODEL}  (reasoning_effort=none)\n")

    all_results: list[dict] = []
    matches = 0
    total_in_tok = 0
    total_out_tok = 0
    api_errors = 0

    # Open raw responses file for appending (one JSON object per line)
    raw_f = open(RAW_RESPONSES_FILE, "w", encoding="utf-8")

    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, total)
        batch = rows[start:end]
        batch_terms = [(r["ITA_ID"], r["English_Term"]) for r in batch]
        prompt = build_prompt(batch_terms)

        print(f"Batch {b+1}/{num_batches}  (terms {start+1}–{end}) ... ", end="", flush=True)

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=4096,
                reasoning_effort="none",      # disable thinking to save tokens
            )

            total_in_tok += resp.usage.prompt_tokens
            total_out_tok += resp.usage.completion_tokens

            raw_content = resp.choices[0].message.content

            # Save raw response for later analysis
            raw_f.write(json.dumps({
                "batch": b + 1,
                "start": start + 1,
                "end": end,
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "raw_response": raw_content,
                "ita_ids": [r["ITA_ID"] for r in batch],
            }, ensure_ascii=False) + "\n")
            raw_f.flush()

            translations = extract_json_array(raw_content)
            trans_map: dict[int, str] = {t["n"]: t["s"] for t in translations}

            batch_matches = 0
            for j, row in enumerate(batch):
                predicted = trans_map.get(j + 1, "")
                gt = row["Sanskrit_IAST"]
                hit = check_match(predicted, gt) if predicted else False
                if hit:
                    matches += 1
                    batch_matches += 1
                all_results.append({
                    "ITA_ID": row["ITA_ID"],
                    "English_Term": row["English_Term"],
                    "Ground_Truth": gt,
                    "Predicted": predicted,
                    "Match": hit,
                })

            print(f"{batch_matches}/{len(batch)} matched  "
                  f"(running {matches}/{len(all_results)} = "
                  f"{100*matches/len(all_results):.1f}%)")

        except Exception as e:
            api_errors += 1
            print(f"ERROR: {e}")
            raw_f.write(json.dumps({
                "batch": b + 1,
                "start": start + 1,
                "end": end,
                "error": str(e),
                "ita_ids": [r["ITA_ID"] for r in batch],
            }, ensure_ascii=False) + "\n")
            raw_f.flush()
            for row in batch:
                all_results.append({
                    "ITA_ID": row["ITA_ID"],
                    "English_Term": row["English_Term"],
                    "Ground_Truth": row["Sanskrit_IAST"],
                    "Predicted": "",
                    "Match": False,
                })

        # Rate-limit: stay well within Groq's 30 RPM free-tier limit
        if b < num_batches - 1:
            time.sleep(2.5)

    raw_f.close()

    # ── Write detailed results ───────────────────────────────────────────
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ITA_ID", "English_Term", "Ground_Truth", "Predicted", "Match"])
        w.writeheader()
        w.writerows(all_results)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total terms:         {total}")
    print(f"Matches:             {matches}  ({100*matches/total:.1f}%)")
    print(f"Non-matches:         {total - matches}  ({100*(total-matches)/total:.1f}%)")
    print(f"API error batches:   {api_errors}")
    print(f"Input tokens:        {total_in_tok:,}")
    print(f"Output tokens:       {total_out_tok:,}")
    print(f"Total API calls:     {num_batches}")
    est_cost = total_in_tok * 0.29 / 1_000_000 + total_out_tok * 0.59 / 1_000_000
    print(f"Estimated cost:      ${est_cost:.4f}")
    print(f"Results saved to:    {OUTPUT_CSV}")
    print(f"Raw responses:       {RAW_RESPONSES_FILE}")


if __name__ == "__main__":
    main()