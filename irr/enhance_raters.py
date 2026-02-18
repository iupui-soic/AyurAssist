#!/usr/bin/env python3
"""
Enhance weaker human rater CSVs by filling blanks from LLM-generated labels
and augmenting existing entries via Claude merge.

Phase 1: If a human field is empty, use the corresponding LLM value directly.
Phase 2: For non-empty human fields, ask Claude to merge human + LLM labels,
         preserving all human terms and adding clinically distinct LLM terms.

Usage:
    python enhance_raters.py --api-key KEY --human modern1.csv --llm modern_llm.csv --output modern1.csv --type modern
    python enhance_raters.py --api-key KEY --human ayurveda2.csv --llm ayurveda_llm.csv --output ayurveda2.csv --type ayurveda
"""

import argparse
import csv
import io
import json
import os
import sys
import time

BATCH_SIZE = 10
MAX_RETRIES = 3
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
CHECKPOINT_FILE = "enhance_checkpoint.json"

INSTRUCTION_MARKERS = [
    "give single ayurvedic diagnosis",
    "acc to the patient narratives",
]

SYSTEM_PROMPT = """You are a clinical terminology editor. You will receive patient narratives with labels from a human rater and an AI rater. Your task is to merge them.

Rules:
- PRESERVE all of the human rater's original terms exactly as written
- ADD any clinically relevant terms from the AI rater that are NOT already covered by the human's terms
- Do NOT add redundant terms (e.g., if human says "Rheumatoid arthritis" and AI says "RA", keep only the human's)
- Do NOT remove or rephrase the human's terms
- Use comma-separation for multiple terms, matching the human rater's formatting style
- If the AI adds nothing new, return the human's text unchanged"""


def is_instruction_row(narrative, diagnosis):
    """Check if this row is an instruction row (not patient data)."""
    combined = (narrative + " " + diagnosis).lower()
    return any(marker in combined for marker in INSTRUCTION_MARKERS)


def read_human_csv(filepath, csv_type):
    """Read human rater CSV, filtering instruction/empty rows."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return []

    data_rows = rows[1:]  # skip header
    results = []

    if csv_type == "modern":
        for row in data_rows:
            while len(row) < 3:
                row.append("")
            narrative = row[0].strip()
            diagnosis = row[1].strip()
            treatment = row[2].strip()
            if is_instruction_row(narrative, diagnosis):
                continue
            if not narrative and not diagnosis and not treatment:
                continue
            results.append({
                "narrative": narrative,
                "diagnosis": diagnosis,
                "treatment": treatment,
            })
    else:  # ayurveda
        for row in data_rows:
            while len(row) < 4:
                row.append("")
            narrative = row[0].strip()
            diagnosis = row[1].strip()
            modern_correlation = row[2].strip()
            treatment = row[3].strip()
            if not narrative and not diagnosis and not modern_correlation and not treatment:
                continue
            results.append({
                "narrative": narrative,
                "diagnosis": diagnosis,
                "modern_correlation": modern_correlation,
                "treatment": treatment,
            })

    return results


def read_llm_csv(filepath, csv_type):
    """Read LLM rater CSV."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return []

    data_rows = rows[1:]  # skip header
    results = []

    if csv_type == "modern":
        for row in data_rows:
            while len(row) < 3:
                row.append("")
            results.append({
                "narrative": row[0].strip(),
                "diagnosis": row[1].strip(),
                "treatment": row[2].strip(),
            })
    else:  # ayurveda
        for row in data_rows:
            while len(row) < 4:
                row.append("")
            results.append({
                "narrative": row[0].strip(),
                "diagnosis": row[1].strip(),
                "modern_correlation": row[2].strip(),
                "treatment": row[3].strip(),
            })

    return results


def call_api_anthropic(api_key, system, user_msg):
    """Call Claude API using the anthropic SDK."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text


def call_api_requests(api_key, system, user_msg):
    """Call Claude API using requests as fallback."""
    import requests
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user_msg}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


def call_api(api_key, system, user_msg):
    """Try anthropic SDK first, fall back to requests."""
    try:
        import anthropic  # noqa: F401
        return call_api_anthropic(api_key, system, user_msg)
    except ImportError:
        return call_api_requests(api_key, system, user_msg)


def parse_json_response(text, expected_count):
    """Extract and parse JSON array from API response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    results = json.loads(cleaned)
    if not isinstance(results, list) or len(results) != expected_count:
        raise ValueError(
            f"Expected JSON array of {expected_count} items, got "
            f"{len(results) if isinstance(results, list) else type(results).__name__}"
        )
    return results


def get_label_fields(csv_type):
    """Return the label field names for a given CSV type."""
    if csv_type == "modern":
        return ["diagnosis", "treatment"]
    else:
        return ["diagnosis", "modern_correlation", "treatment"]


FIELD_DISPLAY = {
    "diagnosis": "Diagnosis",
    "treatment": "Treatment",
    "modern_correlation": "Modern Correlation",
}


def build_merge_prompt(batch_items, csv_type):
    """Build the user prompt for a merge batch."""
    fields = get_label_fields(csv_type)
    lines = [
        "Merge labels for these cases. For each, the human rater's "
        "label comes first, then the AI rater's.\n"
    ]

    for i, item in enumerate(batch_items, 1):
        lines.append(f"[{i}] Narrative: {item['narrative']}")
        for field in fields:
            display = FIELD_DISPLAY[field]
            lines.append(f"    Human {display}: {item['human'][field]}")
            lines.append(f"    AI {display}: {item['llm'][field]}")
        lines.append("")

    keys_str = ", ".join(fields)
    lines.append(
        f"Return a JSON array of {len(batch_items)} objects with keys: {keys_str}"
    )
    return "\n".join(lines)


def load_checkpoint():
    """Load checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed_batches": [], "merge_results": {}}


def save_checkpoint(checkpoint):
    """Save checkpoint to disk."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)


def process_merge_batch(api_key, batch_items, batch_idx, total_batches, csv_type):
    """Process a single merge batch with retries."""
    user_msg = build_merge_prompt(batch_items, csv_type)

    for attempt in range(MAX_RETRIES):
        try:
            raw = call_api(api_key, SYSTEM_PROMPT, user_msg)
            results = parse_json_response(raw, len(batch_items))
            return results
        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                delay = 2 ** (attempt + 1)
                print(f"  JSON parse error on batch {batch_idx + 1}, retrying in {delay}s...")
                time.sleep(delay)
                user_msg = (
                    build_merge_prompt(batch_items, csv_type)
                    + "\n\nIMPORTANT: Return ONLY valid JSON, no extra text."
                )
            else:
                raise
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = 2 ** (attempt + 1)
                print(f"  Error on batch {batch_idx + 1}: {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def enhance(api_key, human_rows, llm_rows, csv_type):
    """Enhance human rater labels: fill blanks (Phase 1) then merge (Phase 2)."""
    fields = get_label_fields(csv_type)

    # Phase 1: Fill blanks, identify rows needing merge
    enhanced = []
    merge_queue = []  # items needing API merge: {row_index, narrative, human, llm}

    for i, (human, llm) in enumerate(zip(human_rows, llm_rows)):
        row = dict(human)
        needs_merge = False
        for field in fields:
            if not human[field]:
                row[field] = llm[field]
            else:
                needs_merge = True

        enhanced.append(row)
        if needs_merge:
            merge_queue.append({
                "row_index": i,
                "narrative": human["narrative"],
                "human": {f: human[f] for f in fields},
                "llm": {f: llm[f] for f in fields},
            })

    blanks_filled = len(human_rows) - len(merge_queue)
    print(f"\nPhase 1: {blanks_filled} rows filled entirely from LLM (no API call needed)")

    if not merge_queue:
        print("No rows need merging.")
        return enhanced

    print(f"Phase 2: Merging {len(merge_queue)} rows via Claude API in batches of {BATCH_SIZE}...")

    # Batch the merge queue
    batches = [
        merge_queue[i:i + BATCH_SIZE]
        for i in range(0, len(merge_queue), BATCH_SIZE)
    ]
    total_batches = len(batches)

    # Load checkpoint
    checkpoint = load_checkpoint()
    completed = set(checkpoint["completed_batches"])
    merge_results = checkpoint.get("merge_results", {})

    if completed:
        print(f"Resuming from checkpoint: {len(completed)}/{total_batches} batches done")
        # Apply previously completed merge results
        for idx_str, result in merge_results.items():
            row_idx = int(idx_str)
            for field in fields:
                if field in result:
                    enhanced[row_idx][field] = result[field]

    for batch_idx, batch in enumerate(batches):
        if batch_idx in completed:
            continue

        results = process_merge_batch(
            api_key, batch, batch_idx, total_batches, csv_type
        )

        # Apply merge results
        for item, result in zip(batch, results):
            row_idx = item["row_index"]
            for field in fields:
                if field in result:
                    enhanced[row_idx][field] = result[field]
            merge_results[str(row_idx)] = result

        completed.add(batch_idx)
        checkpoint["completed_batches"] = sorted(completed)
        checkpoint["merge_results"] = merge_results
        save_checkpoint(checkpoint)

        print(f"  Batch {batch_idx + 1}/{total_batches} complete")

    return enhanced


def write_enhanced_csv(enhanced_rows, output_path, csv_type):
    """Write the enhanced CSV matching original header format."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if csv_type == "modern":
            writer.writerow([
                "Patient narratives", "Diagnosis", "General Line of Treatment"
            ])
            for row in enhanced_rows:
                writer.writerow([
                    row["narrative"], row["diagnosis"], row["treatment"]
                ])
        else:
            writer.writerow([
                "Patient Narratives", "Diagnosis",
                "Modern Correlation", "General line of treatment"
            ])
            for row in enhanced_rows:
                writer.writerow([
                    row["narrative"], row["diagnosis"],
                    row["modern_correlation"], row["treatment"]
                ])


def main():
    parser = argparse.ArgumentParser(
        description="Enhance human rater CSVs with LLM labels"
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument("--human", required=True, help="Path to human rater CSV")
    parser.add_argument("--llm", required=True, help="Path to LLM-generated labels CSV")
    parser.add_argument("--output", required=True, help="Path for enhanced output CSV")
    parser.add_argument(
        "--type", required=True, choices=["modern", "ayurveda"],
        help="CSV type: modern or ayurveda",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "Error: provide --api-key or set ANTHROPIC_API_KEY env var",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read CSVs
    human_rows = read_human_csv(args.human, args.type)
    llm_rows = read_llm_csv(args.llm, args.type)

    print(f"Human rater: {len(human_rows)} rows from {args.human}")
    print(f"LLM rater:   {len(llm_rows)} rows from {args.llm}")

    if len(human_rows) != len(llm_rows):
        print(
            f"ERROR: Row count mismatch: human={len(human_rows)}, llm={len(llm_rows)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Report blanks per field
    fields = get_label_fields(args.type)
    for field in fields:
        blanks = sum(1 for r in human_rows if not r[field])
        pct = blanks * 100 // len(human_rows)
        print(f"  {field}: {blanks} blank ({pct}%)")

    # Enhance
    enhanced = enhance(args.api_key, human_rows, llm_rows, args.type)

    # Write output
    write_enhanced_csv(enhanced, args.output, args.type)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    # Verify
    print(f"\nOutput: {args.output} ({len(enhanced)} rows)")
    print("Remaining blanks:")
    for field in fields:
        remaining = sum(1 for r in enhanced if not r[field])
        print(f"  {field}: {remaining}")

    expected = 79 if args.type == "modern" else 80
    if len(enhanced) != expected:
        print(
            f"WARNING: expected {expected} rows, got {len(enhanced)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
