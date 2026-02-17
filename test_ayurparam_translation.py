#!/usr/bin/env python
"""
Batch-test AyurParam (bharatgenai/AyurParam, 2.9B) on translating English
Ayurvedic terms to Sanskrit IAST.  Runs on Modal GPU infrastructure.

Usage:
    modal run test_ayurparam_translation.py [--limit 100] [--batch-size 15]
"""

import csv
import json
import os
import re
import sys
import unicodedata

import modal

# ── Reuse existing Modal infra from config ───────────────────────────────
# We import only what we need to avoid pulling in the full app
MODAL_APP_NAME = "ayurparam-translation-test"
MODAL_VOLUME_NAME = "ayurparam-models-final"
MODAL_SECRET_HUGGINGFACE = "huggingface-secret"
LLM_MODEL_ID = "bharatgenai/AyurParam"
MODEL_CACHE_DIR = "/cache/models"
VOLUME_MOUNT_PATH = "/cache"
PYTHON_VERSION = "3.11"

GROUND_TRUTH_CSV = "ita_terms.csv"
OUTPUT_CSV = "translation_results_ayurparam.csv"
RAW_RESPONSES_FILE = "raw_model_responses_ayurparam.jsonl"
DEFAULT_BATCH_SIZE = 15   # small batches — AyurParam has 2048 context window

app = modal.App(MODAL_APP_NAME)

gpu_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "torch==2.1.0",
        "numpy==1.24.3",
        "transformers==4.46.0",
        "accelerate==0.34.0",
        "huggingface_hub==0.25.0",
    )
)

volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)


# ── GPU class: loads AyurParam once, generates many times ────────────────

@app.cls(
    image=gpu_image,
    gpu="T4",
    timeout=600,
    scaledown_window=120,
    volumes={VOLUME_MOUNT_PATH: volume},
    secrets=[modal.Secret.from_name(MODAL_SECRET_HUGGINGFACE)],
)
class TranslationEngine:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import login

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_ID,
            use_fast=False,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
            device_map="auto",
        )
        print("AyurParam translation engine ready.")

    @modal.method()
    def translate_batch(self, batch_json: str) -> str:
        """
        Accepts a JSON string of [{n, english}, ...], returns the raw model
        output for the batch translation prompt.
        """
        import torch

        items = json.loads(batch_json)
        lines = [f"{it['n']}. {it['english']}" for it in items]
        user_msg = (
            "Translate each English Ayurvedic term below into its Sanskrit "
            "equivalent in IAST transliteration. Return one per line as "
            "\"N. sanskrit_iast\". No explanations.\n\n" + "\n".join(lines)
        )
        prompt = f"<user> {user_msg} <assistant>"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        max_new = min(400, 2048 - prompt_len)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=0.4,
                top_p=0.95,
                top_k=50,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        new_tokens = output_ids[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Normalisation helpers ────────────────────────────────────────────────

def strip_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize(text: str) -> str:
    t = strip_diacritics(text).lower().strip()
    return re.sub(r"[^a-z0-9 ]", "", t)


def parse_terms(field: str) -> list[str]:
    field = re.sub(r"\d+\.", "", field)
    parts = re.split(r"[;,]", field)
    return [normalize(t) for t in parts if t.strip()]


def check_match(predicted: str, ground_truth: str) -> bool:
    pred_terms = parse_terms(predicted)
    gt_terms = parse_terms(ground_truth)
    for p in pred_terms:
        for g in gt_terms:
            if p and g and (p == g or p in g or g in p):
                return True
    return False


def parse_numbered_lines(text: str, expected_count: int) -> dict[int, str]:
    """Parse 'N. term' lines from model output."""
    result: dict[int, str] = {}
    for line in text.strip().splitlines():
        m = re.match(r"(\d+)\.\s*(.+)", line.strip())
        if m:
            n = int(m.group(1))
            term = m.group(2).strip().rstrip(".")
            result[n] = term
    return result


# ── Local entrypoint ─────────────────────────────────────────────────────

@app.local_entrypoint()
def main(limit: int = 0, batch_size: int = DEFAULT_BATCH_SIZE):
    # Load ground truth locally
    rows: list[dict] = []
    with open(GROUND_TRUTH_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if limit:
        rows = rows[:limit]

    total = len(rows)
    num_batches = (total + batch_size - 1) // batch_size

    print(f"Terms: {total}  |  Batch size: {batch_size}  |  Batches: {num_batches}")
    print(f"Model: {LLM_MODEL_ID}  (Modal GPU T4)\n")

    # Build all batch payloads
    batch_payloads = []
    batch_rows = []
    for b in range(num_batches):
        start = b * batch_size
        end = min(start + batch_size, total)
        batch = rows[start:end]
        payload = json.dumps([
            {"n": j + 1, "english": r["English_Term"]}
            for j, r in enumerate(batch)
        ], ensure_ascii=False)
        batch_payloads.append(payload)
        batch_rows.append(batch)

    # Send all batches via .map() — Modal handles parallelism
    engine = TranslationEngine()
    raw_responses = list(engine.translate_batch.map(batch_payloads))

    # Process results
    all_results: list[dict] = []
    matches = 0

    raw_f = open(RAW_RESPONSES_FILE, "w", encoding="utf-8")

    for b, (batch, raw_content) in enumerate(zip(batch_rows, raw_responses)):
        start = b * batch_size

        raw_f.write(json.dumps({
            "batch": b + 1,
            "start": start + 1,
            "end": start + len(batch),
            "raw_response": raw_content,
            "ita_ids": [r["ITA_ID"] for r in batch],
        }, ensure_ascii=False) + "\n")

        trans_map = parse_numbered_lines(raw_content, len(batch))

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

        print(f"Batch {b+1}/{num_batches}  (terms {start+1}–{start+len(batch)}): "
              f"{batch_matches}/{len(batch)} matched  "
              f"(running {matches}/{len(all_results)} = "
              f"{100*matches/len(all_results):.1f}%)")

    raw_f.close()

    # Write results
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ITA_ID", "English_Term", "Ground_Truth", "Predicted", "Match"])
        w.writeheader()
        w.writerows(all_results)

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total terms:         {total}")
    print(f"Matches:             {matches}  ({100*matches/total:.1f}%)")
    print(f"Non-matches:         {total - matches}  ({100*(total-matches)/total:.1f}%)")
    print(f"Total API calls:     {num_batches}")
    print(f"Results saved to:    {OUTPUT_CSV}")
    print(f"Raw responses:       {RAW_RESPONSES_FILE}")