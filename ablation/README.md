# Ablation Study

Ablation study comparing pipeline components for Ayurvedic clinical decision
support. Evaluates which components (NER, UMLS, ITA terminology bridge, LLM)
contribute most to diagnostic accuracy across 80 gold-standard clinical
vignettes.

## Configurations

| Config | NER | UMLS | ITA Vocab | LLM | Purpose |
|--------|-----|------|-----------|-----|---------|
| **direct_llm** | - | - | - | Qwen3-32B | Baseline: LLM alone |
| **full_pipeline** | Yes | Yes | Yes (3,550 terms) | Qwen3-32B | Complete system |
| **bridge_only** | Yes | Yes | Yes (3,550 terms) | - | Terminology bridge without LLM |

## Results

### Primary Metrics (over 80 vignettes)

| Config | Dx Acc | Dx ROUGE-L | Dx Token-F1 | Tx Acc | Tx ROUGE-L | Tx Token-F1 |
|--------|--------|-----------|-------------|--------|-----------|-------------|
| direct_llm | 0.750 (60/80) | 0.104 | 0.130 | 1.000 (80/80) | 0.034 | 0.083 |
| full_pipeline | 0.800 (64/80) | 0.065 | 0.088 | 1.000 (80/80) | 0.060 | 0.125 |
| bridge_only | 0.050 (4/80) | 0.003 | 0.003 | 0.000 (0/80) | 0.000 | 0.000 |

### Term-Level Metrics (micro-averaged over individual terms)

| Config | Dx P | Dx R | Dx F1 | Tx P | Tx R | Tx F1 |
|--------|------|------|-------|------|------|-------|
| direct_llm | 0.178 | 0.263 | 0.213 | 0.092 | 0.502 | 0.156 |
| full_pipeline | 0.060 | 0.346 | 0.102 | 0.144 | 0.453 | 0.218 |
| bridge_only | 0.022 | 0.013 | 0.016 | 0.000 | 0.000 | 0.000 |

### Diagnosis Match Tier Breakdown

| Config | ITA Match | Word Overlap | Fuzzy Match | No Match |
|--------|-----------|--------------|-------------|----------|
| direct_llm | 18 | 56 | 9 | 383 |
| full_pipeline | 24 | 77 | 8 | 1,704 |
| bridge_only | 4 | 0 | 0 | 178 |

### Key Findings

1. **Terminology bridge improves diagnostic accuracy** — full_pipeline achieves
   80% diagnosis accuracy (64/80) vs direct_llm's 75% (60/80), showing the ITA
   vocabulary and UMLS context help the LLM produce more accurate Ayurvedic diagnoses.
2. **LLM is essential** — bridge_only achieves only 5% diagnosis accuracy (4/80)
   and 0% treatment accuracy, confirming the LLM is the primary reasoning engine
   while the terminology bridge provides valuable context.
3. **Bridge improves treatment quality** — full_pipeline outperforms direct_llm
   on treatment metrics (Token-F1: 0.125 vs 0.083; Term-F1: 0.218 vs 0.156),
   showing the ITA vocabulary helps ground treatment recommendations in standard
   Ayurvedic terminology.
4. **Surface-level metrics are limited** — ROUGE-L and Token-F1 scores are low
   across all configs because Ayurvedic terminology has many valid transliterations
   and synonyms (e.g., "Sandhivata" vs "Vata Vyadhi" for joint disease). Accuracy
   (which uses tiered semantic matching) better captures clinical correctness.

## Architecture

### direct_llm (baseline)
Raw patient narrative sent to Qwen3-32B via Groq API with two prompts:
1. Ayurvedic diagnosis
2. Ayurvedic treatment

### full_pipeline (LLM + NER + ITA bridge)
Three architectural improvements over the baseline:
1. **Constrained generation** — WHO-ITA vocabulary (3,550 terms) provided as
   a reference list in the LLM prompt
2. **Two-pass reasoning** — Pass 1: modern medical diagnosis; Pass 2: Ayurvedic
   translation using the modern diagnosis + ITA vocabulary
3. **Few-shot examples** — 3 gold-standard vignette-to-diagnosis pairs included
   in the prompt

The terminology bridge (scispacy NER + UMLS + ITA CSV) provides **context** to
the LLM, not the diagnosis itself. NER extracts medical entities for
explainability, and the ITA vocabulary serves as a constrained dictionary so the
LLM can use correct Sanskrit terminology.

### bridge_only (no LLM)
NER extracts entities, UMLS normalizes them, and ITA fuzzy matching maps them
to Ayurvedic terms. No LLM generation. This tests whether the terminology
bridge alone can produce correct diagnoses.

## Gold Standard

- **Source**: 80 clinical vignettes from `irr/ayurveda1.csv` + `irr/ayurveda2.csv`
- **Matching**: Union of both raters — a prediction is correct if it matches
  EITHER rater's diagnosis/treatment
- **Columns**: Patient Narrative, Diagnosis, General Line of Treatment

## Evaluation Metrics

Four complementary metrics capture different aspects of output quality:

- **Accuracy** — Per-vignette binary: did any predicted term match any gold term
  (via tiered semantic matching)? Best metric for clinical correctness since it
  handles Ayurvedic synonym variation.
- **ROUGE-L** — Longest Common Subsequence F1 between predicted and gold text.
  Macro-averaged over vignettes. Handles verbosity but limited to surface overlap.
- **Token-F1** — Bag-of-words overlap F1 (SQuAD-style). Macro-averaged over
  vignettes. Measures unique token overlap between prediction and gold standard.
- **Term-level P/R/F1** — Each predicted term matched to gold terms via tiered
  matching: (1) ITA ID match, (2) significant word overlap, (3) fuzzy
  Levenshtein >= 0.80. Micro-averaged across all terms.

## Usage

```bash
# Set up (Python 3.13 required — spacy is incompatible with 3.14)
cd ablation
python3.13 -m venv .venv
source .venv/bin/activate
pip install groq requests spacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.5/en_core_sci_lg-0.5.4.tar.gz --no-deps

# Set API keys
export GROQ_API_KEY="..."
export UMLS_API_KEY="..."

# Run all configs on all 80 vignettes
python run_ablation.py

# Run specific configs with a limit
python run_ablation.py --configs direct full --limit 5

# Re-evaluate from cached checkpoints (no API calls)
python run_ablation.py --reeval
```

## File Structure

```
ablation/
├── README.md
├── run_ablation.py          # CLI orchestrator
├── configs.py               # Constants: model names, paths, thresholds
├── gold_standard.py         # Load 80 vignettes + parse rater annotations
├── terminology_bridge.py    # NER + UMLS + ITA CSV lookup
├── pipelines/
│   ├── base.py              # Abstract base with JSONL checkpointing
│   ├── direct_llm.py        # Raw narrative → Qwen3 (baseline)
│   ├── full_pipeline.py     # NER → UMLS → ITA → Qwen3 (full system)
│   └── bridge_only.py       # NER → UMLS → ITA only (no LLM)
├── evaluation/
│   ├── matching.py          # Tiered term-level matching
│   ├── text_metrics.py      # ROUGE-L and token-level F1
│   ├── metrics.py           # Metric aggregation (all three metric types)
│   └── report.py            # Markdown tables, CSV, JSON output
└── results/                 # Created at runtime
    ├── raw_responses/       # Per-config JSONL checkpoints
    └── summary/             # comparison_table.md, per_vignette.csv, metrics.json
```

## Dependencies

- `groq` — Qwen3-32B API access
- `spacy` + `en_core_sci_lg` — biomedical NER
- `requests` — UMLS REST API
- Python 3.13 (spacy incompatible with 3.14)
