# AyurAssist Experiments

This branch contains experiments evaluating LLMs on Ayurvedic medical knowledge. Two benchmarks are included:

1. **BhashaBench-Ayur (BBA)** -- Multilingual Ayurvedic MCQ benchmark (English + Hindi)
2. **WHO-ITA Translation** -- English-to-Sanskrit IAST term translation using WHO International Terminologies for Ayurveda

---

## 1. BhashaBench-Ayur (BBA)

### Overview

BBA is a multiple-choice question benchmark for Ayurvedic medical knowledge covering topics like Kayachikitsa, Padartha Vigyana, Dravyaguna, Rachana Sharira, and more. Questions span Easy, Medium, and Hard difficulty levels.

- **English**: 9,348 questions
- **Hindi**: 5,615 questions
- **Total**: 14,963 questions
- **Source**: `bharatgenai/BhashaBench-Ayur` on HuggingFace

### Results

| Model | Params | BBA English | BBA Hindi | BBA Overall |
|-------|--------|-------------|-----------|-------------|
| **Qwen3-32B** | **32B** | **57.35** | **49.07** | **54.24** |
| GPT-OSS-120B | 120B | 55.16 | 47.14 | 52.15 |
| AyurParam-2.9B-Instruct | 2.9B | 41.12 | 38.04 | 39.97 |
| gemma-2-27B-it | 27B | 40.45 | 33.89 | 37.99 |
| Pangea-7B | 7B | 40.69 | 31.93 | 37.41 |
| gpt-oss-20B | 20B | 38.30 | 33.09 | 36.34 |
| Indic-gemma-7B-Navarasa-2.0 | 7B | 37.12 | 31.83 | 35.13 |

### Key Findings

- **Qwen3-32B is the overall best performer** at 54.24%, beating GPT-OSS-120B (52.15%) despite having 4x fewer parameters.
- All models perform worse on Hindi than English, with gaps ranging from 3-7 percentage points.

### Evaluation Method

- **Small & Medium models**: Loglikelihood scoring (standard multiple-choice evaluation via lm-eval-harness)
- **Large models**: Generate-until with chat completions API (Groq), temperature=0, regex-based answer extraction (A-D)

### Files

```
bba/
  analyze_accuracy.py          # Analysis script with breakdowns by level/topic/domain
  bba_english.csv              # English question dataset
  bba_english.json             # English dataset (JSON)
  bba_hindi.csv                # Hindi question dataset
  bba_hindi.json               # Hindi dataset (JSON)
  api_models.patch             # Patch for lm-eval api_models.py (Groq compatibility)
  lm_eval_tasks/               # lm-eval-harness task configurations
    _default_template_genai_yaml
    bba_genai.yaml
    bba_genai_English.yaml
    bba_genai_Hindi.yaml
    utils_bba.py
  results/
    bba_results_table.tex      # LaTeX results table
    qwen3-32b/                 # Per-sample results and summary JSON
    gpt-oss-120b/
    ayurparam-2.9b/
```

---

## 2. WHO-ITA Sanskrit Translation

### Overview

This experiment evaluates LLMs on translating English Ayurvedic terms to Sanskrit in IAST (International Alphabet of Sanskrit Transliteration). The ground truth comes from the WHO International Standard Terminologies on Ayurveda (WHO-ITA) 2022 document.

- **Total terms**: 3,550 Ayurvedic terms
- **Source**: WHO-ITA_2022.pdf (pages 19-491)
- **Task**: Given an English Ayurvedic term, produce the correct Sanskrit IAST equivalent

### Models Evaluated

| Model | Parameters | Provider |
|-------|-----------|----------|
| Qwen3-32B | 32B | Groq API |
| AyurParam-2.9B | 2.9B | Modal (T4 GPU) |
| Claude Sonnet 4.5 | -- | Anthropic API |

### Evaluation Method

- Models receive batches of English terms and must return Sanskrit IAST translations as structured JSON
- **Matching criteria**: Flexible matching with Unicode NFKD normalization (diacritics removed), lowercased. A prediction matches if any predicted sub-term matches any ground truth sub-term via exact or substring match. This accommodates multiple valid Sanskrit synonyms in the ground truth.

### Results

| Model | Params | Matches | Accuracy |
|-------|--------|---------|----------|
| **Claude Sonnet 4.5** | **--** | **1,237 / 3,550** | **34.84%** |
| Qwen3-32B | 32B | 250 / 3,550 | 7.04% |
| AyurParam-2.9B | 2.9B | 67 / 3,550 | 1.89% |

### Key Findings

- **Claude Sonnet 4.5 is the clear winner** at 34.84% accuracy -- nearly 5x better than Qwen3-32B and 18x better than AyurParam.
- **AyurParam underperforms** despite being Ayurveda-specialized, likely due to limited IAST transliteration in training data and difficulty with structured JSON output format. It often produced verbose explanations or Devanagari script instead of IAST.
- **Sanskrit IAST translation is a hard task** for all models. Even with lenient substring matching, the best model only achieves ~35% accuracy, highlighting the challenge of specialized medical terminology translation.
- Scale and training data diversity appear to matter more than domain specialization for this translation task.

### Files

```
who-ita/
  WHO-ITA_2022.pdf                      # Source PDF (WHO standard terminologies)
  extract_ita_terms.py                  # PDF extraction script (pdfplumber)
  ita_terms.csv                         # Extracted terms (3,550 rows)
  ita_terms_ascii.csv                   # ASCII-normalized version
  test_qwen_translation.py             # Qwen evaluation script
  test_ayurparam_translation.py         # AyurParam evaluation script
  test_claude_translation.py            # Claude evaluation script
  translation_results.csv               # Qwen results (per-term)
  translation_results_ayurparam.csv     # AyurParam results (per-term)
  translation_results_claude.csv        # Claude results (per-term)
  raw_model_responses.jsonl             # Qwen raw API responses
  raw_model_responses_ayurparam.jsonl   # AyurParam raw API responses
  raw_model_responses_claude.jsonl      # Claude raw API responses
  requirements.txt                      # Python dependencies
```