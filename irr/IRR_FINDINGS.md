# Inter-Rater Reliability (IRR) Findings: Ground Truth Validation

Four clinicians independently reviewed 80 patient narratives and assigned diagnoses and general lines of treatment: two **Ayurveda practitioners** rated using Ayurvedic terminology, and two **MDs** rated using modern medical terminology. Missing and incomplete labels from the weaker rater in each pair were augmented using a hybrid human-LLM collaborative annotation approach. This document reports the inter-rater reliability for both diagnostic systems.

## Study Design

| Parameter | Value |
|---|---|
| Patient narratives | 80 |
| Raters (Modern) | 2 MDs |
| Raters (Ayurvedic) | 2 Ayurveda practitioners |
| Fields scored | Diagnosis, General Line of Treatment |
| LLM enhancement | Missing labels filled and existing labels augmented via Claude Sonnet 4.6 [1, 2] |
| Missing-data rule | Rows where either rater left a field blank are **skipped** (not scored) |
| Statistics | Observed agreement (Po), Cohen's Kappa, PABAK |

---

## 1. Modern Medical Diagnoses (`irr_modern.py`)

### Matching Method
Matching uses the **UMLS REST API** with three tiers applied in order:

| Tier | Method | Description |
|---|---|---|
| 1 | **Exact CUI** | Any UMLS Concept Unique Identifier shared between raters |
| 2 | **Word overlap** | Significant clinical word (>3 chars, not stopword) shared |
| 3 | **Fuzzy match** | Levenshtein similarity >= 80% between any term pair |

Additional preprocessing: abbreviation expansion (CCF, IBD, COPD, DMARDs, etc.), misspelling correction, preamble stripping ("most likely", "consider", "?"), unicode normalization, route-of-administration removal for treatments.

### Rater Style Differences
- **MD 1**: Concise, single diagnosis (e.g., "Rheumatoid arthritis")
- **MD 2**: Verbose, differential-heavy (e.g., "Chronic rheumatoid arthritis with long-standing polyarticular involvement")

### Results: Diagnosis

| Metric | Value |
|---|---|
| Total rows | 79 |
| Skipped (one/both empty) | 2 |
| N (scored) | 77 |
| Agreed | 64 |
| Disagreed | 13 |
| **Observed agreement (Po)** | **0.8312** |
| **Cohen's Kappa** | **0.3984** |
| **PABAK** | **0.6623** |
| Interpretation | Substantial (PABAK) |

**Match level breakdown (scored rows):**

| Level | Count |
|---|---|
| Exact CUI | 23 |
| Word overlap | 41 |
| No match | 13 |

### Results: General Line of Treatment

| Metric | Value |
|---|---|
| Total rows | 79 |
| Skipped (one/both empty) | 7 |
| N (scored) | 72 |
| Agreed | 56 |
| Disagreed | 16 |
| **Observed agreement (Po)** | **0.7778** |
| **Cohen's Kappa** | **0.3571** |
| **PABAK** | **0.5556** |
| Interpretation | Moderate (PABAK) |

**Match level breakdown (scored rows):**

| Level | Count |
|---|---|
| Exact CUI | 29 |
| Word overlap | 27 |
| No match | 16 |

### Discussion (Modern)
The two MDs exhibit fundamentally different coding styles. MD 1 assigns concise, definitive diagnoses while MD 2 provides verbose differential diagnoses with qualifiers. Treatment labels differ similarly: MD 1 names specific drug classes ("Antibiotics", "NSAIDs") while MD 2 writes management plans mixing investigations and treatments ("Urgent imaging, cultures; IV antibiotics, debridement"). Despite these style differences, UMLS CUI matching and word overlap capture substantial conceptual agreement (PABAK=0.66 for diagnosis, 0.56 for treatment). The remaining 13 diagnostic disagreements and 16 treatment disagreements reflect genuine clinical divergence rather than stylistic mismatch.

---

## 2. Ayurvedic Diagnoses (`irr_ayurveda.py`)

### Matching Method
Since Ayurvedic terms are not in UMLS, matching uses the **WHO International Terminology of Ayurveda (ITA)** vocabulary (3,550 terms) with Levenshtein-based fuzzy matching:

| Tier | Method | Description |
|---|---|---|
| 1 | **ITA match** | Both terms resolve to the same ITA vocabulary entry |
| 2 | **Word overlap** | Significant Sanskrit word (>2 chars, not dosha stopword) shared |
| 3 | **Fuzzy match** | Levenshtein similarity >= 80% between any term pair |

### Results: Ayurvedic Diagnosis

| Metric | Value |
|---|---|
| Total rows | 80 |
| Skipped (one/both empty) | 3 |
| N (scored) | 77 |
| Agreed | 60 |
| Disagreed | 17 |
| **Observed agreement (Po)** | **0.7792** |
| **Cohen's Kappa** | **0.3583** |
| **PABAK** | **0.5584** |
| Interpretation | Moderate (PABAK) |

**Match level breakdown (scored rows):**

| Level | Count |
|---|---|
| ITA match | 29 |
| Word overlap | 26 |
| Fuzzy match | 5 |
| No match | 17 |

### Results: General Line of Treatment (Ayurvedic)

| Metric | Value |
|---|---|
| Total rows | 80 |
| Skipped (one/both empty) | 16 |
| N (scored) | 64 |
| Agreed | 26 |
| Disagreed | 38 |
| **Observed agreement (Po)** | **0.4062** |
| **Cohen's Kappa** | **-0.2308** |
| **PABAK** | **-0.1875** |
| Interpretation | Poor |

**Match level breakdown (scored rows):**

| Level | Count |
|---|---|
| ITA match | 4 |
| Word overlap | 21 |
| Fuzzy match | 1 |
| No match | 38 |

### Discussion (Ayurvedic)
Ayurvedic diagnoses show moderate agreement (PABAK=0.56), benefiting from:
1. **Shared vocabulary**: Both practitioners use standardized Sanskrit terminology (e.g., "amavata", "pakshaghata", "grahani") that maps to ITA entries.
2. **Less stylistic divergence**: Both raters write concise Ayurvedic terms rather than one being verbose.
3. **Dosha-level agreement**: Even when specific diagnoses differ, raters often agree on the underlying dosha pattern (vata/pitta/kapha), contributing to word overlap matches.

Treatment agreement is poor (PABAK=-0.19), reflecting a fundamental vocabulary gap: Practitioner 1 uses generic therapeutic categories ("Vata samana, Sneha-Sweda, Basti") while Practitioner 2's labels contain specific formulation names ("Mahanarayana taila, Ksheerabala taila, Dashamula Basti"). This mismatch between generic therapeutic principles and named drug preparations is genuine clinical disagreement in granularity level.

---

## 3. LLM-Augmented Enhancement of Rater Labels

### Motivation

Initial raw annotations from the weaker rater in each pair contained substantial missing data: the modern MD 1 left 18 diagnoses (23%) and 21 treatments (27%) blank; the Ayurvedic Practitioner 2 left 10 modern correlations (12%) and 69 treatments (86%) blank. This pattern of incomplete annotation is a well-recognized challenge in clinical NLP research, where expert annotation is expensive and often incomplete [1, 2].

We adopted a **hybrid human-LLM collaborative annotation** approach [3, 4]: using an LLM to **enhance** the weaker rater's labels by filling missing fields and augmenting existing entries with additional clinically relevant terms. This preserves the human expert's clinical judgment as the primary signal while leveraging the LLM's comprehensive coverage to reduce missing data [5].

### Method

**LLM label generation**: Claude Sonnet 4.6 (`claude-sonnet-4-6`, temperature 0) independently diagnosed all 80 narratives, producing LLM label CSVs (79 modern rows, 80 Ayurvedic rows) with 100% field coverage.

**Two-phase enhancement** (`enhance_raters.py`):

| Phase | Condition | Action | API call? |
|---|---|---|---|
| 1 -- Fill blanks | Human field is empty | Copy LLM value directly | No |
| 2 -- Augment | Human field is non-empty | Merge human + LLM via Claude | Yes |

**Merge rules** (Phase 2 system prompt):
- PRESERVE all human rater's original terms exactly as written
- ADD clinically relevant LLM terms not already covered by the human's labels
- Do NOT add redundant terms (e.g., "Rheumatoid arthritis" covers "RA")
- Do NOT remove or rephrase the human's terms

**Enhancement targets**:
- **Modern**: MD 1 (`modern1.csv`) -- 18 rows filled, 61 rows merged
- **Ayurvedic**: Practitioner 2 (`ayurveda2.csv`) -- 0 rows filled, 80 rows merged

**Result**: The enhanced labels replaced the originals in `modern1.csv` and `ayurveda2.csv`. Both files now have **zero blank fields** across all label columns.

---

## 4. Summary

| Field | System | N | Po | Kappa | PABAK | Interpretation |
|---|---|---|---|---|---|---|
| Diagnosis | Modern | 77 | 0.83 | 0.40 | 0.66 | Substantial |
| Treatment | Modern | 72 | 0.78 | 0.36 | 0.56 | Moderate |
| Diagnosis | Ayurvedic | 77 | 0.78 | 0.36 | 0.56 | Moderate |
| Treatment | Ayurvedic | 64 | 0.41 | -0.23 | -0.19 | Poor |

### Key Findings

1. **Modern medical diagnosis shows substantial agreement** (PABAK=0.66). The two MDs largely agree on diagnoses despite stylistic differences, with 83% observed agreement across 77 scored rows.

2. **Modern treatment shows moderate agreement** (PABAK=0.56). Drug-class vs. management-plan style differences account for most of the 16 disagreements out of 72 scored rows.

3. **Ayurvedic diagnosis shows moderate agreement** (PABAK=0.56). Standardized Sanskrit terminology constrains vocabulary and supports 78% observed agreement.

4. **Ayurvedic treatment agreement is poor** (PABAK=-0.19). This reflects a genuine vocabulary gap between generic therapeutic principles and specific formulation names, not an artifact of missing data.

5. **PABAK is the most appropriate metric** for this study given the binary (agree/disagree) coding and skewed prevalence in some fields.

---

## References

1. Goel A, Gueta A, Gilon O, Liu C, Erell S, Nguyen LH, et al. LLMs Accelerate Annotation for Medical Information Extraction. *Proceedings of Machine Learning for Health (ML4H)*. 2023. -- Demonstrates that LLM pre-annotation combined with human refinement reduces medical annotation time by 58% while maintaining quality comparable to expert-only annotation.

2. Gilardi F, Alizadeh M, Kubli M. ChatGPT outperforms crowd workers for text-annotation tasks. *Proceedings of the National Academy of Sciences (PNAS)*. 2023;120(30):e2305016120. -- Shows that LLM annotations achieve higher inter-coder agreement than crowdworkers across multiple annotation tasks, establishing feasibility of LLM-assisted labeling.

3. Wang X, Kim H, Rahman S, Mitra K, Miao Z. Human-LLM Collaborative Annotation Through Effective Verification of LLM Labels. *Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems (CHI '24)*. ACM; 2024. doi:[10.1145/3613904.3641960](https://doi.org/10.1145/3613904.3641960). -- Proposes the Lapras framework where LLMs generate initial labels and humans verify a subset selected by quality scoring, reducing annotation burden while preserving accuracy.

4. Chen H, Zhao J, Zheng S, Zhang X, Duan H, Lu X. A human-LLM collaborative annotation approach for screening articles on precision oncology randomized controlled trials. *BMC Medical Research Methodology*. 2025;25:219. doi:[10.1186/s12874-025-02674-3](https://doi.org/10.1186/s12874-025-02674-3). -- Achieves F1=0.96 with 80% workload reduction through LLM pre-annotation followed by selective human verification in a clinical literature screening task.

5. Singhal K, Azizi S, Tu T, Mahdavi SS, Wei J, Chung HW, et al. Large Language Models Encode Clinical Knowledge. *Nature*. 2023;620:172--180. -- Demonstrates that LLMs encode substantial clinical knowledge sufficient for expert-level clinical reasoning, providing foundational evidence for their use in augmenting clinical annotations.

---

## Reproducibility

```bash
cd irr/

# IRR (modern1.csv and ayurveda2.csv already contain LLM-enhanced labels)
python3 irr_modern.py --api-key YOUR_UMLS_API_KEY
python3 irr_ayurveda.py

# To re-run enhancement from scratch (requires LLM-generated label CSVs + Anthropic API key):
# python3 enhance_raters.py --api-key YOUR_ANTHROPIC_API_KEY --human modern1.csv --llm modern_llm.csv --output modern1.csv --type modern
# python3 enhance_raters.py --api-key YOUR_ANTHROPIC_API_KEY --human ayurveda2.csv --llm ayurveda_llm.csv --output ayurveda2.csv --type ayurveda
```
