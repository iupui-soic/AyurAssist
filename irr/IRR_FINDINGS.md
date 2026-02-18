# Inter-Rater Reliability (IRR) Findings: Ground Truth Validation

Two clinicians independently reviewed 80 patient narratives and assigned diagnoses and general lines of treatment. This document reports the inter-rater reliability for both **modern medical** and **Ayurvedic** diagnostic coding.

## Study Design

| Parameter | Value |
|---|---|
| Patient narratives | 80 |
| Raters | 2 clinicians per system |
| Fields scored | Diagnosis, General Line of Treatment |
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
- **Rater 1**: Concise, single diagnosis (e.g., "Rheumatoid arthritis")
- **Rater 2**: Verbose, differential-heavy (e.g., "Chronic rheumatoid arthritis with long-standing polyarticular involvement")

### Results: Diagnosis

| Metric | Value |
|---|---|
| Total rows | 79 |
| Skipped (one/both empty) | 20 |
| N (scored) | 59 |
| Agreed | 25 |
| Disagreed | 34 |
| **Observed agreement (Po)** | **0.4237** |
| Expected agreement (Pe) | 0.5116 |
| **Cohen's Kappa** | **-0.18** |
| **PABAK** | **-0.15** |
| Interpretation | Poor (near chance) |

**Match level breakdown (scored rows):**

| Level | Count |
|---|---|
| Exact CUI | 11 |
| Word overlap | 14 |
| No match | 34 |

### Results: General Line of Treatment

| Metric | Value |
|---|---|
| Total rows | 79 |
| Skipped (one/both empty) | 26 |
| N (scored) | 53 |
| Agreed | 11 |
| Disagreed | 42 |
| **Observed agreement (Po)** | **0.2075** |
| Expected agreement (Pe) | 0.6711 |
| **Cohen's Kappa** | **-1.41** |
| **PABAK** | **-0.58** |
| Interpretation | Poor |

**Match level breakdown (scored rows):**

| Level | Count |
|---|---|
| Exact CUI | 6 |
| Word overlap | 5 |
| No match | 42 |

### Discussion (Modern)
The low modern IRR reflects fundamentally different coding styles rather than clinical disagreement. Rater 1 assigns concise, definitive diagnoses while Rater 2 provides verbose differential diagnoses with qualifiers. Treatment agreement is even lower because Rater 1 names specific drug classes ("Antibiotics", "NSAIDs") while Rater 2 writes management plans mixing investigations and treatments ("Urgent imaging, cultures; IV antibiotics, debridement"). Many genuine conceptual matches (e.g., "CCF" vs "Congestive heart failure with volume overload") are captured by CUI matching, but the style gap drives the majority of non-matches.

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
| Agreed | 54 |
| Disagreed | 23 |
| **Observed agreement (Po)** | **0.7013** |
| Expected agreement (Pe) | 0.5810 |
| **Cohen's Kappa** | **0.29** |
| **PABAK** | **0.40** |
| Interpretation | Fair |

**Match level breakdown (scored rows):**

| Level | Count |
|---|---|
| ITA match | 26 |
| Word overlap | 24 |
| Fuzzy match | 4 |
| No match | 23 |

### Results: General Line of Treatment (Ayurvedic)

| Metric | Value |
|---|---|
| Total rows | 80 |
| Skipped (one/both empty) | 69 |
| N (scored) | 11 |
| Agreed | 10 |
| Disagreed | 1 |
| **Observed agreement (Po)** | **0.9091** |
| Expected agreement (Pe) | 0.8347 |
| **Cohen's Kappa** | **0.45** |
| **PABAK** | **0.82** |
| Interpretation | Moderate |

**Note:** Only 11 of 80 rows had both raters providing treatment. The first 9 rows (identical between files) are the main contributors; the high agreement may not generalize.

**Match level breakdown (scored rows):**

| Level | Count |
|---|---|
| ITA match | 3 |
| Word overlap | 6 |
| Fuzzy match | 1 |
| No match | 1 |

### Discussion (Ayurvedic)
Ayurvedic diagnoses show substantially better agreement (Po=0.70, PABAK=0.40) than modern diagnoses (Po=0.42, PABAK=-0.15). This is partly because:
1. **Shared vocabulary**: Both raters use standardized Sanskrit terminology (e.g., "amavata", "pakshaghata", "grahani") that maps to ITA entries.
2. **Less stylistic divergence**: Both raters write concise Ayurvedic terms rather than one being verbose.
3. **Dosha-level agreement**: Even when specific diagnoses differ, raters often agree on the underlying dosha pattern (vata/pitta/kapha), contributing to word overlap matches.

Treatment agreement is high (Po=0.91) but based on only 11 scorable rows, since Rater 2 left treatment blank for most cases after row 9.

---

## 3. Comparative Summary

| Field | System | N (scored) | Po | Kappa | PABAK | Interpretation |
|---|---|---|---|---|---|---|
| Diagnosis | Modern | 59 | 0.42 | -0.18 | -0.15 | Poor |
| Diagnosis | Ayurvedic | 77 | 0.70 | 0.29 | 0.40 | Fair |
| Treatment | Modern | 53 | 0.21 | -1.41 | -0.58 | Poor |
| Treatment | Ayurvedic | 11 | 0.91 | 0.45 | 0.82 | Moderate* |

*\*Small sample (N=11) — interpret with caution.*

### Key Takeaways
1. **Ayurvedic diagnoses are more reproducible** than modern medical diagnoses across these two raters, likely due to a more constrained vocabulary (Sanskrit terminology vs. free-form English).
2. **Modern treatment agreement is very poor** because of fundamental style differences (drug names vs. management plans).
3. **PABAK is the more appropriate metric** for this study given the binary (agree/disagree) coding and skewed prevalence.
4. **Missing data is substantial** — particularly for Ayurvedic treatments (only 14% scorable) and modern diagnoses (25% empty from one rater).

---

## Reproducibility

```bash
cd irr/

# Modern IRR (requires UMLS API key and requests package)
python3 irr_modern.py --api-key YOUR_UMLS_API_KEY --skip-hierarchical

# Ayurvedic IRR (no external dependencies)
python3 irr_ayurveda.py
```
