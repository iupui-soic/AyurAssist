# Ablation Study Results

## Primary Metrics (over 80 vignettes)

| Config | Dx Accuracy | Dx ROUGE-L | Dx Token-F1 | Tx Accuracy | Tx ROUGE-L | Tx Token-F1 |
|--------|------------|-----------|-------------|------------|-----------|-------------|
| direct_llm | 0.750 (60/80) | 0.104 | 0.130 | 1.000 (80/80) | 0.034 | 0.083 |
| full_pipeline | 0.800 (64/80) | 0.065 | 0.088 | 1.000 (80/80) | 0.060 | 0.125 |
| bridge_only | 0.050 (4/80) | 0.003 | 0.003 | 0.000 (0/80) | 0.000 | 0.000 |

## Term-Level Metrics (micro-averaged over terms)

| Config | N | Dx P | Dx R | Dx F1 | Tx P | Tx R | Tx F1 |
|--------|---|------|------|-------|------|------|-------|
| direct_llm | 80 | 0.178 | 0.263 | 0.213 | 0.092 | 0.501 | 0.155 |
| full_pipeline | 80 | 0.060 | 0.346 | 0.102 | 0.144 | 0.455 | 0.219 |
| bridge_only | 80 | 0.022 | 0.013 | 0.016 | 0.000 | 0.000 | 0.000 |

## Match Tier Breakdown (Diagnosis)

| Config | ITA Match | Word Overlap | Fuzzy Match | No Match |
|--------|-----------|--------------|-------------|----------|
| direct_llm | 18 | 56 | 9 | 383 |
| full_pipeline | 24 | 77 | 8 | 1704 |
| bridge_only | 4 | 0 | 0 | 178 |
