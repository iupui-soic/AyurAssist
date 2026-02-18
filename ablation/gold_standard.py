"""
Load and parse the 80 gold-standard clinical vignettes from two rater CSVs.

Uses the union of both raters' annotations: a prediction is correct if it
matches EITHER rater's diagnosis or treatment.
"""

import csv
import io
import re
from dataclasses import dataclass, field

from configs import AYURVEDA1_CSV, AYURVEDA2_CSV

# ---------------------------------------------------------------------------
# Preamble / normalization (reused from irr_ayurveda.py)
# ---------------------------------------------------------------------------
PREAMBLE_PATTERNS = [
    r"^\?\s*",
    r"^\?\?\s*",
    r"^likely\s+",
    r"^probable\s+",
    r"^possible\s+",
]


def normalize_text(text):
    if not text:
        return ""
    text = text.replace("\u2011", "-").replace("\u2010", "-")
    text = re.sub(r"\n\s*\n", "; ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_preamble(term):
    term = term.strip()
    for pattern in PREAMBLE_PATTERNS:
        term = re.sub(pattern, "", term, flags=re.IGNORECASE).strip()
    return term


def split_terms(text):
    """Split a multi-term Ayurvedic cell into individual terms."""
    if not text:
        return []
    text = normalize_text(text)
    if not text:
        return []
    text = re.sub(r"\([^)]{30,}\)", "", text)
    parts = re.split(r"[;,/]|\band/or\b|\bor\b(?!\w)", text)
    cleaned = []
    for t in parts:
        t = strip_preamble(t)
        t = re.sub(r"^[\s\?\!\.\,\:\d\)]+|[\s\?\!\.\,\:]+$", "", t)
        t = re.sub(r"\([^)]*\)", "", t).strip()
        if t and len(t) > 1:
            cleaned.append(t)
    return cleaned


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Vignette:
    index: int
    narrative: str
    rater1_diagnosis: str
    rater2_diagnosis: str
    rater1_treatment: str
    rater2_treatment: str


def _read_csv(filepath):
    """Read Ayurveda CSV. Returns list of (narrative, diagnosis, treatment)."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        return []
    data_rows = rows[1:]  # skip header
    results = []
    for row in data_rows:
        if len(row) < 4:
            row.extend([""] * (4 - len(row)))
        narrative = row[0].strip()
        diagnosis = row[1].strip()
        # row[2] is Modern Correlation — not used
        treatment = row[3].strip()
        if not narrative and not diagnosis and not treatment:
            continue
        results.append((narrative, diagnosis, treatment))
    return results


def load_vignettes(csv1=AYURVEDA1_CSV, csv2=AYURVEDA2_CSV):
    """Load vignettes from both rater CSVs, paired by index."""
    rows1 = _read_csv(csv1)
    rows2 = _read_csv(csv2)
    n = min(len(rows1), len(rows2))

    vignettes = []
    for i in range(n):
        narr1, diag1, treat1 = rows1[i]
        narr2, diag2, treat2 = rows2[i]
        # Use rater1's narrative (they should be identical)
        vignettes.append(Vignette(
            index=i,
            narrative=narr1 or narr2,
            rater1_diagnosis=diag1,
            rater2_diagnosis=diag2,
            rater1_treatment=treat1,
            rater2_treatment=treat2,
        ))
    return vignettes


def get_gold_terms(vignette, field="diagnosis"):
    """Get union of both raters' terms for a field.

    Returns a set of individual terms (split and cleaned).
    """
    if field == "diagnosis":
        text1 = vignette.rater1_diagnosis
        text2 = vignette.rater2_diagnosis
    elif field == "treatment":
        text1 = vignette.rater1_treatment
        text2 = vignette.rater2_treatment
    else:
        raise ValueError(f"Unknown field: {field}")

    terms1 = split_terms(text1)
    terms2 = split_terms(text2)
    return set(terms1) | set(terms2)
