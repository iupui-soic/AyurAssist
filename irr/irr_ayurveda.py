#!/usr/bin/env python3
"""
IRR Calculation: Cohen's Kappa for Ayurvedic Diagnoses & Treatments

Compares two clinicians' Ayurvedic diagnoses and treatments across 80 patient
narratives using WHO-ITA terminology normalization and Levenshtein-based
fuzzy matching.
"""

import argparse
import csv
import io
import os
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ITA_FILE = os.path.join(os.path.dirname(__file__), "..", "who-ita", "ita_terms_ascii.csv")

# Common Ayurvedic stopwords / modifiers to ignore during matching
AYUR_STOPWORDS = {
    "vata", "pitta", "kapha",  # doshas — too generic alone
    "dosha", "dushti", "vikara", "roga", "vyadhi",  # generic disease terms
    "samana", "shamana", "hara",  # generic treatment suffixes
    "chikitsa", "therapy", "treatment",
    "the", "a", "an", "of", "in", "on", "for", "to", "is", "and", "or",
    "with", "due", "type",
}

# Preamble phrases to strip
PREAMBLE_PATTERNS = [
    r"^\?\s*",
    r"^\?\?\s*",
    r"^likely\s+",
    r"^probable\s+",
    r"^possible\s+",
]


# ---------------------------------------------------------------------------
# Levenshtein
# ---------------------------------------------------------------------------
def levenshtein(s1, s2):
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def normalized_similarity(s1, s2):
    """Normalized Levenshtein similarity in [0, 1]."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    return 1.0 - levenshtein(s1, s2) / max_len


# ---------------------------------------------------------------------------
# ITA Vocabulary
# ---------------------------------------------------------------------------
class ITAVocabulary:
    """WHO-ITA terminology loaded from ita_terms_ascii.csv."""

    def __init__(self, filepath):
        self.terms = {}       # ita_id -> (english, sanskrit_list)
        self.sanskrit_index = {}  # lowercase sanskrit term -> ita_id
        self.english_index = {}   # lowercase english term -> ita_id
        self._load(filepath)

    def _load(self, filepath):
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(io.StringIO(f.read()))
            rows = list(reader)

        for row in rows[1:]:  # skip header
            if len(row) < 3:
                continue
            ita_id = row[0].strip()
            english = row[1].strip()
            sanskrit_raw = row[2].strip()

            # Parse multiple Sanskrit terms (separated by ; or ,)
            sanskrit_terms = []
            for part in re.split(r"[;,]", sanskrit_raw):
                term = re.sub(r"^\d+\.\s*", "", part).strip()  # remove "1.", "2." prefixes
                term = term.rstrip("/").strip()
                if term:
                    sanskrit_terms.append(term)

            self.terms[ita_id] = (english, sanskrit_terms)

            # Index
            for st in sanskrit_terms:
                key = st.lower().replace("-", "").replace(" ", "")
                self.sanskrit_index[key] = ita_id
            eng_key = english.lower().strip()
            self.english_index[eng_key] = ita_id

    def find_best_match(self, term, threshold=0.75):
        """
        Find the best ITA match for a clinical term.
        Returns: (ita_id, matched_term, similarity) or None
        """
        term_clean = term.lower().replace("-", "").replace(" ", "")

        # Exact match in Sanskrit index
        if term_clean in self.sanskrit_index:
            ita_id = self.sanskrit_index[term_clean]
            return ita_id, term, 1.0

        # Fuzzy match against Sanskrit terms
        best = None
        best_sim = 0.0
        for key, ita_id in self.sanskrit_index.items():
            sim = normalized_similarity(term_clean, key)
            if sim > best_sim:
                best_sim = sim
                best = (ita_id, key, sim)

        if best and best_sim >= threshold:
            return best

        # Try English index
        term_lower = term.lower().strip()
        if term_lower in self.english_index:
            ita_id = self.english_index[term_lower]
            return ita_id, term, 1.0

        return None

    def get_ita_ids_for_terms(self, terms, threshold=0.75):
        """Resolve a list of terms to ITA IDs."""
        results = {}
        for term in terms:
            match = self.find_best_match(term, threshold)
            if match:
                ita_id, matched, sim = match
                results[term] = (ita_id, sim)
        return results


# ---------------------------------------------------------------------------
# CSV Parsing
# ---------------------------------------------------------------------------
def read_csv(filepath):
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
        # row[2] is Modern Correlation — not used for Ayurvedic IRR
        treatment = row[3].strip()

        if not narrative and not diagnosis and not treatment:
            continue

        results.append((narrative, diagnosis, treatment))

    return results


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------
def normalize_text(text):
    """Clean and normalize Ayurvedic clinical text."""
    if not text:
        return ""
    # Normalize unicode dashes
    text = text.replace("\u2011", "-").replace("\u2010", "-")
    # Replace newlines
    text = re.sub(r"\n\s*\n", "; ", text)
    text = re.sub(r"\n", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_preamble(term):
    """Remove preamble like '?'."""
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

    # Remove parenthetical explanations longer than 30 chars
    text = re.sub(r"\([^)]{30,}\)", "", text)

    # Split on semicolons, commas, slashes, and/or
    parts = re.split(r"[;,/]|\band/or\b|\bor\b(?!\w)", text)

    cleaned = []
    for t in parts:
        t = strip_preamble(t)
        # Remove leading/trailing punctuation and numbers like "1)" "2)"
        t = re.sub(r"^[\s\?\!\.\,\:\d\)]+|[\s\?\!\.\,\:]+$", "", t)
        # Remove short parenthetical notes
        t = re.sub(r"\([^)]*\)", "", t).strip()
        if t and len(t) > 1:
            cleaned.append(t)

    return cleaned


# ---------------------------------------------------------------------------
# Matching Logic
# ---------------------------------------------------------------------------
def get_significant_words(terms):
    """Extract significant clinical words from terms (excluding stopwords)."""
    words = set()
    for term in terms:
        for w in re.split(r"[\s\-/]+", term.lower()):
            w = re.sub(r"[^a-z]", "", w)
            if w and len(w) > 2 and w not in AYUR_STOPWORDS:
                words.add(w)
    return words


def fuzzy_term_match(terms1, terms2, threshold=0.80):
    """Check if any pair of terms are fuzzy matches (Levenshtein)."""
    for t1 in terms1:
        t1_clean = t1.lower().strip()
        if len(t1_clean) < 3:
            continue
        for t2 in terms2:
            t2_clean = t2.lower().strip()
            if len(t2_clean) < 3:
                continue
            sim = normalized_similarity(t1_clean, t2_clean)
            if sim >= threshold:
                return True, sim
    return False, 0.0


def match_terms(terms1, terms2, ita_vocab):
    """
    Match two sets of Ayurvedic terms.
    Returns: (agree: bool | None, match_level: str)
      None = skip (one or both empty)
    Tiers:
      1. ITA ID match — both resolve to same ITA term
      2. Word overlap — significant Sanskrit word shared
      3. Fuzzy match — Levenshtein similarity >= 0.80
    """
    is_empty1 = len(terms1) == 0
    is_empty2 = len(terms2) == 0

    if is_empty1 and is_empty2:
        return None, "both_empty"
    if is_empty1 or is_empty2:
        return None, "one_empty"

    # Tier 1: ITA ID match
    ita1 = ita_vocab.get_ita_ids_for_terms(terms1)
    ita2 = ita_vocab.get_ita_ids_for_terms(terms2)

    ids1 = {v[0] for v in ita1.values()}
    ids2 = {v[0] for v in ita2.values()}

    if ids1 & ids2:
        return True, "ita_match"

    # Tier 2: Significant word overlap
    words1 = get_significant_words(terms1)
    words2 = get_significant_words(terms2)
    if words1 & words2:
        return True, "word_overlap"

    # Tier 3: Fuzzy Levenshtein match
    matched, sim = fuzzy_term_match(terms1, terms2, threshold=0.80)
    if matched:
        return True, "fuzzy_match"

    return False, "no_match"


# ---------------------------------------------------------------------------
# Cohen's Kappa & PABAK
# ---------------------------------------------------------------------------
def compute_kappa(agreements):
    """Compute Cohen's Kappa and PABAK from binary agreements list."""
    n = len(agreements)
    if n == 0:
        return {"n": 0, "agree": 0, "disagree": 0, "po": 0, "pe": 0,
                "kappa": 0, "pabak": 0}

    agree_count = sum(agreements)
    disagree_count = n - agree_count
    po = agree_count / n

    p_agree = agree_count / n
    p_disagree = disagree_count / n
    pe = p_agree**2 + p_disagree**2

    kappa = (po - pe) / (1 - pe) if pe < 1.0 else 1.0
    pabak = 2 * po - 1

    return {
        "n": n, "agree": agree_count, "disagree": disagree_count,
        "po": po, "pe": pe, "kappa": kappa, "pabak": pabak,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def truncate(s, maxlen=50):
    s = s.replace("\n", " ").strip()
    return s[:maxlen] + "..." if len(s) > maxlen else s


def interpret_kappa(k):
    if k < 0:
        return "Poor"
    elif k < 0.21:
        return "Slight"
    elif k < 0.41:
        return "Fair"
    elif k < 0.61:
        return "Moderate"
    elif k < 0.81:
        return "Substantial"
    else:
        return "Almost Perfect"


def print_agreement_table(results, field_name):
    print(f"\n{'='*130}")
    print(f"  Per-Row Agreement: {field_name}")
    print(f"{'='*130}")
    header = f"{'Row':>4}  {'Rater1':<50}  {'Rater2':<50}  {'Agree':>6}  {'Level':<15}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r["agree"] is None:
            agree_str = "SKIP"
        elif r["agree"]:
            agree_str = "YES"
        else:
            agree_str = "NO"
        print(
            f"{r['row']:>4}  {truncate(r['r1_text']):<50}  "
            f"{truncate(r['r2_text']):<50}  {agree_str:>6}  {r['level']:<15}"
        )


def print_summary(stats, field_name, level_counts, total_rows):
    skipped = total_rows - stats["n"]
    print(f"\n{'='*60}")
    print(f"  Summary: {field_name}")
    print(f"{'='*60}")
    print(f"  Total rows:            {total_rows}")
    print(f"  Skipped (empty):       {skipped}  (both_empty={level_counts.get('both_empty',0)}, one_empty={level_counts.get('one_empty',0)})")
    print(f"  N (scored):            {stats['n']}")
    print(f"  Agreed:                {stats['agree']}")
    print(f"  Disagreed:             {stats['disagree']}")
    print(f"  Observed agreement:    {stats['po']:.4f}")
    print(f"  Expected agreement:    {stats['pe']:.4f}")
    print(f"  Cohen's Kappa:         {stats['kappa']:.4f}")
    print(f"  PABAK:                 {stats['pabak']:.4f}")
    print(f"\n  Match Level Breakdown (scored rows only):")
    for level, count in sorted(level_counts.items()):
        if level not in ("both_empty", "one_empty"):
            print(f"    {level:<20}: {count}")
    print(f"\n  Skipped Breakdown:")
    for level in ("both_empty", "one_empty"):
        if level in level_counts:
            print(f"    {level:<20}: {level_counts[level]}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Calculate Cohen's Kappa IRR for Ayurvedic diagnoses"
    )
    parser.add_argument(
        "--file1", default="ayurveda1.csv",
        help="First rater CSV (default: ayurveda1.csv)",
    )
    parser.add_argument(
        "--file2", default="ayurveda2.csv",
        help="Second rater CSV (default: ayurveda2.csv)",
    )
    parser.add_argument(
        "--ita", default=ITA_FILE,
        help=f"ITA terms file (default: {ITA_FILE})",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Levenshtein similarity threshold (default: 0.80)",
    )
    args = parser.parse_args()

    # Load ITA vocabulary
    print(f"Loading ITA vocabulary from {args.ita}...")
    ita_vocab = ITAVocabulary(args.ita)
    print(f"  {len(ita_vocab.terms)} ITA terms loaded")
    print(f"  {len(ita_vocab.sanskrit_index)} Sanskrit index entries")

    # Parse CSVs
    print(f"\nParsing {args.file1}...")
    rows1 = read_csv(args.file1)
    print(f"  {len(rows1)} data rows")

    print(f"Parsing {args.file2}...")
    rows2 = read_csv(args.file2)
    print(f"  {len(rows2)} data rows")

    n = min(len(rows1), len(rows2))
    if len(rows1) != len(rows2):
        print(f"\nWARNING: Row count mismatch ({len(rows1)} vs {len(rows2)}). Using first {n} rows.")

    print(f"\nProcessing {n} rows...\n")

    diag_results = []
    treat_results = []

    for i in range(n):
        _, diag1, treat1 = rows1[i]
        _, diag2, treat2 = rows2[i]

        row_num = i + 1

        diag_terms1 = split_terms(diag1)
        diag_terms2 = split_terms(diag2)
        treat_terms1 = split_terms(treat1)
        treat_terms2 = split_terms(treat2)

        d_agree, d_level = match_terms(diag_terms1, diag_terms2, ita_vocab)
        t_agree, t_level = match_terms(treat_terms1, treat_terms2, ita_vocab)

        diag_results.append({
            "row": row_num, "r1_text": diag1, "r2_text": diag2,
            "agree": d_agree, "level": d_level,
        })
        treat_results.append({
            "row": row_num, "r1_text": treat1, "r2_text": treat2,
            "agree": t_agree, "level": t_level,
        })

        print(f"  Row {row_num}/{n}: Dx={d_level:<15} Tx={t_level:<15}")

    # Compute stats
    diag_scored = [r["agree"] for r in diag_results if r["agree"] is not None]
    treat_scored = [r["agree"] for r in treat_results if r["agree"] is not None]

    diag_stats = compute_kappa(diag_scored)
    treat_stats = compute_kappa(treat_scored)

    diag_levels = Counter(r["level"] for r in diag_results)
    treat_levels = Counter(r["level"] for r in treat_results)

    print_agreement_table(diag_results, "Ayurvedic Diagnosis")
    print_summary(diag_stats, "Ayurvedic Diagnosis", diag_levels, n)
    print(f"  Interpretation: {interpret_kappa(diag_stats['kappa'])}")

    print_agreement_table(treat_results, "General Line of Treatment")
    print_summary(treat_stats, "General Line of Treatment", treat_levels, n)
    print(f"  Interpretation: {interpret_kappa(treat_stats['kappa'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
