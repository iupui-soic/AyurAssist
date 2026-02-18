#!/usr/bin/env python3
"""
IRR Calculation: Cohen's Kappa for Modern Medical Diagnoses

Compares two clinicians' diagnoses and treatments across 80 patient narratives
using UMLS-based semantic matching and Cohen's Kappa.
"""

import argparse
import csv
import io
import json
import os
import re
import time
from collections import Counter

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package required. Install with: pip install requests")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UMLS_SEARCH_URL = "https://uts-ws.nlm.nih.gov/rest/search/current"
UMLS_CONTENT_URL = "https://uts-ws.nlm.nih.gov/rest/content/current"
CACHE_FILE = "umls_cache.json"
RATE_LIMIT_DELAY = 0.05  # 50ms between requests (20 req/sec)
INSTRUCTION_MARKERS = [
    "give single ayurvedic diagnosis",
    "acc to the patient narratives",
]

# Common medical abbreviations -> expanded form for UMLS lookup
ABBREVIATIONS = {
    "CCF": "Congestive cardiac failure",
    "CHF": "Congestive heart failure",
    "IBD": "Inflammatory bowel disease",
    "COPD": "Chronic obstructive pulmonary disease",
    "DMARDs": "Disease-modifying antirheumatic drugs",
    "DMARDS": "Disease-modifying antirheumatic drugs",
    "NSAIDs": "Nonsteroidal anti-inflammatory drugs",
    "NSAIDS": "Nonsteroidal anti-inflammatory drugs",
    "DM": "Diabetes mellitus",
    "DM-2": "Type 2 diabetes mellitus",
    "UTI": "Urinary tract infection",
    "DVT": "Deep vein thrombosis",
    "PCOD": "Polycystic ovarian disease",
    "PCOS": "Polycystic ovary syndrome",
    "ADEM": "Acute disseminated encephalomyelitis",
    "ARDS": "Acute respiratory distress syndrome",
    "ALI": "Acute lung injury",
    "ATT": "Antitubercular therapy",
    "ICS": "Inhaled corticosteroids",
    "SABA": "Short-acting beta-agonist",
    "PPI": "Proton pump inhibitor",
    "ACE": "Angiotensin-converting enzyme",
    "ARB": "Angiotensin receptor blocker",
    "OA": "Osteoarthritis",
    "RA": "Rheumatoid arthritis",
    "SLE": "Systemic lupus erythematosus",
    "TB": "Tuberculosis",
    "CKD": "Chronic kidney disease",
    "RRT": "Renal replacement therapy",
    "IVF": "In vitro fertilization",
    "IUI": "Intrauterine insemination",
    "UAE": "Uterine artery embolization",
    "5-ASA": "Mesalamine",
    "MRI": "Magnetic resonance imaging",
    "CT": "Computed tomography",
    "USG": "Ultrasonography",
    "ECG": "Electrocardiography",
    # Common misspellings
    "Rhematoid arthritis": "Rheumatoid arthritis",
    "gastroentritis": "Gastroenteritis",
    "Acute gastroentritis": "Acute gastroenteritis",
    "physitherapy": "Physiotherapy",
    "Amitryptilline": "Amitriptyline",
    "Sulfasalasine": "Sulfasalazine",
}

# Preamble phrases to strip from diagnosis text
PREAMBLE_PATTERNS = [
    r"^most likely\s+",
    r"^likely\s+",
    r"^consider\s+",
    r"^rule out\s+",
    r"^r/o\s+",
    r"^possible\s+",
    r"^probable\s+",
    r"^suspect\s+",
    r"^suggestive of\s+",
    r"^consistent with\s+",
    r"^features of\s+",
    r"^secondary to\s+",
    r"^due to\s+",
    r"^\?\s*",
    r"^\?\?\s*",
]


# ---------------------------------------------------------------------------
# UMLS Cache
# ---------------------------------------------------------------------------
class UMLSCache:
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self.data = {}
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.data = {}

    def save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value


# ---------------------------------------------------------------------------
# CSV Parsing
# ---------------------------------------------------------------------------
def read_csv(filepath):
    """Read CSV handling BOM, multiline fields. Returns list of (narrative, diagnosis, treatment)."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        content = f.read()

    reader = csv.reader(io.StringIO(content))
    rows = list(reader)

    if not rows:
        return []

    # Skip header row
    data_rows = rows[1:]

    results = []
    for row in data_rows:
        if len(row) < 3:
            row.extend([""] * (3 - len(row)))
        narrative = row[0].strip()
        diagnosis = row[1].strip()
        treatment = row[2].strip()

        # Skip instruction rows
        if is_instruction_row(narrative, diagnosis):
            continue

        # Skip completely empty rows
        if not narrative and not diagnosis and not treatment:
            continue

        results.append((narrative, diagnosis, treatment))

    return results


def is_instruction_row(narrative, diagnosis):
    """Check if this row is an instruction row (not patient data)."""
    combined = (narrative + " " + diagnosis).lower()
    return any(marker in combined for marker in INSTRUCTION_MARKERS)


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------
def normalize_text(text):
    """Clean and normalize clinical text."""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Normalize unicode dashes (en-dash, non-breaking hyphen) to regular hyphen
    text = text.replace("\u2011", "-").replace("\u2010", "-")
    # Replace newlines/paragraph breaks with semicolons to preserve structure
    text = re.sub(r"\n\s*\n", "; ", text)
    text = re.sub(r"\n", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_preamble(term):
    """Remove preamble phrases like 'most likely', 'consider', '?'."""
    term = term.strip()
    for pattern in PREAMBLE_PATTERNS:
        term = re.sub(pattern, "", term, flags=re.IGNORECASE).strip()
    return term


def expand_abbreviation(term):
    """Expand known medical abbreviations and fix common misspellings."""
    stripped = term.strip()
    # Exact match
    if stripped in ABBREVIATIONS:
        return ABBREVIATIONS[stripped]
    if stripped.upper() in ABBREVIATIONS:
        return ABBREVIATIONS[stripped.upper()]
    # Case-insensitive match for multi-word terms
    stripped_lower = stripped.lower()
    for key, val in ABBREVIATIONS.items():
        if key.lower() == stripped_lower:
            return val
    return stripped


def split_terms(text):
    """Split a multi-diagnosis/treatment cell into individual terms."""
    if not text:
        return []

    text = normalize_text(text)
    if not text:
        return []

    # Remove trailing differential/explanation clauses ONLY if they appear
    # after some primary diagnosis (not at the very start of text)
    diff_pattern = re.compile(
        r"\b(?:Also consider|Differentiate from|Need to rule out|Must rule out|"
        r"must strongly suspect|must evaluate for|Other possibilities|"
        r"Less likely|Consider also)\b",
        re.IGNORECASE,
    )
    m = diff_pattern.search(text)
    if m and m.start() > 10:  # Only strip if pattern is not near the start
        text = text[: m.start()]

    # Remove long parenthetical explanations (more than 30 chars)
    text = re.sub(r"\([^)]{30,}\)", "", text)

    # Split on em-dash / en-dash separator (rater 2 uses " – " to separate dx from explanation)
    text = re.split(r"\s*[–—]\s*", text)[0]

    # Split on sentence boundaries after extracting core
    text = re.split(r"\.\s+[A-Z]", text)[0]

    # Split on common delimiters: semicolons, commas, "and/or", "or", " and "
    parts = re.split(r"[;/]|\band/or\b|\bor\b(?!\w)", text)

    # Further split on commas
    final_parts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "," in part:
            subparts = [s.strip() for s in part.split(",")]
            if all(len(s.split()) <= 8 for s in subparts if s):
                final_parts.extend(subparts)
            else:
                final_parts.append(part)
        else:
            final_parts.append(part)

    # Clean each term
    cleaned = []
    for t in final_parts:
        t = strip_preamble(t)
        t = expand_abbreviation(t)
        # Remove leading/trailing punctuation
        t = re.sub(r"^[\s\?\!\.\,\:\d\)]+|[\s\?\!\.\,\:]+$", "", t)
        # Strip trailing modifiers
        core = re.split(
            r"\s+(?:with|due to|secondary to|on background of|after|leading|in a)\s+",
            t,
            maxsplit=1,
        )[0].strip()
        # Strip route-of-administration prefixes for treatments
        core = re.sub(
            r"^(?:IV|IM|oral|inhaled|topical|intralesional|intrathecal|subcutaneous)\s+",
            "",
            core,
            flags=re.IGNORECASE,
        ).strip()
        # Strip "start" / "urgent" / "aggressive" prefixes
        core = re.sub(
            r"^(?:start|urgent|aggressive|long-term|short)\s+",
            "",
            core,
            flags=re.IGNORECASE,
        ).strip()
        if core and len(core) > 2:
            cleaned.append(core)

    return cleaned


# ---------------------------------------------------------------------------
# UMLS API
# ---------------------------------------------------------------------------
def get_umls_ticket(api_key):
    """Get a single-use service ticket (UMLS REST API uses API key directly now)."""
    # Modern UMLS REST API accepts apiKey as query parameter
    return api_key


def search_umls(term, api_key, cache):
    """Search UMLS for a term, return list of CUIs."""
    cache_key = f"search:{term.lower().strip()}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    time.sleep(RATE_LIMIT_DELAY)

    try:
        params = {
            "apiKey": api_key,
            "string": term,
            "searchType": "words",
            "returnIdType": "concept",
            "pageSize": 3,
        }
        resp = requests.get(UMLS_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        cuis = []
        results = data.get("result", {}).get("results", [])
        for r in results:
            cui = r.get("ui", "")
            if cui and cui != "NONE":
                cuis.append(cui)

        cache.set(cache_key, cuis)
        return cuis

    except requests.RequestException as e:
        print(f"  UMLS search error for '{term}': {e}")
        cache.set(cache_key, [])
        return []


def get_cui_ancestors(cui, api_key, cache, max_depth=2):
    """Get ancestor CUIs up to max_depth levels via UMLS relations."""
    cache_key = f"ancestors:{cui}:{max_depth}"
    cached = cache.get(cache_key)
    if cached is not None:
        return set(cached)

    ancestors = set()
    current_level = {cui}

    for depth in range(max_depth):
        next_level = set()
        for c in current_level:
            time.sleep(RATE_LIMIT_DELAY)
            try:
                url = f"{UMLS_CONTENT_URL}/CUI/{c}/relations"
                params = {
                    "apiKey": api_key,
                    "pageSize": 25,
                }
                resp = requests.get(url, params=params, timeout=30)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for rel in data.get("result", []):
                    rel_label = rel.get("relationLabel", "")
                    if rel_label in ("PAR", "RB"):  # parent / broader
                        related_uri = rel.get("relatedId", "")
                        if related_uri:
                            parent_cui = related_uri.rstrip("/").split("/")[-1]
                            if parent_cui.startswith("C"):
                                ancestors.add(parent_cui)
                                next_level.add(parent_cui)
            except requests.RequestException:
                continue
        current_level = next_level
        if not current_level:
            break

    cache.set(cache_key, list(ancestors))
    return ancestors


# ---------------------------------------------------------------------------
# Matching Logic
# ---------------------------------------------------------------------------
def search_umls_approximate(term, api_key, cache):
    """Search UMLS with approximate matching (handles misspellings)."""
    cache_key = f"search_approx:{term.lower().strip()}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    time.sleep(RATE_LIMIT_DELAY)
    try:
        params = {
            "apiKey": api_key,
            "string": term,
            "searchType": "approximate",
            "returnIdType": "concept",
            "pageSize": 3,
        }
        resp = requests.get(UMLS_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cuis = []
        for r in data.get("result", {}).get("results", []):
            cui = r.get("ui", "")
            if cui and cui != "NONE":
                cuis.append(cui)
        cache.set(cache_key, cuis)
        return cuis
    except requests.RequestException:
        cache.set(cache_key, [])
        return []


def resolve_terms_to_cuis(terms, api_key, cache):
    """Resolve a list of clinical terms to a set of CUIs, with fallbacks."""
    all_cuis = {}
    for term in terms:
        # Try full term (words search)
        cuis = search_umls(term, api_key, cache)
        if cuis:
            all_cuis[term] = cuis
            continue

        # Try abbreviation expansion
        expanded = expand_abbreviation(term)
        if expanded != term:
            cuis = search_umls(expanded, api_key, cache)
            if cuis:
                all_cuis[expanded] = cuis
                continue

        # Try approximate search (handles misspellings)
        cuis = search_umls_approximate(term, api_key, cache)
        if cuis:
            all_cuis[term] = cuis
            continue

        # Try truncated to first 4 words (handles verbose descriptions)
        words = term.split()
        if len(words) > 4:
            shorter = " ".join(words[:4])
            cuis = search_umls(shorter, api_key, cache)
            if cuis:
                all_cuis[shorter] = cuis
                continue

        # Try first 3 words
        if len(words) > 3:
            shorter = " ".join(words[:3])
            cuis = search_umls(shorter, api_key, cache)
            if cuis:
                all_cuis[shorter] = cuis
                continue

    return all_cuis


def levenshtein(s1, s2):
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def fuzzy_term_match(terms1, terms2, threshold=0.8):
    """
    Check if any pair of terms from two lists are fuzzy matches.
    Uses normalized Levenshtein similarity.
    Returns True if any pair has similarity >= threshold.
    """
    for t1 in terms1:
        t1_lower = t1.lower().strip()
        if len(t1_lower) < 4:
            continue
        for t2 in terms2:
            t2_lower = t2.lower().strip()
            if len(t2_lower) < 4:
                continue
            max_len = max(len(t1_lower), len(t2_lower))
            dist = levenshtein(t1_lower, t2_lower)
            similarity = 1 - (dist / max_len)
            if similarity >= threshold:
                return True
    return False


# Stopwords to ignore in word-overlap matching
CLINICAL_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "is", "are", "was",
    "with", "and", "or", "not", "no", "from", "by", "at", "as", "this",
    "that", "be", "has", "had", "have", "most", "likely", "chronic",
    "acute", "severe", "mild", "moderate", "primary", "secondary",
    "type", "related", "associated", "bilateral", "unilateral", "left",
    "right", "upper", "lower", "early", "late", "advanced", "progressive",
    "recurrent", "persistent", "long-standing",
}


def get_clinical_words(terms):
    """Extract significant clinical words from a list of terms."""
    words = set()
    for term in terms:
        for w in re.split(r"[\s\-/]+", term.lower()):
            w = re.sub(r"[^a-z]", "", w)
            if w and len(w) > 3 and w not in CLINICAL_STOPWORDS:
                words.add(w)
    return words


def match_terms(terms1, terms2, api_key, cache):
    """
    Match two sets of clinical terms using UMLS + word overlap.
    Returns: (agree: bool | None, match_level: str)
      - None means "skip" (one or both raters left field empty)
      - True/False means agree/disagree
    match_level: 'exact_cui', 'word_overlap', 'hierarchical',
                 'both_empty', 'one_empty', 'no_match'
    """
    is_empty1 = len(terms1) == 0
    is_empty2 = len(terms2) == 0

    if is_empty1 and is_empty2:
        return None, "both_empty"
    if is_empty1 or is_empty2:
        return None, "one_empty"

    # Resolve terms to CUIs
    cuis1 = resolve_terms_to_cuis(terms1, api_key, cache)
    cuis2 = resolve_terms_to_cuis(terms2, api_key, cache)

    # Tier 1: Exact CUI match
    all_cuis_1 = set()
    for cui_list in cuis1.values():
        all_cuis_1.update(cui_list)

    all_cuis_2 = set()
    for cui_list in cuis2.values():
        all_cuis_2.update(cui_list)

    if all_cuis_1 & all_cuis_2:
        return True, "exact_cui"

    # Tier 2: Significant clinical word overlap
    words1 = get_clinical_words(terms1)
    words2 = get_clinical_words(terms2)
    if words1 & words2:
        return True, "word_overlap"

    # Tier 3: Fuzzy term match (catches typos like gastroentritis/gastroenteritis)
    if fuzzy_term_match(terms1, terms2, threshold=0.8):
        return True, "fuzzy_match"

    # Tier 4: Hierarchical match (shared ancestor within depth 2)
    # Limit to top CUI per term to avoid combinatorial explosion
    top_cuis_1 = set()
    for cui_list in cuis1.values():
        if cui_list:
            top_cuis_1.add(cui_list[0])
    top_cuis_2 = set()
    for cui_list in cuis2.values():
        if cui_list:
            top_cuis_2.add(cui_list[0])

    for cui_a in top_cuis_1:
        ancestors_a = get_cui_ancestors(cui_a, api_key, cache) | {cui_a}
        for cui_b in top_cuis_2:
            ancestors_b = get_cui_ancestors(cui_b, api_key, cache) | {cui_b}
            if cui_b in ancestors_a or cui_a in ancestors_b:
                return True, "hierarchical"
            if ancestors_a & ancestors_b:
                return True, "hierarchical"

    return False, "no_match"


# ---------------------------------------------------------------------------
# Cohen's Kappa & PABAK
# ---------------------------------------------------------------------------
def compute_kappa(agreements):
    """
    Compute Cohen's Kappa and PABAK from a list of binary agreements.
    agreements: list of bool (True=agree, False=disagree)
    """
    n = len(agreements)
    if n == 0:
        return {"n": 0, "po": 0, "pe": 0, "kappa": 0, "pabak": 0}

    agree_count = sum(agreements)
    disagree_count = n - agree_count

    po = agree_count / n  # observed agreement

    # For Cohen's Kappa with binary agree/disagree:
    # We model each rater as independently choosing agree/disagree
    # p(agree by chance) = p1_agree * p2_agree + p1_disagree * p2_disagree
    # Since both raters produce the same agreement vector, we use marginals
    # In this simplified binary case:
    p_agree = agree_count / n
    p_disagree = disagree_count / n
    pe = p_agree**2 + p_disagree**2  # expected agreement by chance

    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)

    # PABAK: Prevalence-Adjusted Bias-Adjusted Kappa
    pabak = 2 * po - 1

    return {
        "n": n,
        "agree": agree_count,
        "disagree": disagree_count,
        "po": po,
        "pe": pe,
        "kappa": kappa,
        "pabak": pabak,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def truncate(s, maxlen=50):
    s = s.replace("\n", " ").strip()
    return s[:maxlen] + "..." if len(s) > maxlen else s


def print_agreement_table(results, field_name):
    """Print per-row agreement table."""
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
    """Print summary statistics."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Calculate Cohen's Kappa IRR for modern medical diagnoses"
    )
    parser.add_argument("--api-key", required=True, help="UMLS REST API key")
    parser.add_argument(
        "--file1", default="modern1.csv", help="First rater CSV (default: modern1.csv)"
    )
    parser.add_argument(
        "--file2",
        default="modern2.csv",
        help="Second rater CSV (default: modern2.csv)",
    )
    parser.add_argument(
        "--cache",
        default=CACHE_FILE,
        help=f"Cache file path (default: {CACHE_FILE})",
    )
    parser.add_argument(
        "--skip-hierarchical",
        action="store_true",
        help="Skip hierarchical matching (faster, only exact CUI)",
    )
    args = parser.parse_args()

    cache = UMLSCache(args.cache)

    # Verify API key works
    print("Verifying UMLS API key...")
    test_cuis = search_umls("diabetes", args.api_key, cache)
    if not test_cuis:
        print("ERROR: UMLS API key verification failed. Check your key.")
        return 1
    print(f"  API key valid (test: 'diabetes' -> {test_cuis[:2]})")

    # Parse CSVs
    print(f"\nParsing {args.file1}...")
    rows1 = read_csv(args.file1)
    print(f"  {len(rows1)} data rows")

    print(f"Parsing {args.file2}...")
    rows2 = read_csv(args.file2)
    print(f"  {len(rows2)} data rows")

    n = min(len(rows1), len(rows2))
    if len(rows1) != len(rows2):
        print(
            f"\nWARNING: Row count mismatch ({len(rows1)} vs {len(rows2)}). Using first {n} rows."
        )

    print(f"\nProcessing {n} rows...\n")

    diag_results = []
    treat_results = []

    for i in range(n):
        narrative1, diag1, treat1 = rows1[i]
        narrative2, diag2, treat2 = rows2[i]

        row_num = i + 1
        print(f"  Row {row_num}/{n}: ", end="", flush=True)

        # Parse terms
        diag_terms1 = split_terms(diag1)
        diag_terms2 = split_terms(diag2)
        treat_terms1 = split_terms(treat1)
        treat_terms2 = split_terms(treat2)

        # Match diagnoses
        if args.skip_hierarchical:
            d_agree, d_level = match_terms_exact_only(
                diag_terms1, diag_terms2, args.api_key, cache
            )
        else:
            d_agree, d_level = match_terms(
                diag_terms1, diag_terms2, args.api_key, cache
            )

        diag_results.append(
            {
                "row": row_num,
                "r1_text": diag1,
                "r2_text": diag2,
                "agree": d_agree,
                "level": d_level,
            }
        )

        # Match treatments
        if args.skip_hierarchical:
            t_agree, t_level = match_terms_exact_only(
                treat_terms1, treat_terms2, args.api_key, cache
            )
        else:
            t_agree, t_level = match_terms(
                treat_terms1, treat_terms2, args.api_key, cache
            )

        treat_results.append(
            {
                "row": row_num,
                "r1_text": treat1,
                "r2_text": treat2,
                "agree": t_agree,
                "level": t_level,
            }
        )

        print(f"Dx={d_level:<15} Tx={t_level:<15}")

        # Save cache periodically
        if row_num % 10 == 0:
            cache.save()

    # Final cache save
    cache.save()
    print(f"\nCache saved to {args.cache}")

    # Compute and display results
    # Only include rows where BOTH raters provided a value (agree is not None)
    diag_scored = [r["agree"] for r in diag_results if r["agree"] is not None]
    treat_scored = [r["agree"] for r in treat_results if r["agree"] is not None]

    diag_stats = compute_kappa(diag_scored)
    treat_stats = compute_kappa(treat_scored)

    diag_levels = Counter(r["level"] for r in diag_results)
    treat_levels = Counter(r["level"] for r in treat_results)

    print_agreement_table(diag_results, "Diagnosis")
    print_summary(diag_stats, "Diagnosis", diag_levels, n)
    print(f"  Interpretation: {interpret_kappa(diag_stats['kappa'])}")

    print_agreement_table(treat_results, "General Line of Treatment")
    print_summary(treat_stats, "General Line of Treatment", treat_levels, n)
    print(f"  Interpretation: {interpret_kappa(treat_stats['kappa'])}")

    return 0


def match_terms_exact_only(terms1, terms2, api_key, cache):
    """Exact CUI match + word overlap (no hierarchical), faster."""
    is_empty1 = len(terms1) == 0
    is_empty2 = len(terms2) == 0

    if is_empty1 and is_empty2:
        return None, "both_empty"
    if is_empty1 or is_empty2:
        return None, "one_empty"

    cuis1 = resolve_terms_to_cuis(terms1, api_key, cache)
    cuis2 = resolve_terms_to_cuis(terms2, api_key, cache)

    all_cuis_1 = set()
    for cui_list in cuis1.values():
        all_cuis_1.update(cui_list)

    all_cuis_2 = set()
    for cui_list in cuis2.values():
        all_cuis_2.update(cui_list)

    if all_cuis_1 & all_cuis_2:
        return True, "exact_cui"

    # Word overlap fallback
    words1 = get_clinical_words(terms1)
    words2 = get_clinical_words(terms2)
    if words1 & words2:
        return True, "word_overlap"

    # Fuzzy term match (catches typos)
    if fuzzy_term_match(terms1, terms2, threshold=0.8):
        return True, "fuzzy_match"

    return False, "no_match"


if __name__ == "__main__":
    raise SystemExit(main())
