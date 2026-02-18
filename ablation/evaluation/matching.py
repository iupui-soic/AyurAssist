"""
Tiered matching for comparing predicted terms against gold standard.

Term-level matching: each predicted term is matched against individual
gold terms using a greedy algorithm with tiered priority.

Tiers (checked in order):
  Tier 1: ITA ID match — both terms resolve to the same ITA term
  Tier 2: Significant word overlap (>2 chars, excluding Ayurvedic stopwords)
  Tier 3: Fuzzy Levenshtein >= threshold
"""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs import FUZZY_THRESHOLD
from terminology_bridge import _get_ita_vocab
from gold_standard import split_terms
from evaluation.text_metrics import compute_text_metrics


# ---------------------------------------------------------------------------
# Levenshtein (inlined from irr/irr_ayurveda.py to avoid path issues)
# ---------------------------------------------------------------------------
def levenshtein(s1, s2):
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
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    return 1.0 - levenshtein(s1, s2) / max_len


# Common Ayurvedic stopwords (from irr/irr_ayurveda.py)
AYUR_STOPWORDS = {
    "vata", "pitta", "kapha",
    "dosha", "dushti", "vikara", "roga", "vyadhi",
    "samana", "shamana", "hara",
    "chikitsa", "therapy", "treatment",
    "the", "a", "an", "of", "in", "on", "for", "to", "is", "and", "or",
    "with", "due", "type",
}


# ---------------------------------------------------------------------------
# Term-level matching
# ---------------------------------------------------------------------------
def get_significant_words(term):
    """Extract significant clinical words from a single term."""
    words = set()
    for w in re.split(r"[\s\-/]+", term.lower()):
        w = re.sub(r"[^a-z]", "", w)
        if w and len(w) > 2 and w not in AYUR_STOPWORDS:
            words.add(w)
    return words


def _resolve_ita_ids(terms, ita_vocab):
    """Batch-resolve ITA IDs for a list of terms. Returns {term: ita_id or None}."""
    cache = {}
    for term in terms:
        if term not in cache:
            match = ita_vocab.find_best_match(term)
            cache[term] = match[0] if match else None
    return cache


def match_term_to_term(pred_term, gold_term, pred_ita_id, gold_ita_id):
    """Check if a single predicted term matches a single gold term.

    Uses pre-resolved ITA IDs to avoid redundant vocabulary scans.

    Returns: (matched: bool, level: str)
    """
    # Tier 1: ITA ID match
    if pred_ita_id and gold_ita_id and pred_ita_id == gold_ita_id:
        return True, "ita_match"

    # Tier 2: Significant word overlap
    pred_words = get_significant_words(pred_term)
    gold_words = get_significant_words(gold_term)
    if pred_words and gold_words and (pred_words & gold_words):
        return True, "word_overlap"

    # Tier 3: Fuzzy Levenshtein
    pred_clean = pred_term.lower().strip()
    gold_clean = gold_term.lower().strip()
    if len(pred_clean) >= 3 and len(gold_clean) >= 3:
        sim = normalized_similarity(pred_clean, gold_clean)
        if sim >= FUZZY_THRESHOLD:
            return True, "fuzzy_match"

    return False, "no_match"


def match_field_terms(predicted_terms, gold_terms, ita_vocab=None):
    """Term-level greedy matching for one field (diagnosis or treatment).

    For each predicted term, find the best-matching gold term (preferring
    higher tiers). Matched gold terms are consumed and cannot be reused.

    Returns dict with:
      tp, fp, fn: term-level counts
      tier_breakdown: count per match tier
      predicted_terms, gold_terms: the input term lists
      term_matches: list of per-term match details
      unmatched_gold: gold terms not matched by any prediction
    """
    if ita_vocab is None:
        ita_vocab = _get_ita_vocab()

    gold_list = list(gold_terms) if gold_terms else []
    pred_list = list(predicted_terms) if predicted_terms else []

    if not pred_list and not gold_list:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "tier_breakdown": {},
            "predicted_terms": [], "gold_terms": [],
            "term_matches": [], "unmatched_gold": [],
        }

    # Batch-resolve ITA IDs once (avoids O(n*m*vocab) in inner loop)
    all_terms = set(pred_list) | set(gold_list)
    ita_cache = _resolve_ita_ids(all_terms, ita_vocab)

    remaining_gold = list(gold_list)
    term_matches = []
    tier_breakdown = {}
    level_priority = {"ita_match": 0, "word_overlap": 1, "fuzzy_match": 2}

    for pred in pred_list:
        best_gold = None
        best_level = None
        best_priority = 999

        pred_ita_id = ita_cache.get(pred)

        for gold in remaining_gold:
            gold_ita_id = ita_cache.get(gold)
            matched, level = match_term_to_term(pred, gold, pred_ita_id, gold_ita_id)
            if matched:
                priority = level_priority.get(level, 999)
                if priority < best_priority:
                    best_priority = priority
                    best_gold = gold
                    best_level = level

        if best_gold is not None:
            remaining_gold.remove(best_gold)
            term_matches.append({
                "predicted": pred, "gold": best_gold,
                "matched": True, "level": best_level,
            })
            tier_breakdown[best_level] = tier_breakdown.get(best_level, 0) + 1
        else:
            term_matches.append({
                "predicted": pred, "gold": None,
                "matched": False, "level": "no_match",
            })
            tier_breakdown["no_match"] = tier_breakdown.get("no_match", 0) + 1

    tp = sum(1 for m in term_matches if m["matched"])
    fp = sum(1 for m in term_matches if not m["matched"])
    fn = len(remaining_gold)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "tier_breakdown": tier_breakdown,
        "predicted_terms": pred_list,
        "gold_terms": gold_list,
        "term_matches": term_matches,
        "unmatched_gold": remaining_gold,
    }


def compute_vignette_match(predicted_diagnosis, predicted_treatment,
                           gold_diagnosis_terms, gold_treatment_terms,
                           ita_vocab=None):
    """Evaluate a single vignette's predictions against gold standard.

    Does term-level matching: each predicted term matched against individual
    gold terms via tiered matching (ITA > word overlap > fuzzy).

    Returns dict with diagnosis and treatment match results, each containing
    tp/fp/fn counts and tier breakdown.
    """
    if ita_vocab is None:
        ita_vocab = _get_ita_vocab()

    pred_diag_terms = split_terms(predicted_diagnosis) if predicted_diagnosis else []
    pred_treat_terms = split_terms(predicted_treatment) if predicted_treatment else []

    diag_result = match_field_terms(pred_diag_terms, gold_diagnosis_terms, ita_vocab)
    treat_result = match_field_terms(pred_treat_terms, gold_treatment_terms, ita_vocab)

    # Text-level metrics (ROUGE-L and token F1)
    gold_diag_text = "; ".join(gold_diagnosis_terms) if gold_diagnosis_terms else ""
    gold_treat_text = "; ".join(gold_treatment_terms) if gold_treatment_terms else ""

    diag_result["text_metrics"] = compute_text_metrics(
        predicted_diagnosis or "", gold_diag_text
    )
    treat_result["text_metrics"] = compute_text_metrics(
        predicted_treatment or "", gold_treat_text
    )

    return {"diagnosis": diag_result, "treatment": treat_result}
