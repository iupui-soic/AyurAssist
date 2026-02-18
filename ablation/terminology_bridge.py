"""
Terminology bridge: NER + UMLS + ITA CSV lookup.

This replaces the SNOMED-based lookup from the main pipeline with
ITA-based English term fuzzy matching against the full 3,550-term
WHO-ITA vocabulary.
"""

import csv
import io
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import requests

from configs import (
    NER_MODEL,
    ITA_TERMS_CSV,
    UMLS_SEARCH_URL,
    UMLS_REQUEST_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Levenshtein (inlined from irr/irr_ayurveda.py)
# ---------------------------------------------------------------------------
def _levenshtein(s1, s2):
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _normalized_similarity(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    return 1.0 - _levenshtein(s1, s2) / max_len


class ITAVocabulary:
    """WHO-ITA terminology loaded from ita_terms_ascii.csv."""

    def __init__(self, filepath=ITA_TERMS_CSV):
        self.terms = {}           # ita_id -> (english, sanskrit_list)
        self.sanskrit_index = {}  # lowercase collapsed key -> ita_id
        self.english_index = {}   # lowercase english term -> ita_id
        self._load(filepath)

    def _load(self, filepath):
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(io.StringIO(f.read()))
            rows = list(reader)

        for row in rows[1:]:
            if len(row) < 3:
                continue
            ita_id = row[0].strip()
            english = row[1].strip()
            sanskrit_raw = row[2].strip()

            sanskrit_terms = []
            for part in re.split(r"[;,]", sanskrit_raw):
                term = re.sub(r"^\d+\.\s*", "", part).strip()
                term = term.rstrip("/").strip()
                if term:
                    sanskrit_terms.append(term)

            self.terms[ita_id] = (english, sanskrit_terms)

            for st in sanskrit_terms:
                key = st.lower().replace("-", "").replace(" ", "")
                self.sanskrit_index[key] = ita_id
            eng_key = english.lower().strip()
            self.english_index[eng_key] = ita_id

    def find_best_english_match(self, term, threshold=0.60):
        """Fuzzy-match a term against all English terms in the ITA vocabulary.

        Returns: (ita_id, english_term, sanskrit_iast, similarity) or None
        """
        term_lower = term.lower().strip()

        # Exact match first
        if term_lower in self.english_index:
            ita_id = self.english_index[term_lower]
            english, sanskrit_list = self.terms[ita_id]
            sanskrit = "; ".join(sanskrit_list) if sanskrit_list else ""
            return ita_id, english, sanskrit, 1.0

        # Fuzzy match (with length pruning for speed)
        best_id = None
        best_sim = 0.0
        tl_len = len(term_lower)
        max_dist_ratio = 1.0 - threshold
        for eng_key, ita_id in self.english_index.items():
            ek_len = len(eng_key)
            max_len = max(tl_len, ek_len)
            if max_len > 0 and abs(tl_len - ek_len) > max_dist_ratio * max_len:
                continue
            sim = SequenceMatcher(None, term_lower, eng_key).ratio()
            if sim > best_sim:
                best_sim = sim
                best_id = ita_id

        if best_id and best_sim >= threshold:
            english, sanskrit_list = self.terms[best_id]
            sanskrit = "; ".join(sanskrit_list) if sanskrit_list else ""
            return best_id, english, sanskrit, best_sim

        return None

    def find_best_match(self, term, threshold=0.75):
        """Find best ITA match (Sanskrit or English). Returns (ita_id, matched, sim) or None."""
        term_clean = term.lower().replace("-", "").replace(" ", "")

        # Exact Sanskrit
        if term_clean in self.sanskrit_index:
            ita_id = self.sanskrit_index[term_clean]
            return ita_id, term, 1.0

        # Fuzzy Sanskrit (with length pruning for speed)
        best = None
        best_sim = 0.0
        tc_len = len(term_clean)
        max_dist_ratio = 1.0 - threshold  # e.g. 0.25 for threshold=0.75
        for key, ita_id in self.sanskrit_index.items():
            # Length pruning: if lengths differ too much, skip
            k_len = len(key)
            max_len = max(tc_len, k_len)
            if max_len > 0 and abs(tc_len - k_len) > max_dist_ratio * max_len:
                continue
            sim = _normalized_similarity(term_clean, key)
            if sim > best_sim:
                best_sim = sim
                best = (ita_id, key, sim)

        if best and best_sim >= threshold:
            return best

        # English exact
        term_lower = term.lower().strip()
        if term_lower in self.english_index:
            ita_id = self.english_index[term_lower]
            return ita_id, term, 1.0

        return None

    def get_ita_ids_for_terms(self, terms, threshold=0.75):
        results = {}
        for term in terms:
            match = self.find_best_match(term, threshold)
            if match:
                ita_id, matched, sim = match
                results[term] = (ita_id, sim)
        return results


# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load(NER_MODEL)
    return _nlp


# Common non-clinical words that scispacy often tags as entities
_NER_STOPWORDS = {
    "year", "years", "old", "old man", "old woman", "month", "months",
    "day", "days", "week", "weeks", "report", "case", "history",
    "patient", "patient's", "patients", "time", "high", "low",
    "normal", "result", "results", "diagnosis", "prognosis",
    "test", "tests", "scan", "origin", "type", "side", "since",
    "male", "female", "man", "woman", "boy", "girl",
}


def extract_entities(text):
    """Run scispacy NER on text. Returns list of entity strings.

    Filters out very short entities and common non-clinical stopwords
    to reduce noise in downstream ITA matching.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    entities = []
    seen = set()
    for ent in doc.ents:
        word = ent.text.strip()
        key = word.lower()
        if key in seen:
            continue
        # Skip very short entities and stopwords
        if len(word) < 4 or key in _NER_STOPWORDS:
            continue
        seen.add(key)
        entities.append(word)
    return entities


# ---------------------------------------------------------------------------
# UMLS lookup (adapted from main.py)
# ---------------------------------------------------------------------------
def lookup_umls(keyword, api_key=None):
    """Two-step UMLS: keyword -> CUI, then CUI -> preferred term.

    Returns: (umls_cui, preferred_term) or ("N/A", "N/A")
    """
    if api_key is None:
        api_key = os.environ.get("UMLS_API_KEY", "")
    if not api_key:
        return "N/A", "N/A"

    umls_cui = "N/A"
    preferred_term = "N/A"

    # Step 1: keyword -> CUI
    try:
        params = {
            "string": keyword,
            "apiKey": api_key,
            "returnIdType": "concept",
        }
        r = requests.get(
            UMLS_SEARCH_URL,
            params=params,
            timeout=UMLS_REQUEST_TIMEOUT,
        )
        if r.status_code == 200:
            results = r.json().get("result", {}).get("results", [])
            if results:
                umls_cui = results[0].get("ui", "N/A")
                preferred_term = results[0].get("name", "N/A")
    except Exception as e:
        print(f"  UMLS search error for '{keyword}': {e}")
        return umls_cui, preferred_term

    return umls_cui, preferred_term


# ---------------------------------------------------------------------------
# Bridge context (used by pipelines)
# ---------------------------------------------------------------------------
@dataclass
class EntityITAMatch:
    """A single NER entity matched to an ITA term."""
    entity: str             # original NER entity text
    umls_term: str          # UMLS preferred term (or "N/A")
    ita_id: str             # ITA ID
    english_term: str       # ITA English term
    sanskrit_iast: str      # ITA Sanskrit IAST
    match_similarity: float # how well entity matched ITA


@dataclass
class BridgeContext:
    """All NER entities with their ITA matches — used as LLM context."""
    entities: list                      # all raw NER entities
    ita_matches: list                   # list of EntityITAMatch (deduplicated by ITA ID)
    unmatched_entities: list            # entities with no ITA match


def build_bridge_context(narrative, umls_api_key=None, threshold=0.85):
    """Build terminology context: extract ALL entities and match each to ITA.

    Returns ALL entity→ITA matches to serve as a vocabulary dictionary
    for the LLM. Returns a BridgeContext.
    """
    entities = extract_entities(narrative)
    if not entities:
        return BridgeContext(entities=[], ita_matches=[], unmatched_entities=[])

    ita_vocab = _get_ita_vocab()
    seen_ita_ids = set()
    ita_matches = []
    unmatched = []

    for entity in entities:
        matched = False

        # Try direct ITA match on entity text
        ita_match = ita_vocab.find_best_english_match(entity, threshold=threshold)
        if ita_match:
            ita_id, eng, skt, sim = ita_match
            if ita_id not in seen_ita_ids:
                seen_ita_ids.add(ita_id)
                ita_matches.append(EntityITAMatch(
                    entity=entity, umls_term="N/A",
                    ita_id=ita_id, english_term=eng,
                    sanskrit_iast=skt, match_similarity=sim,
                ))
            matched = True

        # Try UMLS → ITA (may find a different/better ITA term)
        umls_cui, umls_pref = lookup_umls(entity, api_key=umls_api_key)
        if umls_pref != "N/A":
            ita_match = ita_vocab.find_best_english_match(umls_pref, threshold=threshold)
            if ita_match:
                ita_id, eng, skt, sim = ita_match
                if ita_id not in seen_ita_ids:
                    seen_ita_ids.add(ita_id)
                    ita_matches.append(EntityITAMatch(
                        entity=entity, umls_term=umls_pref,
                        ita_id=ita_id, english_term=eng,
                        sanskrit_iast=skt, match_similarity=sim,
                    ))
                matched = True

        if not matched:
            unmatched.append(entity)

    return BridgeContext(
        entities=entities,
        ita_matches=ita_matches,
        unmatched_entities=unmatched,
    )


_ita_vocab = None


def _get_ita_vocab():
    global _ita_vocab
    if _ita_vocab is None:
        _ita_vocab = ITAVocabulary()
    return _ita_vocab
