"""
Ablation study configuration constants.

Env vars needed: GROQ_API_KEY, UMLS_API_KEY
"""

import os

# ── LLM models ────────────────────────────────────────────────────────────
GROQ_MODEL = "qwen/qwen3-32b"

# ── Data paths (relative to this file) ────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
ITA_TERMS_CSV = os.path.join(_HERE, "..", "who-ita", "ita_terms_ascii.csv")
AYURVEDA1_CSV = os.path.join(_HERE, "..", "irr", "ayurveda1.csv")
AYURVEDA2_CSV = os.path.join(_HERE, "..", "irr", "ayurveda2.csv")
RESULTS_DIR = os.path.join(_HERE, "results")
RAW_RESPONSES_DIR = os.path.join(RESULTS_DIR, "raw_responses")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")

# ── NER ───────────────────────────────────────────────────────────────────
NER_MODEL = "en_core_sci_lg"

# ── UMLS API ──────────────────────────────────────────────────────────────
UMLS_SEARCH_URL = "https://uts-ws.nlm.nih.gov/rest/search/current"
UMLS_REQUEST_TIMEOUT = 10  # seconds

# ── Matching thresholds ───────────────────────────────────────────────────
FUZZY_THRESHOLD = 0.80

# ── API call settings ────────────────────────────────────────────────────
GROQ_RATE_LIMIT_DELAY = 2.5  # seconds between Groq calls

# ── LLM generation params ────────────────────────────────────────────────
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024
