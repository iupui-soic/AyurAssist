# ──────────────────────────────────────────────────────────────
# AyurAssist configuration
#
# All tuneable constants live here. Secrets (API keys, tokens)
# are stored in Modal's secret manager -- only their *names*
# are referenced below.
# ──────────────────────────────────────────────────────────────

# ── Modal ─────────────────────────────────────────────────────
MODAL_APP_NAME = "ayurparam-service"
MODAL_VOLUME_NAME = "ayurparam-models-final"
MODAL_SECRET_HUGGINGFACE = "huggingface-secret"
MODAL_SECRET_UMLS = "my-umls-secret"

# ── GPU tier (LLMEngine) ─────────────────────────────────────
GPU_TYPE = "T4"
GPU_TIMEOUT = 600              # seconds
GPU_MIN_CONTAINERS = 0
GPU_SCALEDOWN_WINDOW = 60      # seconds idle before shutdown

# ── CPU tier (ASGI + NER orchestrator) ───────────────────────
CPU_TIMEOUT = 1200             # seconds
CPU_SCALEDOWN_WINDOW = 300     # seconds idle before shutdown

# ── Models ───────────────────────────────────────────────────
LLM_MODEL_ID = "bharatgenai/AyurParam"
LLM_MAX_MODEL_LEN = 2048
LLM_MAX_TOKENS = 600
LLM_TEMPERATURE = 0.6
LLM_TOP_P = 0.95
LLM_TOP_K = 50
LLM_REPETITION_PENALTY = 1.1
LLM_DTYPE = "half"

NER_MODEL_ID = "d4data/biomedical-ner-all"
NER_AGGREGATION_STRATEGY = "simple"

# ── Data ─────────────────────────────────────────────────────
CSV_SOURCE_PATH = "ayurveda_snomed_mapping.csv"
CSV_CONTAINER_PATH = "/app/ayurveda_snomed_mapping.csv"
MODEL_CACHE_DIR = "/cache/models"
VOLUME_MOUNT_PATH = "/cache"

# ── UMLS API ─────────────────────────────────────────────────
UMLS_SEARCH_URL = "https://uts-ws.nlm.nih.gov/rest/search/current"
UMLS_ATOMS_URL_TEMPLATE = "https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}/atoms"
UMLS_REQUEST_TIMEOUT = 10      # seconds

# ── Fuzzy matching ───────────────────────────────────────────
FUZZY_MATCH_THRESHOLD = 0.6

# ── Python version for container images ──────────────────────
PYTHON_VERSION = "3.11"
