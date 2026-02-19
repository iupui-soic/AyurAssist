import modal
import os
import re
import csv
import io
import asyncio
from dataclasses import dataclass
from difflib import SequenceMatcher
from contextlib import asynccontextmanager

from config import (
    MODAL_APP_NAME, MODAL_SECRET_UMLS, MODAL_SECRET_GROQ,
    CPU_TIMEOUT, CPU_SCALEDOWN_WINDOW,
    GROQ_MODEL, GROQ_RATE_LIMIT_DELAY,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
    NER_MODEL_NAME,
    ITA_CSV_SOURCE_PATH, ITA_CSV_CONTAINER_PATH, FUZZY_THRESHOLD,
    UMLS_SEARCH_URL, UMLS_ATOMS_URL_TEMPLATE, UMLS_REQUEST_TIMEOUT,
    PYTHON_VERSION,
)

app = modal.App(MODAL_APP_NAME)

# ---------------------------------------------------------------------------
# Image (CPU only — no GPU tier)
# ---------------------------------------------------------------------------
cpu_image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(
        "scispacy==0.5.5",
        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz",
        "fastapi[standard]==0.109.0",
        "requests==2.31.0",
        "groq==0.25.0",
    )
    .add_local_file(ITA_CSV_SOURCE_PATH, ITA_CSV_CONTAINER_PATH)
    .add_local_file("config.py", "/root/config.py")
)


# ===================================================================
# ITA Vocabulary (ported from ablation/terminology_bridge.py)
# ===================================================================

class ITAVocabulary:
    """WHO-ITA terminology loaded from ita_terms_ascii.csv."""

    def __init__(self, filepath=ITA_CSV_CONTAINER_PATH):
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
        """Fuzzy-match a term against all English terms in the ITA vocabulary."""
        term_lower = term.lower().strip()

        if term_lower in self.english_index:
            ita_id = self.english_index[term_lower]
            english, sanskrit_list = self.terms[ita_id]
            sanskrit = "; ".join(sanskrit_list) if sanskrit_list else ""
            return ita_id, english, sanskrit, 1.0

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


# ===================================================================
# NER (ported from ablation/terminology_bridge.py)
# ===================================================================

_NER_STOPWORDS = {
    "year", "years", "old", "old man", "old woman", "month", "months",
    "day", "days", "week", "weeks", "report", "case", "history",
    "patient", "patient's", "patients", "time", "high", "low",
    "normal", "result", "results", "diagnosis", "prognosis",
    "test", "tests", "scan", "origin", "type", "side", "since",
    "male", "female", "man", "woman", "boy", "girl",
}


def _extract_entities(nlp, text):
    """Run scispacy NER on text. Returns list of entity strings."""
    doc = nlp(text)
    entities = []
    seen = set()
    for ent in doc.ents:
        word = ent.text.strip()
        key = word.lower()
        if key in seen:
            continue
        if len(word) < 4 or key in _NER_STOPWORDS:
            continue
        seen.add(key)
        entities.append(word)
    return entities


def _extract_entity_dicts(nlp, text):
    """Run scispacy NER returning dicts for the API response."""
    doc = nlp(text)
    entities = []
    seen = set()
    for ent in doc.ents:
        word = ent.text.strip()
        key = word.lower()
        if key in seen:
            continue
        if len(word) < 4 or key in _NER_STOPWORDS:
            continue
        seen.add(key)
        entities.append({
            "word": word,
            "score": 1.0,
            "entity_group": ent.label_,
        })
    return entities


# ===================================================================
# UMLS lookup (enhanced from aravind's version — returns 4-tuple)
# ===================================================================

def _lookup_umls(api_key, keyword):
    """Search UMLS for a keyword.
    Returns: (CUI, SNOMED_CODE, PREFERRED_NAME, ICD10_CODE)
    """
    import requests

    res = ("N/A", "N/A", keyword, "N/A")
    if not api_key or len(keyword) < 2:
        return res

    try:
        params = {"string": keyword, "apiKey": api_key, "returnIdType": "concept"}
        r = requests.get(UMLS_SEARCH_URL, params=params, timeout=UMLS_REQUEST_TIMEOUT)
        if r.status_code != 200:
            return res

        results = r.json().get("result", {}).get("results", [])
        if not results:
            return res

        top = results[0]
        cui = top.get("ui")
        name = top.get("name")

        snomed = "N/A"
        icd10 = "N/A"

        atoms_url = UMLS_ATOMS_URL_TEMPLATE.format(cui=cui)
        r2 = requests.get(atoms_url, params={
            "apiKey": api_key,
            "sabs": "SNOMEDCT_US,ICD10CM",
            "ttys": "PT",
            "pageSize": 20,
        }, timeout=UMLS_REQUEST_TIMEOUT)

        if r2.status_code == 200:
            for atom in r2.json().get("result", []):
                src = atom.get("rootSource")
                if src == "SNOMEDCT_US" and snomed == "N/A":
                    code = atom.get("code", "").split("/")[-1]
                    snomed = code
                if src == "ICD10CM" and icd10 == "N/A":
                    code = atom.get("code", "").split("/")[-1]
                    icd10 = code

        return (cui, snomed, name, icd10)

    except Exception as e:
        print(f"UMLS Error: {e}")
        return res


# ===================================================================
# Bridge context (ported from ablation/terminology_bridge.py)
# ===================================================================

@dataclass
class EntityITAMatch:
    entity: str
    umls_term: str
    ita_id: str
    english_term: str
    sanskrit_iast: str
    match_similarity: float


@dataclass
class BridgeContext:
    entities: list
    ita_matches: list
    unmatched_entities: list


def _build_bridge_context(nlp, ita_vocab, narrative, umls_api_key=None, threshold=FUZZY_THRESHOLD):
    """Build terminology context: extract entities and match each to ITA."""
    import requests

    entities = _extract_entities(nlp, narrative)
    if not entities:
        return BridgeContext(entities=[], ita_matches=[], unmatched_entities=[])

    seen_ita_ids = set()
    ita_matches = []
    unmatched = []

    for entity in entities:
        matched = False

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

        # Try UMLS preferred term -> ITA
        if umls_api_key:
            try:
                params = {
                    "string": entity,
                    "apiKey": umls_api_key,
                    "returnIdType": "concept",
                }
                r = requests.get(UMLS_SEARCH_URL, params=params, timeout=UMLS_REQUEST_TIMEOUT)
                if r.status_code == 200:
                    results = r.json().get("result", {}).get("results", [])
                    if results:
                        umls_pref = results[0].get("name", "N/A")
                        if umls_pref != "N/A":
                            ita_match2 = ita_vocab.find_best_english_match(umls_pref, threshold=threshold)
                            if ita_match2:
                                ita_id2, eng2, skt2, sim2 = ita_match2
                                if ita_id2 not in seen_ita_ids:
                                    seen_ita_ids.add(ita_id2)
                                    ita_matches.append(EntityITAMatch(
                                        entity=entity, umls_term=umls_pref,
                                        ita_id=ita_id2, english_term=eng2,
                                        sanskrit_iast=skt2, match_similarity=sim2,
                                    ))
                                matched = True
            except Exception as e:
                print(f"  UMLS search error for '{entity}': {e}")

        if not matched:
            unmatched.append(entity)

    return BridgeContext(
        entities=entities,
        ita_matches=ita_matches,
        unmatched_entities=unmatched,
    )


# ===================================================================
# Few-shot examples (from ablation/pipelines/full_pipeline.py)
# ===================================================================

FEW_SHOT_EXAMPLES = [
    {
        "narrative": (
            "A 78 year old woman with a cutaneous ulcer as well as a "
            "duodenal ulcer and renal failure."
        ),
        "diagnosis": "Parinama Shoola; Vrana; Mutrakshaya; Amlapitta",
        "treatment": (
            "Vata-Pitta shamana; Vrana chikitsa; Yashtimadhu; "
            "Shatavari; Avipattikar churna for ulcer; Jatyadi taila "
            "for wound; Punarnava for renal support"
        ),
    },
    {
        "narrative": (
            "A 60 year old woman with fatigue and reduced appetite for "
            "1 year, feels exhausted even after doing nothing and "
            "barely eats half her normal meals."
        ),
        "diagnosis": "Agnimandya; Dhatu Kshaya",
        "treatment": (
            "Deepana-Pachana (Trikatu, Chitrakadi Vati); Ashwagandha; "
            "Shatavari; Amalaki Rasayana; Basti Panchakarma; "
            "nutritious Pathya Ahara"
        ),
    },
    {
        "narrative": (
            "A 60 year old female with dizziness and vomiting, room "
            "feels like it's spinning, worse when she turns her head."
        ),
        "diagnosis": "Bhrama; Vataja Shiroroga",
        "treatment": (
            "Vata-Pitta shamana; Brahmi; Shankhapushpi; Ashwagandha; "
            "Shirobasti with Ksheerabala taila; Nasya with Anu taila; "
            "light easily digestible diet"
        ),
    },
]


# ===================================================================
# Prompt builders (ported from ablation/pipelines/full_pipeline.py)
# ===================================================================

def _build_few_shot_block():
    lines = ["Here are examples of Ayurvedic clinical assessments:\n"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Case: {ex['narrative']}")
        lines.append(f"  Diagnosis: {ex['diagnosis']}")
        lines.append(f"  Treatment: {ex['treatment']}")
        lines.append("")
    return "\n".join(lines)


def _build_ita_dictionary_block(bridge_context):
    if not bridge_context.ita_matches:
        return ""
    lines = []
    for m in bridge_context.ita_matches:
        sanskrit_part = f" = {m.sanskrit_iast}" if m.sanskrit_iast else ""
        lines.append(f"- {m.english_term}{sanskrit_part} ({m.ita_id})")
    return (
        "Relevant Ayurvedic terminology from WHO-ITA vocabulary "
        "(use these terms where applicable):\n"
        + "\n".join(lines)
    )


def _build_entities_block(bridge_context):
    if not bridge_context.entities:
        return ""
    return (
        "Medical entities identified in this case:\n"
        + "\n".join(f"- {e}" for e in bridge_context.entities)
    )


def _strip_thinking(text):
    """Strip <think>...</think> blocks from Qwen3 responses."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


# ===================================================================
# Input cleaning (from aravind's main.py)
# ===================================================================

def _clean_input_smart(text):
    """Remove conversational filler so we are left with the core medical concept."""
    t = text.lower()
    fillers = [
        "i am", "i'm", "i have", "i've", "i feel", "i think", "why do", "what is",
        "having", "feeling", "suffering", "diagnosed", "with", "from",
        "severe", "mild", "acute", "chronic", "very", "bad", "really",
        "my", "the", "a", "an", "and", "or", "in", "on", "at", "to",
    ]
    for f in fillers:
        t = re.sub(r'\b' + re.escape(f) + r'\b', '', t)
    t = re.sub(r'[^\w\s]', '', t)
    return t.strip()


# ===================================================================
# 13-question protocol (adapted from aravind's main.py)
# ===================================================================

def _build_questions(condition, original_text, ita_dict_block=""):
    """Standard Ayurvedic Clinical Assessment Protocol — 13 questions."""
    ita_context = f"\n\n{ita_dict_block}" if ita_dict_block else ""
    return [
        # Q0: overview_dosha_causes
        (
            f"You are an expert Ayurvedic physician.{ita_context}\n\n"
            f"The patient reports: '{original_text}'. Focusing on '{condition}', "
            f"explain the Ayurvedic perspective (Nidana/Pathogenesis). "
            f"Which Doshas are aggravated? List the main causes."
        ),
        # Q1: symptoms
        f"You are an expert Ayurvedic physician.{ita_context}\n\nWhat are the main symptoms (Rupa) of '{condition}' in Ayurveda?",
        # Q2: single_drug_remedies
        f"You are an expert Ayurvedic physician.{ita_context}\n\nList 3 specific single-herb remedies (Eka Mulika / Ottamooli) for '{condition}'. For each give: name, Sanskrit name, part used, preparation, dosage, and duration.",
        # Q3: classical_formulations
        f"You are an expert Ayurvedic physician.{ita_context}\n\nSuggest 2-3 classical Ayurvedic formulations (Yogas) for '{condition}'. Give name, form, dosage, and reference text.",
        # Q4: panchakarma
        f"You are an expert Ayurvedic physician.{ita_context}\n\nWhat Panchakarma therapies are indicated for '{condition}'?",
        # Q5: diet_lifestyle
        f"You are an expert Ayurvedic physician.{ita_context}\n\nProvide dietary advice (Pathya/Apathya) and lifestyle recommendations for '{condition}'.",
        # Q6: yoga
        f"You are an expert Ayurvedic physician.{ita_context}\n\nWhat Yoga Asanas or Pranayama are beneficial for '{condition}'?",
        # Q7: prognosis
        f"You are an expert Ayurvedic physician.{ita_context}\n\nWhat is the prognosis (Sadhya/Asadhya) for '{condition}'?",
        # Q8: modern_correlation_warnings
        (
            f"You are a medical expert with knowledge of both Ayurveda and modern medicine.{ita_context}\n\n"
            f"For '{condition}': "
            f"1) What is the modern medical correlation? "
            f"2) What is the general line of treatment in modern medicine? "
            f"3) What are the danger signs or red flags requiring immediate medical attention?"
        ),
        # Q9: differential_diagnosis
        f"You are an expert Ayurvedic physician.{ita_context}\n\nWhat is the Differential Diagnosis (Vyavachedaka Nidana) in Ayurveda for '{condition}'?",
        # Q10: investigations_labs
        f"You are a medical expert.{ita_context}\n\nWhat modern lab investigations are recommended for '{condition}'?",
        # Q11: prevention_recurrence
        f"You are an expert Ayurvedic physician.{ita_context}\n\nSuggest Rasayana (Rejuvenation) therapy to prevent recurrence of '{condition}'.",
        # Q12: psychotherapy_satvavajaya
        f"You are an expert Ayurvedic physician.{ita_context}\n\nIs there a psychosomatic component (Manasika Dosha) to '{condition}'? Suggest Satvavajaya (counseling) measures.",
    ]


RESPONSE_KEYS = [
    "overview_dosha_causes",
    "symptoms",
    "single_drug_remedies",
    "classical_formulations",
    "panchakarma",
    "diet_lifestyle",
    "yoga",
    "prognosis",
    "modern_correlation_warnings",
    "differential_diagnosis",
    "investigations_labs",
    "prevention_recurrence",
    "psychotherapy_satvavajaya",
]


# ===================================================================
# Groq API call wrapper
# ===================================================================

def _call_groq_sync(client, prompt):
    """Call Qwen3 via Groq API (reasoning disabled for speed)."""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        reasoning_effort="none",
    )
    raw = resp.choices[0].message.content
    return _strip_thinking(raw)


# ===================================================================
# ASGI app
# ===================================================================

@asynccontextmanager
async def lifespan(web_app):
    import spacy

    web_app.state.ner = await asyncio.to_thread(spacy.load, NER_MODEL_NAME)
    web_app.state.ita_vocab = await asyncio.to_thread(ITAVocabulary)
    web_app.state.umls_api_key = os.environ.get("UMLS_API_KEY", "")
    web_app.state.groq_api_key = os.environ.get("GROQ_API_KEY", "")
    print("CPU engine ready (NER + ITA + Groq).")
    yield


@app.function(
    image=cpu_image,
    timeout=CPU_TIMEOUT,
    scaledown_window=CPU_SCALEDOWN_WINDOW,
    secrets=[
        modal.Secret.from_name(MODAL_SECRET_UMLS),
        modal.Secret.from_name(MODAL_SECRET_GROQ),
    ],
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    web = FastAPI(lifespan=lifespan)
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web.get("/warmup")
    async def warmup():
        return {"status": "ready"}

    @web.post("/")
    async def analyze(request: Request):
        from groq import Groq

        try:
            body = await request.json()
            user_input = body.get("text", "").strip()
            if not user_input:
                raise HTTPException(status_code=400, detail="Missing 'text' field")

            st = request.app.state

            # --- 1. Clean input and run NER ---
            cleaned_phrase = _clean_input_smart(user_input)

            entity_dicts = await asyncio.to_thread(
                _extract_entity_dicts, st.ner, user_input
            )

            # --- 2. Build ITA bridge context (NER -> UMLS -> ITA) ---
            bridge_ctx = await asyncio.to_thread(
                _build_bridge_context,
                st.ner, st.ita_vocab, user_input,
                umls_api_key=st.umls_api_key,
                threshold=FUZZY_THRESHOLD,
            )

            # --- 3. UMLS lookup for the cleaned phrase ---
            cui, snomed, preferred_name, icd10 = await asyncio.to_thread(
                _lookup_umls, st.umls_api_key, cleaned_phrase
            )

            # If phrase lookup failed, try NER entities
            if snomed == "N/A" and bridge_ctx.entities:
                for entity in sorted(bridge_ctx.entities, key=len, reverse=True):
                    e_cui, e_snomed, e_name, e_icd = await asyncio.to_thread(
                        _lookup_umls, st.umls_api_key, entity
                    )
                    if e_snomed != "N/A":
                        cui, snomed, preferred_name, icd10 = e_cui, e_snomed, e_name, e_icd
                        break

            condition_name = preferred_name if preferred_name != "N/A" else (cleaned_phrase or user_input)

            # --- 4. Build ITA match info for response ---
            csv_match = None
            if bridge_ctx.ita_matches:
                best = bridge_ctx.ita_matches[0]
                csv_match = {
                    "ita_id": best.ita_id,
                    "ayurveda_term": best.english_term,
                    "sanskrit_iast": best.sanskrit_iast,
                    "sanskrit": best.sanskrit_iast,
                    "description": "",
                }

            # --- 5. Three-pass diagnosis (sequential with rate limiting) ---
            groq_client = Groq(api_key=st.groq_api_key)

            entities_block = _build_entities_block(bridge_ctx)
            ita_dict_block = _build_ita_dictionary_block(bridge_ctx)
            few_shot_block = _build_few_shot_block()

            # PASS 1: Modern diagnosis
            modern_prompt = (
                "You are a medical expert. Based on the following patient "
                "case, list the most likely modern medical diagnoses. "
                "Be concise — just list condition names separated by "
                "semicolons.\n\n"
                f"{entities_block}\n\n"
                f"Patient case:\n{user_input}\n\n"
                "Modern medical diagnoses:"
            )
            modern_diagnosis = ""
            try:
                modern_diagnosis = await asyncio.to_thread(
                    _call_groq_sync, groq_client, modern_prompt
                )
            except Exception as e:
                print(f"Modern diagnosis error: {e}")

            await asyncio.sleep(GROQ_RATE_LIMIT_DELAY)

            # PASS 2: Ayurvedic diagnosis
            ayur_diag_prompt = (
                "You are an expert Ayurvedic physician.\n\n"
                f"{few_shot_block}\n"
                "Now assess this new case.\n\n"
                f"Modern medical assessment: {modern_diagnosis}\n\n"
                f"{entities_block}\n\n"
                f"{ita_dict_block}\n\n"
                f"Patient case:\n{user_input}\n\n"
                "Following the same format as the examples above, provide "
                "your Ayurvedic diagnosis. Use standard Ayurvedic condition "
                "names in Sanskrit (e.g., Vidradhi, Vata Vyadhi, Pandu Roga, "
                "Jwara, Prameha). Use the WHO-ITA terms listed above where "
                "applicable. List diagnosis terms separated by semicolons.\n\n"
                "Diagnosis:"
            )
            ayur_diagnosis = ""
            try:
                ayur_diagnosis = await asyncio.to_thread(
                    _call_groq_sync, groq_client, ayur_diag_prompt
                )
            except Exception as e:
                print(f"Ayurvedic diagnosis error: {e}")

            await asyncio.sleep(GROQ_RATE_LIMIT_DELAY)

            # PASS 3: Ayurvedic treatment
            ayur_treat_prompt = (
                "You are an expert Ayurvedic physician.\n\n"
                f"{few_shot_block}\n"
                "Now recommend treatment for this case.\n\n"
                f"Ayurvedic diagnosis: {ayur_diagnosis}\n\n"
                f"{ita_dict_block}\n\n"
                f"Patient case:\n{user_input}\n\n"
                "Following the same format as the examples above, recommend "
                "the general line of Ayurvedic treatment. Include treatment "
                "principles (Shamana/Shodhana), key formulations, "
                "Panchakarma if relevant, and dietary advice. "
                "Be concise.\n\n"
                "Treatment:"
            )
            ayur_treatment = ""
            try:
                ayur_treatment = await asyncio.to_thread(
                    _call_groq_sync, groq_client, ayur_treat_prompt
                )
            except Exception as e:
                print(f"Ayurvedic treatment error: {e}")

            # Use the Ayurvedic diagnosis as condition name if available
            ayur_condition = ayur_diagnosis.split(";")[0].strip() if ayur_diagnosis else condition_name
            sanskrit_name = ""
            if csv_match:
                sanskrit_name = csv_match.get("sanskrit_iast", "")

            # --- 6. 13 detail questions (parallel with semaphore) ---
            questions = _build_questions(
                ayur_condition or condition_name,
                user_input,
                ita_dict_block=ita_dict_block,
            )

            sem = asyncio.Semaphore(5)

            async def _call_with_limit(prompt):
                async with sem:
                    try:
                        result = await asyncio.to_thread(
                            _call_groq_sync, groq_client, prompt
                        )
                        return result
                    except Exception as e:
                        print(f"Groq question error: {e}")
                        return ""

            responses = await asyncio.gather(*[
                _call_with_limit(q) for q in questions
            ])

            # --- 7. Assemble response ---
            ayurparam_responses = {}
            for key, resp in zip(RESPONSE_KEYS, responses):
                ayurparam_responses[key] = resp

            treatment_info = {
                "condition_name": ayur_condition or condition_name,
                "sanskrit_name": sanskrit_name,
                "disclaimer": "This information is for educational purposes only. Consult a qualified Ayurvedic practitioner.",
                "ayurparam_responses": ayurparam_responses,
            }

            return {
                "input_text": user_input,
                "clinical_entities": entity_dicts if entity_dicts else [
                    {"word": cleaned_phrase or user_input, "score": 1.0, "entity_group": "ENTITY"}
                ],
                "umls_cui": cui,
                "snomed_code": snomed,
                "snomed_name": preferred_name if preferred_name != "N/A" else condition_name,
                "icd10_code": icd10,
                "csv_match": csv_match,
                "results": [{
                    "ayurveda_term": ayur_condition or condition_name,
                    "snomed_code": snomed,
                    "icd10_code": icd10,
                    "treatment_info": treatment_info,
                }],
            }

        except HTTPException:
            raise
        except Exception as e:
            print(f"Request error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return web
