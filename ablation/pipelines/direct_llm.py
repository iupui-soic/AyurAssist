"""
direct_llm pipeline: Raw narrative -> Qwen3 (no terminology bridge).

Tests whether the LLM alone can produce correct Ayurvedic
diagnoses and treatments without the terminology bridge.
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groq import Groq

from pipelines.base import PipelineBase, PipelineResult
from configs import (
    GROQ_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    GROQ_RATE_LIMIT_DELAY,
)


def _strip_thinking(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


class DirectLLMPipeline(PipelineBase):
    name = "direct_llm"

    def __init__(self):
        super().__init__()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable required")
        self._client = Groq(api_key=api_key)

    def _call_groq(self, prompt):
        resp = self._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            reasoning_effort="none",
        )
        raw = resp.choices[0].message.content
        return _strip_thinking(raw)

    def process_vignette(self, vignette):
        narrative = vignette.narrative

        # Prompt 1: Diagnosis
        diag_prompt = (
            "You are an expert Ayurvedic physician. Based on the following "
            "patient case, provide your Ayurvedic diagnosis. List the most "
            "likely Ayurvedic condition names (in English and Sanskrit if known). "
            "Be concise — just list the diagnosis terms, separated by semicolons.\n\n"
            f"Patient case:\n{narrative}\n\n"
            "Ayurvedic diagnosis:"
        )

        # Prompt 2: Treatment
        treat_prompt = (
            "You are an expert Ayurvedic physician. Based on the following "
            "patient case, recommend the general line of Ayurvedic treatment. "
            "Include treatment principles, key formulations, and therapies. "
            "Be concise.\n\n"
            f"Patient case:\n{narrative}\n\n"
            "Ayurvedic treatment:"
        )

        responses = []
        predicted_diagnosis = ""
        predicted_treatment = ""

        try:
            diag_resp = self._call_groq(diag_prompt)
            responses.append(diag_resp)
            predicted_diagnosis = diag_resp
        except Exception as e:
            responses.append(f"[ERROR: {e}]")
            predicted_diagnosis = ""

        time.sleep(GROQ_RATE_LIMIT_DELAY)

        try:
            treat_resp = self._call_groq(treat_prompt)
            responses.append(treat_resp)
            predicted_treatment = treat_resp
        except Exception as e:
            responses.append(f"[ERROR: {e}]")
            predicted_treatment = ""

        return PipelineResult(
            config_name=self.name,
            vignette_index=vignette.index,
            narrative=vignette.narrative[:200],
            entities=[],
            predicted_diagnosis=predicted_diagnosis,
            predicted_treatment=predicted_treatment,
            raw_responses=responses,
        )
