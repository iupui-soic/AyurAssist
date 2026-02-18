"""
bridge_only pipeline: NER -> UMLS -> ITA only (no LLM).

Tests whether the terminology bridge alone can produce correct diagnoses.
Diagnosis = all ITA English terms + Sanskrit found via entity matching.
Treatment is empty since there's no LLM to generate it.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipelines.base import PipelineBase, PipelineResult
from terminology_bridge import build_bridge_context


class BridgeOnlyPipeline(PipelineBase):
    name = "bridge_only"

    def process_vignette(self, vignette):
        umls_api_key = os.environ.get("UMLS_API_KEY", "")
        bridge_ctx = build_bridge_context(vignette.narrative, umls_api_key=umls_api_key)

        if not bridge_ctx.ita_matches:
            return PipelineResult(
                config_name=self.name,
                vignette_index=vignette.index,
                narrative=vignette.narrative[:200],
                entities=bridge_ctx.entities,
                predicted_diagnosis="",
                predicted_treatment="",
            )

        # Diagnosis = all matched ITA English terms + Sanskrit
        diag_parts = []
        for m in bridge_ctx.ita_matches:
            diag_parts.append(m.english_term)
            if m.sanskrit_iast:
                diag_parts.append(m.sanskrit_iast)
        predicted_diagnosis = "; ".join(diag_parts)

        ita_match = {
            "ita_matches": [
                {
                    "entity": m.entity,
                    "ita_id": m.ita_id,
                    "english_term": m.english_term,
                    "sanskrit_iast": m.sanskrit_iast,
                    "similarity": m.match_similarity,
                }
                for m in bridge_ctx.ita_matches
            ],
            "unmatched_entities": bridge_ctx.unmatched_entities,
        }

        return PipelineResult(
            config_name=self.name,
            vignette_index=vignette.index,
            narrative=vignette.narrative[:200],
            entities=bridge_ctx.entities,
            ita_match=ita_match,
            predicted_diagnosis=predicted_diagnosis,
            predicted_treatment="",  # No LLM
        )
