"""
Abstract pipeline base with JSONL checkpointing.

Each pipeline saves results per-vignette to a JSONL file. On resume,
already-completed vignette indices are skipped.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict

from configs import RAW_RESPONSES_DIR


@dataclass
class PipelineResult:
    config_name: str
    vignette_index: int
    narrative: str
    entities: list = field(default_factory=list)
    ita_match: dict = field(default_factory=dict)
    predicted_diagnosis: str = ""
    predicted_treatment: str = ""
    raw_responses: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str = ""


class PipelineBase(ABC):
    """Abstract base for all ablation pipeline configurations."""

    name: str = "base"

    def __init__(self):
        os.makedirs(RAW_RESPONSES_DIR, exist_ok=True)
        self._checkpoint_path = os.path.join(
            RAW_RESPONSES_DIR, f"{self.name}.jsonl"
        )

    def _load_completed(self):
        """Load already-completed vignette indices from checkpoint."""
        completed = {}
        if os.path.exists(self._checkpoint_path):
            with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        idx = record.get("vignette_index")
                        if idx is not None:
                            completed[idx] = PipelineResult(**{
                                k: record[k]
                                for k in PipelineResult.__dataclass_fields__
                                if k in record
                            })
                    except (json.JSONDecodeError, TypeError):
                        continue
        return completed

    def _save_result(self, result):
        """Append a single result to the checkpoint file."""
        with open(self._checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    @abstractmethod
    def process_vignette(self, vignette):
        """Process a single vignette. Returns a PipelineResult."""
        ...

    def run_all(self, vignettes, limit=None):
        """Run pipeline on all vignettes with checkpointing.

        Args:
            vignettes: list of Vignette objects
            limit: optional max number of vignettes to process

        Returns: list of PipelineResult
        """
        completed = self._load_completed()
        subset = vignettes[:limit] if limit else vignettes
        results = []

        print(f"\n{'='*60}")
        print(f"  Pipeline: {self.name}")
        print(f"  Vignettes: {len(subset)} (checkpoint has {len(completed)})")
        print(f"{'='*60}\n")

        for i, vignette in enumerate(subset):
            if vignette.index in completed:
                print(f"  [{i+1}/{len(subset)}] Vignette {vignette.index}: cached")
                results.append(completed[vignette.index])
                continue

            print(f"  [{i+1}/{len(subset)}] Vignette {vignette.index}: processing...", end=" ", flush=True)
            start = time.time()

            try:
                result = self.process_vignette(vignette)
                result.elapsed_seconds = time.time() - start
            except Exception as e:
                result = PipelineResult(
                    config_name=self.name,
                    vignette_index=vignette.index,
                    narrative=vignette.narrative[:200],
                    error=str(e),
                    elapsed_seconds=time.time() - start,
                )

            self._save_result(result)
            results.append(result)
            print(f"done ({result.elapsed_seconds:.1f}s)")

        return results
