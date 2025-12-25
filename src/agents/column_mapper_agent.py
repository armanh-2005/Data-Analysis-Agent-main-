# src/agents/column_mapper_agent.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from src.db.repository import SQLiteRepository


class ColumnMapperAgent(BaseAgent):
    name = "column_mapper_agent"
    prompt_file = "column_mapper.md"
    output_schema = {
        "type": "object",
        "properties": {
            "mapped_columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column_name": {"type": "string"},
                        "question_id": {"type": "string"},
                        "question_text": {"type": "string"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                        "confidence_label": {"type": "string"},
                        "inferred_role": {"type": "string"},
                        "privacy_level": {"type": "string"},
                        "value_constraints": {"type": "object"},
                    },
                    "required": ["column_name", "confidence"],
                    "additionalProperties": True,
                },
            }
        },
        "required": ["mapped_columns"],
        "additionalProperties": True,
    }

    default_prompt = """
You map the user question to questionnaire columns.
You receive the schema registry (list of questions with column_name/question_text/type/allowed_values/privacy_level).
Return ONLY JSON:
{
  "mapped_columns": [
    {
      "column_name": "...",
      "reason": "...",
      "confidence": 0.0,
      "confidence_label": "low|medium|high",
      "inferred_role": "target|group_by|filter|time|other"
    }
  ]
}

User question:
{{user_question}}

Schema:
{{schema}}
""".strip()

    def __init__(self, model: str = "glm-4.6", db_path: str = "data/app.db", prompts_dir: Optional[str] = None):
        # Resolve prompts_dir robustly (relative to this file) if not provided.
        if prompts_dir is None:
            prompts_dir = str(Path(__file__).resolve().parents[1] / "prompts")
        
        # ارسال پارامتر model به کلاس پایه
        super().__init__(model=model, prompts_dir=prompts_dir)
        self.repo = SQLiteRepository(db_path)
    def _build_variables(self, state: Any) -> Dict[str, Any]:
        # The mapper needs a questionnaire_id to load schema
        notes = getattr(state, "notes", {}) or {}
        questionnaire_id = notes.get("questionnaire_id")

        schema: List[Dict[str, Any]] = []
        if questionnaire_id:
            schema = self.repo.load_schema(questionnaire_id, active_only=True)

        return {
            "user_question": getattr(state, "user_question", ""),
            "schema": schema,
        }

    def _to_patch(self, payload: Dict[str, Any], state: Any) -> Dict[str, Any]:
        notes = getattr(state, "notes", {}) or {}
        questionnaire_id = notes.get("questionnaire_id")

        mapped: List[Dict[str, Any]] = payload.get("mapped_columns", []) or []

        # If no questionnaire_id, we can only pass-through what the LLM produced.
        if not questionnaire_id:
            for m in mapped:
                m["confidence"] = self._clamp_confidence(m.get("confidence", 0.0))
            return {"mapped_columns": mapped}

        # Load schema and build a lookup to validate/fill fields deterministically.
        schema = self.repo.load_schema(questionnaire_id, active_only=True)
        schema_by_col = {q["column_name"]: q for q in schema}

        cleaned: List[Dict[str, Any]] = []
        for m in mapped:
            col = str(m.get("column_name", "")).strip()
            if not col:
                continue

            # Drop columns not present in schema (prevents hallucinated columns).
            q = schema_by_col.get(col)
            if q is None:
                continue

            conf = self._clamp_confidence(m.get("confidence", 0.0))

            # Fill missing metadata from schema.
            out = dict(m)
            out["column_name"] = col
            out["confidence"] = conf
            out.setdefault("question_id", q.get("question_id"))
            out.setdefault("question_text", q.get("question_text"))
            out.setdefault("privacy_level", q.get("privacy_level", "normal"))

            # Attach constraints if available_values exist in schema.
            allowed = q.get("allowed_values")
            if allowed is not None:
                out.setdefault("value_constraints", {"allowed_values": allowed})

            # Normalize confidence label if not provided.
            out.setdefault("confidence_label", self._label_confidence(conf))

            cleaned.append(out)

        # Sort by confidence (highest first) for downstream stability.
        cleaned.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

        return {"mapped_columns": cleaned}

    @staticmethod
    def _clamp_confidence(v: Any) -> float:
        try:
            c = float(v)
        except Exception:
            c = 0.0
        if c < 0.0:
            return 0.0
        if c > 1.0:
            return 1.0
        return c

    @staticmethod
    def _label_confidence(c: float) -> str:
        if c >= 0.80:
            return "high"
        if c >= 0.50:
            return "medium"
        return "low"
