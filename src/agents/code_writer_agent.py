# src/agents/code_writer_agent.py
from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class CodeWriterAgent(BaseAgent):
    name = "code_writer_agent"
    prompt_file = "code_writer.md"
    output_schema = {
        "type": "object",
        "properties": {"code_draft": {"type": "string"}},
        "required": ["code_draft"],
        "additionalProperties": True,
    }

    default_prompt = """
You write Python analysis code.
Constraints:
- Use SQLite (read-only) via sqlite3 and pandas.
- Use ONLY: sqlite3, json, math, statistics, pandas, numpy, scipy, statsmodels, matplotlib.
- Do not write to network. File writes only into artifacts_dir.
- Produce results as a JSON dict assigned to a variable named RESULTS, and artifact paths in ARTIFACTS (list).

Return ONLY JSON:
{ "code_draft": "..." }

Inputs:
db_path={{db_path}}
questionnaire_id={{questionnaire_id}}
mapped_columns={{mapped_columns}}
analysis_plan={{analysis_plan}}
stats_params={{stats_params}}
artifacts_dir={{artifacts_dir}}
""".strip()

    def _build_variables(self, state: Any) -> Dict[str, Any]:
        notes = getattr(state, "notes", {}) or {}
        return {
            "db_path": notes.get("db_path", "data/app.db"),
            "questionnaire_id": notes.get("questionnaire_id", ""),
            "artifacts_dir": notes.get("artifacts_dir", "artifacts"),
            "mapped_columns": getattr(state, "mapped_columns", []) or [],
            "analysis_plan": getattr(state, "analysis_plan", {}) or {},
            "stats_params": getattr(state, "stats_params", {}) or {},
        }

    def _to_patch(self, payload: Dict[str, Any], state: Any) -> Dict[str, Any]:
        return {"code_draft": payload.get("code_draft", "") or ""}
