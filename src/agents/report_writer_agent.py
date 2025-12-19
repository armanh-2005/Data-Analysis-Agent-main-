# src/agents/report_writer_agent.py
from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class ReportWriterAgent(BaseAgent):
    name = "report_writer_agent"
    prompt_file = "report_writer.md"
    output_schema = {
        "type": "object",
        "properties": {"final_report": {"type": "string"}},
        "required": ["final_report"],
        "additionalProperties": True,
    }

    default_prompt = """
You are a report writer.
Write a clear report in Markdown. Include methods, findings, limitations.
Do NOT invent numbers; only use provided execution results.

Return ONLY JSON:
{ "final_report": "..." }

User question:
{{user_question}}

Mapped columns:
{{mapped_columns}}

Analysis plan:
{{analysis_plan}}

Execution results:
{{execution}}

Quality review:
{{quality_review}}
""".strip()

    def _build_variables(self, state: Any) -> Dict[str, Any]:
        return {
            "user_question": getattr(state, "user_question", ""),
            "mapped_columns": getattr(state, "mapped_columns", []) or [],
            "analysis_plan": getattr(state, "analysis_plan", {}) or {},
            "execution": getattr(state, "execution", {}) or {},
            "quality_review": getattr(state, "quality_review", {}) or {},
        }

    def _to_patch(self, payload: Dict[str, Any], state: Any) -> Dict[str, Any]:
        return {"final_report": payload.get("final_report", "") or ""}
