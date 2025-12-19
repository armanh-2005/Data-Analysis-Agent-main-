# src/agents/planner_agent.py
from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "planner_agent"
    prompt_file = "planner.md"
    output_schema = {
        "type": "object",
        "properties": {"analysis_plan": {"type": "object"}},
        "required": ["analysis_plan"],
        "additionalProperties": True,
    }

    default_prompt = """
You are an analysis planner.
Given the user's question, mapped columns, and data profile, produce a robust analysis plan.
Return ONLY JSON:
{
  "analysis_plan": {
     "goal": "...",
     "columns_used": ["..."],
     "analysis_type": "descriptive|comparison|correlation|modeling",
     "steps": ["..."],
     "assumptions": ["..."],
     "outputs": ["tables","charts","metrics"]
  }
}

User question:
{{user_question}}

Mapped columns:
{{mapped_columns}}

Data profile:
{{data_profile}}
""".strip()

    def _build_variables(self, state: Any) -> Dict[str, Any]:
        return {
            "user_question": getattr(state, "user_question", ""),
            "mapped_columns": getattr(state, "mapped_columns", []) or [],
            "data_profile": getattr(state, "data_profile", None) or {},
        }

    def _to_patch(self, payload: Dict[str, Any], state: Any) -> Dict[str, Any]:
        return {"analysis_plan": payload.get("analysis_plan", {}) or {}}
