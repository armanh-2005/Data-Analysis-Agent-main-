# src/agents/quality_review_agent.py
from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class QualityReviewAgent(BaseAgent):
    name = "quality_review_agent"
    prompt_file = "quality_review.md"
    output_schema = {
        "type": "object",
        "properties": {"quality_review": {"type": "object"}},
        "required": ["quality_review"],
        "additionalProperties": True,
    }

    default_prompt = """
You are a quality reviewer.
Given the user question, analysis plan, and execution results, decide if results are sufficient and correct.
Return ONLY JSON:
{
  "quality_review": {
    "approved": true/false,
    "feedback": "...",
    "issues": [{"type":"...","detail":"..."}],
    "score": 0.0
  }
}

User question:
{{user_question}}

Analysis plan:
{{analysis_plan}}

Execution:
{{execution}}
""".strip()

    def _build_variables(self, state: Any) -> Dict[str, Any]:
        return {
            "user_question": getattr(state, "user_question", ""),
            "analysis_plan": getattr(state, "analysis_plan", {}) or {},
            "execution": getattr(state, "execution", {}) or {},
        }

    def _to_patch(self, payload: Dict[str, Any], state: Any) -> Dict[str, Any]:
        return {"quality_review": payload.get("quality_review", {}) or {}}
