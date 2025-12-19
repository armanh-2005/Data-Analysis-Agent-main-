# src/agents/router_agent.py
from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent, AgentResponse


class RouterAgent(BaseAgent):
    name = "router_agent"
    prompt_file = "router.md"
    output_schema = {
        "type": "object",
        "properties": {
            "is_related": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["is_related", "reason"],
        "additionalProperties": True,
    }

    default_prompt = """
You are a router. Decide if the user's question is related to the questionnaire dataset.
Return ONLY valid JSON:
{
  "is_related": true/false,
  "reason": "short explanation"
}

User question:
{{user_question}}

Optional context:
{{context}}
""".strip()

    def _build_variables(self, state: Any) -> Dict[str, Any]:
        notes = getattr(state, "notes", {}) or {}
        return {
            "user_question": getattr(state, "user_question", ""),
            "context": notes.get("questionnaire_context", ""),
        }

    def _to_patch(self, payload: Dict[str, Any], state: Any) -> Dict[str, Any]:
        notes = dict(getattr(state, "notes", {}) or {})
        notes["router_reason"] = payload.get("reason")
        return {"is_related": bool(payload["is_related"]), "notes": notes}
