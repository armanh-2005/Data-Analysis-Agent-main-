# src/agents/stats_params_agent.py
from __future__ import annotations

from typing import Any, Dict

from .base import BaseAgent


class StatsParamsAgent(BaseAgent):
    name = "stats_params_agent"
    prompt_file = "stats_params.md"
    output_schema = {
        "type": "object",
        "properties": {"stats_params": {"type": "object"}},
        "required": ["stats_params"],
        "additionalProperties": True,
    }

    default_prompt = """
You choose statistical parameters and tests based on the plan and data profile.
Return ONLY JSON:
{
  "stats_params": {
    "alpha": 0.05,
    "tests": ["..."],
    "effect_sizes": ["..."],
    "multiple_testing": {"method":"none|bonferroni|fdr_bh"},
    "notes": ["..."]
  }
}

Analysis plan:
{{analysis_plan}}

Data profile:
{{data_profile}}
""".strip()

    def _build_variables(self, state: Any) -> Dict[str, Any]:
        return {
            "analysis_plan": getattr(state, "analysis_plan", {}) or {},
            "data_profile": getattr(state, "data_profile", None) or {},
        }

    def _to_patch(self, payload: Dict[str, Any], state: Any) -> Dict[str, Any]:
        return {"stats_params": payload.get("stats_params", {}) or {}}
