# src/agents/code_reviewer_agent.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .base import BaseAgent


def _static_safety_scan(code: str) -> Tuple[bool, List[str]]:
    """
    Deterministic safety checks before any execution.
    This is not a full sandbox; it's a guardrail.
    """
    forbidden_patterns = [
        r"\bimport\s+os\b",
        r"\bimport\s+sys\b",
        r"\bimport\s+subprocess\b",
        r"\bimport\s+socket\b",
        r"\bimport\s+requests\b",
        r"\bimport\s+http\b",
        r"\bimport\s+urllib\b",
        r"\bfrom\s+os\b",
        r"\bfrom\s+sys\b",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bopen\s*\(",  # file IO should be controlled; executor can provide safe APIs later
        r"\b__import__\b",
    ]
    issues: List[str] = []
    for pat in forbidden_patterns:
        if re.search(pat, code):
            issues.append(f"Forbidden pattern detected: {pat}")
    return (len(issues) == 0), issues


class CodeReviewerAgent(BaseAgent):
    name = "code_reviewer_agent"
    prompt_file = "code_reviewer.md"
    output_schema = {
        "type": "object",
        "properties": {
            "code_review": {
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean"},
                    "feedback": {"type": "string"},
                    "issues": {"type": "array"},
                    "score": {"type": "number"},
                },
                "required": ["approved", "feedback"],
                "additionalProperties": True,
            }
        },
        "required": ["code_review"],
        "additionalProperties": True,
    }

    default_prompt = """
You are a strict code reviewer for data analysis code.
Return ONLY JSON:
{
  "code_review": {
    "approved": true/false,
    "feedback": "actionable feedback",
    "issues": [{"type":"...","detail":"..."}],
    "score": 0.0
  }
}

Analysis plan:
{{analysis_plan}}

Stats params:
{{stats_params}}

Code:
{{code_draft}}
""".strip()

    def invoke(self, state: Any):
        # First run deterministic scan; reject immediately if unsafe.
        code = getattr(state, "code_draft", "") or ""
        ok, issues = _static_safety_scan(code)
        if not ok:
            return super()._to_patch(
                {
                    "code_review": {
                        "approved": False,
                        "feedback": "Static safety scan failed. Remove forbidden imports/calls.",
                        "issues": [{"type": "safety", "detail": x} for x in issues],
                        "score": 0.0,
                    }
                },
                state,
            )  # type: ignore[return-value]

        # If LLM is available, use it for deeper review; otherwise approve by default.
        if self.llm is None:
            return {"code_review": {"approved": True, "feedback": "Approved by static scan (no LLM review).", "score": 0.7}}

        resp = super().invoke(state)
        return resp.patch

    def _build_variables(self, state: Any) -> Dict[str, Any]:
        return {
            "analysis_plan": getattr(state, "analysis_plan", {}) or {},
            "stats_params": getattr(state, "stats_params", {}) or {},
            "code_draft": getattr(state, "code_draft", "") or "",
        }

    def _to_patch(self, payload: Dict[str, Any], state: Any) -> Dict[str, Any]:
        return {"code_review": payload.get("code_review", {}) or {}}
