# src/agents/code_reviewer_agent.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Tuple

from langchain_core.messages import HumanMessage

from src.agents.utils import (
    get_llm,
    render_prompt,
    parse_json_object,
    validate_with_jsonschema,
    PromptNotFound
)


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
        r"\bopen\s*\(",
        r"\b__import__\b",
    ]
    issues: List[str] = []
    for pat in forbidden_patterns:
        if re.search(pat, code):
            issues.append(f"Forbidden pattern detected: {pat}")
    return (len(issues) == 0), issues


class CodeReviewerAgent:
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

    def __init__(self, model: str, prompts_dir: str = "src/prompts"):
        self.llm = get_llm(model)
        self.prompts_dir = Path(prompts_dir)

    def run(self, state: Any) -> Any:
        """
        Executes the code review logic:
        1. Performs a static safety scan (regex-based).
        2. If unsafe, returns rejection immediately.
        3. If safe, calls LLM for logic/stats review.
        4. Updates state with 'code_review'.
        """
        code_draft = getattr(state, "code_draft", "") or ""

        # 1. Static Safety Scan
        is_safe, issues = _static_safety_scan(code_draft)
        
        if not is_safe:
            # Immediate rejection without LLM
            review_payload = {
                "approved": False,
                "feedback": "Static safety scan failed. Remove forbidden imports/calls.",
                "issues": [{"type": "safety", "detail": x} for x in issues],
                "score": 0.0,
            }
            return self._update_state(state, review_payload)

        # 2. Prepare Inputs for LLM
        variables = {
            "analysis_plan": getattr(state, "analysis_plan", {}) or {},
            "stats_params": getattr(state, "stats_params", {}) or {},
            "code_draft": code_draft,
        }

        # 3. Render Prompt
        prompt_path = self.prompts_dir / self.prompt_file
        if not prompt_path.exists():
            raise PromptNotFound(f"Prompt file not found: {prompt_path}")

        prompt_text = prompt_path.read_text(encoding="utf-8")
        rendered_prompt = render_prompt(prompt_text, variables)

        # 4. Call LLM
        messages = [HumanMessage(content=rendered_prompt)]
        response = self.llm.invoke(messages)

        # 5. Parse & Validate
        payload = parse_json_object(response.content)
        validate_with_jsonschema(payload, self.output_schema)
        
        code_review = payload.get("code_review", {})
        
        # 6. Update State
        return self._update_state(state, code_review)

    def _update_state(self, state: Any, review: dict) -> Any:
        """Helper to update state safely."""
        if hasattr(state, "patch"):
            return state.patch(code_review=review)
        else:
            state["code_review"] = review
            return state