# src/agents/quality_review_agent.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import HumanMessage

from src.agents.utils import (
    get_llm,
    render_prompt,
    parse_json_object,
    validate_with_jsonschema,
    PromptNotFound
)

class QualityReviewAgent:
    name = "quality_review_agent"
    prompt_file = "quality_review.md"
    
    output_schema = {
        "type": "object",
        "properties": {"quality_review": {"type": "object"}},
        "required": ["quality_review"],
        "additionalProperties": True,
    }

    def __init__(self, model: str, prompts_dir: str = "src/prompts"):
        self.llm = get_llm(model)
        self.prompts_dir = Path(prompts_dir)

    def run(self, state: Any) -> Any:
        """
        Executes the quality review logic:
        1. Prepares variables from state.
        2. Renders the prompt.
        3. Calls the LLM.
        4. Updates the state with the quality review result.
        """
        # 1. Prepare Inputs
        variables = {
            "user_question": getattr(state, "user_question", ""),
            "analysis_plan": getattr(state, "analysis_plan", {}) or {},
            "execution": getattr(state, "execution", {}) or {},
        }

        # 2. Render Prompt
        prompt_path = self.prompts_dir / self.prompt_file
        if not prompt_path.exists():
            # Fallback or error
            raise PromptNotFound(f"Prompt file not found: {prompt_path}")

        prompt_text = prompt_path.read_text(encoding="utf-8")
        rendered_prompt = render_prompt(prompt_text, variables)

        # 3. Call LLM
        messages = [HumanMessage(content=rendered_prompt)]
        response = self.llm.invoke(messages)

        # 4. Parse & Validate
        payload = parse_json_object(response.content)
        validate_with_jsonschema(payload, self.output_schema)
        
        quality_review = payload.get("quality_review", {})

        # 5. Update State
        if hasattr(state, "patch"):
            return state.patch(quality_review=quality_review)
        else:
            state["quality_review"] = quality_review
            return state