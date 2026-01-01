# src/agents/planner_agent.py
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


class PlannerAgent:
    name = "planner_agent"
    prompt_file = "planner.md"
    
    # Schema for validation
    output_schema = {
        "type": "object",
        "properties": {"analysis_plan": {"type": "object"}},
        "required": ["analysis_plan"],
        "additionalProperties": True,
    }

    def __init__(self, model: str, prompts_dir: str = "src/prompts"):
        self.llm = get_llm(model)
        self.prompts_dir = Path(prompts_dir)

    def run(self, state: Any) -> Any:
        """
        Executes the planner logic:
        1. Prepares variables from state.
        2. Renders the prompt.
        3. Calls the LLM.
        4. Updates the state with the analysis plan.
        """
        # 1. Prepare Inputs
        variables = {
            "user_question": getattr(state, "user_question", ""),
            "mapped_columns": getattr(state, "mapped_columns", []) or [],
            "data_profile": getattr(state, "data_profile", None) or {},
        }

        # 2. Load & Render Prompt
        prompt_path = self.prompts_dir / self.prompt_file
        if not prompt_path.exists():
            # You could add a default string fallback here if desired, 
            # but raising an error ensures configuration is correct.
            raise PromptNotFound(f"Prompt file not found: {prompt_path}")

        prompt_text = prompt_path.read_text(encoding="utf-8")
        rendered_prompt = render_prompt(prompt_text, variables)

        # 3. Call LLM
        messages = [HumanMessage(content=rendered_prompt)]
        response = self.llm.invoke(messages)

        # 4. Parse & Validate
        payload = parse_json_object(response.content)
        validate_with_jsonschema(payload, self.output_schema)
        
        analysis_plan = payload.get("analysis_plan", {})

        # 5. Update State
        if hasattr(state, "patch"):
            return state.patch(analysis_plan=analysis_plan)
        else:
            state["analysis_plan"] = analysis_plan
            return state