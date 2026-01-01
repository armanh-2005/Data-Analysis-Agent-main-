# src/agents/stats_params_agent.py
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


class StatsParamsAgent:
    name = "stats_params_agent"
    prompt_file = "stats_params.md"
    
    output_schema = {
        "type": "object",
        "properties": {"stats_params": {"type": "object"}},
        "required": ["stats_params"],
        "additionalProperties": True,
    }

    def __init__(self, model: str, prompts_dir: str = "src/prompts"):
        self.llm = get_llm(model)
        self.prompts_dir = Path(prompts_dir)

    def run(self, state: Any) -> Any:
        """
        Executes the stats params logic:
        1. Reads analysis plan and data profile from state.
        2. Suggests statistical tests/parameters via LLM.
        3. Updates state with 'stats_params'.
        """
        # 1. Prepare Inputs
        variables = {
            "analysis_plan": getattr(state, "analysis_plan", {}) or {},
            "data_profile": getattr(state, "data_profile", None) or {},
        }

        # 2. Render Prompt
        prompt_path = self.prompts_dir / self.prompt_file
        if not prompt_path.exists():
            raise PromptNotFound(f"Prompt file not found: {prompt_path}")

        prompt_text = prompt_path.read_text(encoding="utf-8")
        rendered_prompt = render_prompt(prompt_text, variables)

        # 3. Call LLM
        messages = [HumanMessage(content=rendered_prompt)]
        response = self.llm.invoke(messages)

        # 4. Parse & Validate
        payload = parse_json_object(response.content)
        validate_with_jsonschema(payload, self.output_schema)
        
        stats_params = payload.get("stats_params", {})

        # 5. Update State
        if hasattr(state, "patch"):
            return state.patch(stats_params=stats_params)
        else:
            state["stats_params"] = stats_params
            return state