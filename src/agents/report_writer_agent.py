# src/agents/report_writer_agent.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from src.agents.utils import (
    get_llm,
    render_prompt,
    parse_json_object,
    validate_with_jsonschema,
    PromptNotFound
)


class ReportWriterAgent:
    name = "report_writer_agent"
    prompt_file = "report_writer.md"
    
    output_schema = {
        "type": "object",
        "properties": {"final_report": {"type": "string"}},
        "required": ["final_report"],
        "additionalProperties": True,
    }

    def __init__(self, model: str, prompts_dir: str = "src/prompts"):
        self.llm = get_llm(model)
        self.prompts_dir = Path(prompts_dir)

    def run(self, state: Any) -> Any:
        """
        Executes the report writer logic:
        1. Gathers context (question, plan, execution results).
        2. Prompts LLM to write the final markdown report.
        3. Updates state with 'final_report'.
        """
        # 1. Prepare Inputs
        variables = {
            "user_question": getattr(state, "user_question", ""),
            "mapped_columns": getattr(state, "mapped_columns", []) or [],
            "analysis_plan": getattr(state, "analysis_plan", {}) or {},
            "execution": getattr(state, "execution", {}) or {},
            "quality_review": getattr(state, "quality_review", {}) or {},
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
        
        final_report = payload.get("final_report", "")

        # 5. Update State
        if hasattr(state, "patch"):
            return state.patch(final_report=final_report)
        else:
            state["final_report"] = final_report
            return state