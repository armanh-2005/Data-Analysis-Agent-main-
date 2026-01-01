# src/agents/code_writer_agent.py
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


class CodeWriterAgent:
    name = "code_writer_agent"
    prompt_file = "code_writer.md"
    
    output_schema = {
        "type": "object",
        "properties": {"code_draft": {"type": "string"}},
        "required": ["code_draft"],
        "additionalProperties": True,
    }

    def __init__(self, model: str, prompts_dir: str = "src/prompts"):
        self.llm = get_llm(model)
        self.prompts_dir = Path(prompts_dir)

    def run(self, state: Any) -> Any:
        """
        Executes the code writer logic:
        1. Prepares context variables (db_path, plan, params).
        2. Prompts LLM to generate analysis code.
        3. Updates state with 'code_draft'.
        """
        # 1. Prepare Inputs
        # Extract context from 'notes' (usually set by UI/Config) or defaults
        notes = getattr(state, "notes", {}) or {}
        
        variables = {
            "db_path": notes.get("db_path", "data/app.db"),
            "questionnaire_id": getattr(state, "questionnaire_id", "") or notes.get("questionnaire_id", ""),
            "artifacts_dir": notes.get("artifacts_dir", "artifacts"),
            "mapped_columns": getattr(state, "mapped_columns", []) or [],
            "analysis_plan": getattr(state, "analysis_plan", {}) or {},
            "stats_params": getattr(state, "stats_params", {}) or {},
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
        
        code_draft = payload.get("code_draft", "")

        # 5. Update State
        if hasattr(state, "patch"):
            return state.patch(code_draft=code_draft)
        else:
            state["code_draft"] = code_draft
            return state