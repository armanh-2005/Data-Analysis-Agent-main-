# src/agents/code_writer_agent.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal
from langchain_core.messages import HumanMessage

# ایمپورت‌های پروژه
from src.agents.utils import (
    get_llm,
    render_prompt,
    parse_json_object,
    validate_with_jsonschema,
    PromptNotFound,
    extract_python_code # اضافه شده برای حل مشکل extract
)

# Define allowed modes
AgentMode = Literal["analysis", "visualization"]

class CodeWriterAgent:
    name = "code_writer_agent"
    
    # Standard JSON Output Schema
    output_schema = {
        "type": "object",
        "properties": {"code_draft": {"type": "string"}},
        "required": ["code_draft"],
        "additionalProperties": True,
    }

    def __init__(self, model: str, mode: AgentMode, prompts_dir: str = "src/prompts"):
        """
        Args:
            model: Name of the LLM model to use.
            mode: 'analysis' for calculation code, 'visualization' for plotting code.
            prompts_dir: Directory containing markdown prompts.
        """
        self.llm = get_llm(model)
        self.prompts_dir = Path(prompts_dir)
        self.mode = mode
        
        # Select prompt file based on mode
        if self.mode == "analysis":
            self.prompt_file = "analyst.md"
        elif self.mode == "visualization":
            self.prompt_file = "visualizer.md"
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def run(self, state: Any) -> Any:
        """
        Executes the agent logic based on the mode.
        """
        # 1. Prepare Context Variables
        # Common variables
        notes = getattr(state, "notes", {}) or {}
        variables = {
            "user_question": getattr(state, "user_question", ""),
            "artifacts_dir": getattr(state, "artifacts_dir", "artifacts") or "artifacts",
            # Feedback from previous attempts (if any)
            "code_review": getattr(state, "code_review", {}), 
            "execution_error": getattr(state, "execution", {}).get("error_trace", "")
            
        }

        # Mode-specific variables
        if self.mode == "analysis":
            variables.update({
                "mapped_columns": getattr(state, "mapped_columns", []) or [],
                "analysis_plan": getattr(state, "analysis_plan", {}) or {},
                "questionnaire_id": getattr(state, "questionnaire_id", ""),
                # If retrying, pass previous code
                "previous_code": getattr(state, "analysis_code", ""),
                "analysis_output": getattr(state, "analysis_output", ""),
            })
            
        elif self.mode == "visualization":
            variables.update({
                # Visualization needs to know what happened in analysis
                "analysis_code": getattr(state, "analysis_code", ""),
                "analysis_output": getattr(state, "analysis_output", ""),
                # If retrying, pass previous code
                "previous_code": getattr(state, "viz_code", ""),
            })

        # 2. Load & Render Prompt
        prompt_path = self.prompts_dir / self.prompt_file
        
        # Fallback: اگر فایل پیدا نشد، از پرامپت پیش‌فرض استفاده کن (برای جلوگیری از کرش)
        if not prompt_path.exists():
            # raise PromptNotFound(f"Prompt file not found: {prompt_path}")
            # استفاده از پرامپت داخلی به عنوان فال‌بک
            print(f"Warning: Prompt file {prompt_path} not found. Using default prompt.")
            if self.mode == "analysis":
                prompt_text = "You are a Data Analyst. Write python code using variable df to analyze: {{user_question}}. Context: {{analysis_plan}}"
            else:
                prompt_text = "You are a Data Visualizer. Write python code using matplotlib/seaborn to visualize the data. Context: {{user_question}}"
        else:
            prompt_text = prompt_path.read_text(encoding="utf-8")
            
        rendered_prompt = render_prompt(prompt_text, variables)

        # 3. Call LLM
        messages = [HumanMessage(content=rendered_prompt)]
        response = self.llm.invoke(messages)

        print("\n" + "="*50)
        print("RAW LLM RESPONSE:")
        print(response.content)
        print("="*50 + "\n")

        # 4. Parse & Validate
        # به جای تلاش برای جیسون، کد را تمیز میکنیم
        code_content = extract_python_code(response.content)

        # اصلاح مهم: کلید دیکشنری باید با output_schema یکی باشد (code_draft)
        payload = {"code_draft": code_content}
        
        validate_with_jsonschema(payload, self.output_schema)
        
        generated_code = payload.get("code_draft", "")

        # 5. Update State (Targeting specific fields)
        if hasattr(state, "patch"):
            if self.mode == "analysis":
                # In analysis mode, we update 'analysis_code' and reset review/execution status
                return state.patch(
                    analysis_code=generated_code,
                    code_draft=generated_code, # Keep for backward compatibility if needed
                    # Reset execution flags for new run
                    execution={}, 
                    code_review={}
                )
            else:
                # In visualization mode, we update 'viz_code'
                return state.patch(
                    viz_code=generated_code,
                    code_draft=generated_code,
                    execution={},
                    code_review={}
                )
        else:
            # Fallback for dict-based state
            if self.mode == "analysis":
                state["analysis_code"] = generated_code
            else:
                state["viz_code"] = generated_code
            state["code_draft"] = generated_code
            return state