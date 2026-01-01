# src/agents/router_mapper_agent.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.db.repository import SQLiteRepository
from src.agents.utils import (
    get_llm, 
    render_prompt, 
    parse_json_object, 
    validate_with_jsonschema,
    PromptNotFound
)

class RouterMapperAgent:
    name = "router_mapper_agent"
    prompt_file = "router_mapper.md"
    
    # Combined schema for validation
    output_schema = {
        "type": "object",
        "properties": {
            "is_related": {"type": "boolean"},
            "reason": {"type": "string"},
            "mapped_columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column_name": {"type": "string"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                        "inferred_role": {"type": "string"},
                    },
                    "required": ["column_name", "confidence"],
                },
            }
        },
        "required": ["is_related", "mapped_columns"],
    }

    def __init__(self, model: str, db_path: str, prompts_dir: str = "src/prompts"):
        self.llm = get_llm(model)
        self.repo = SQLiteRepository(db_path)
        self.prompts_dir = Path(prompts_dir)

    def run(self, state: Any) -> Any:
        """
        Executes the agent logic:
        1. Loads schema from DB.
        2. Prompts LLM to check relevance AND map columns.
        3. Validates and enriches the result.
        4. Updates state.
        """
        # 1. Prepare Inputs
        user_question = getattr(state, "user_question", "")
        # Access notes/questionnaire_id safely
        notes = getattr(state, "notes", {}) or {}
        questionnaire_id = notes.get("questionnaire_id")
        
        # Load schema if we have a questionnaire ID
        schema_list: List[Dict[str, Any]] = []
        if questionnaire_id:
            schema_list = self.repo.load_schema(questionnaire_id, active_only=True)
            
        # 2. Build Prompt
        prompt_path = self.prompts_dir / self.prompt_file
        if not prompt_path.exists():
            raise PromptNotFound(f"Prompt file not found: {prompt_path}")
            
        prompt_text = prompt_path.read_text(encoding="utf-8")
        rendered_prompt = render_prompt(prompt_text, {
            "user_question": user_question,
            "schema": schema_list
        })

        # 3. Call LLM
        messages = [HumanMessage(content=rendered_prompt)]
        response = self.llm.invoke(messages)
        
        # 4. Parse Output
        payload = parse_json_object(response.content)
        validate_with_jsonschema(payload, self.output_schema)
        
        # 5. Process & Enrich Results
        is_related = payload.get("is_related", False)
        reason = payload.get("reason", "")
        raw_mappings = payload.get("mapped_columns", [])
        
        final_mappings = []
        
        # Only process mappings if the question is related and we have a schema to validate against
        if is_related and questionnaire_id and schema_list:
            # Create a lookup for fast validation/enrichment
            schema_by_col = {q["column_name"]: q for q in schema_list}
            
            for m in raw_mappings:
                col_name = m.get("column_name", "").strip()
                
                # Validation: Skip columns that don't exist in the schema (hallucination check)
                if col_name not in schema_by_col:
                    continue
                
                schema_def = schema_by_col[col_name]
                conf = self._clamp_confidence(m.get("confidence", 0.0))
                
                # Merge LLM output with Schema metadata
                enriched = {
                    "column_name": col_name,
                    "reason": m.get("reason", ""),
                    "confidence": conf,
                    "confidence_label": self._label_confidence(conf),
                    "inferred_role": m.get("inferred_role", "other"),
                    # Enriched fields from DB:
                    "question_id": schema_def.get("question_id"),
                    "question_text": schema_def.get("question_text"),
                    "privacy_level": schema_def.get("privacy_level", "normal"),
                }
                
                # Add constraints if they exist
                if schema_def.get("allowed_values"):
                    enriched["value_constraints"] = {"allowed_values": schema_def["allowed_values"]}
                    
                final_mappings.append(enriched)

            # Sort by confidence
            final_mappings.sort(key=lambda x: x["confidence"], reverse=True)

        # 6. Update State
        # We need to update: is_related, notes (router reason), and mapped_columns
        
        # Helper to get mutable dict from state or create new
        if hasattr(state, "patch"):
            # If using the new state.py with .patch() method
            updated_notes = dict(notes)
            updated_notes["router_reason"] = reason
            
            return state.patch(
                is_related=is_related,
                mapped_columns=final_mappings,
                notes=updated_notes
            )
        else:
            # Fallback for dictionary-based state
            state["is_related"] = is_related
            state["mapped_columns"] = final_mappings
            if "notes" not in state:
                state["notes"] = {}
            state["notes"]["router_reason"] = reason
            return state

    # --- Helper methods ---

    @staticmethod
    def _clamp_confidence(v: Any) -> float:
        try:
            c = float(v)
        except Exception:
            c = 0.0
        return max(0.0, min(1.0, c))

    @staticmethod
    def _label_confidence(c: float) -> str:
        if c >= 0.80: return "high"
        if c >= 0.50: return "medium"
        return "low"