# src/agents/base.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None


class LLM(Protocol):
    # Minimal LLM protocol to keep agents swappable.
    # Implementations may accept either a prompt string or chat messages and return a string.
    def __call__(self, prompt: str) -> str: ...


class AgentError(Exception):
    # Base error for agent failures.
    pass


class PromptNotFound(AgentError):
    pass


class AgentOutputParseError(AgentError):
    pass


class AgentOutputValidationError(AgentError):
    pass


@dataclass(frozen=True)
class AgentResponse:
    # Only the patch to be applied on the WorkflowState.
    patch: Dict[str, Any]
    raw_text: Optional[str] = None


def _read_text_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise PromptNotFound(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    """
    Render prompt using {{var}} placeholders to avoid .format() collisions with JSON examples.
    Unknown variables remain unchanged.
    """
    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in variables:
            return match.group(0)
        v = variables[key]
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, indent=2)
        return str(v)

    return re.sub(r"\{\{\s*([a-zA-Z0-9_\.]+)\s*\}\}", repl, template)


def parse_json_object(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from text.
    Tries strict parse; if it fails, tries to extract the first {...} block.
    """
    s = text.strip()
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise AgentOutputParseError("Expected a JSON object at top level.")
        return obj
    except Exception:
        # Best-effort extraction of a JSON object block
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise AgentOutputParseError("Could not find a JSON object in the model output.")
        try:
            obj = json.loads(s[start : end + 1])
            if not isinstance(obj, dict):
                raise AgentOutputParseError("Extracted JSON is not an object.")
            return obj
        except Exception as e:
            raise AgentOutputParseError(f"Failed to parse JSON object: {e}") from e


def validate_with_jsonschema(payload: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> None:
    if schema is None:
        return
    if jsonschema is None:
        # If jsonschema isn't installed, do a minimal check for required fields only.
        required = schema.get("required", [])
        for k in required:
            if k not in payload:
                raise AgentOutputValidationError(f"Missing required field: {k}")
        return
    try:
        jsonschema.validate(instance=payload, schema=schema)  # type: ignore[attr-defined]
    except Exception as e:
        raise AgentOutputValidationError(str(e)) from e


class BaseAgent:
    """
    Swappable agent base:
      - Loads prompt from prompts/<prompt_file> (optional fallback to default_prompt)
      - Calls LLM
      - Parses JSON object output
      - Validates output by jsonschema
      - Converts to a state patch (dict)
    """

    name: str = "base_agent"
    prompt_file: Optional[str] = None
    default_prompt: str = ""
    output_schema: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        llm: Optional[Callable[[str], str]] = None,
        prompts_dir: str = "src/prompts",
    ):
        self.llm = llm
        self.prompts_dir = Path(prompts_dir)

    def invoke(self, state: Any) -> AgentResponse:
        """
        Input: WorkflowState (dataclass) or dict-like object.
        Output: AgentResponse(patch=...)
        """
        prompt = self._build_prompt(state)

        if self.llm is None:
            raise AgentError(f"LLM is not configured for agent '{self.name}'.")

        raw = self.llm(prompt)
        payload = parse_json_object(raw)
        validate_with_jsonschema(payload, self.output_schema)
        patch = self._to_patch(payload, state)
        return AgentResponse(patch=patch, raw_text=raw)

    # -------------------------
    # Overridables
    # -------------------------

    def _build_variables(self, state: Any) -> Dict[str, Any]:
        # Override in child classes.
        return {}

    def _to_patch(self, payload: Dict[str, Any], state: Any) -> Dict[str, Any]:
        # Override in child classes if needed.
        return payload

    # -------------------------
    # Prompt building
    # -------------------------

    def _load_prompt_template(self) -> str:
        if self.prompt_file:
            path = self.prompts_dir / self.prompt_file
            return _read_text_file(path)
        if self.default_prompt.strip():
            return self.default_prompt
        raise PromptNotFound(f"No prompt_file/default_prompt defined for agent '{self.name}'.")

    def _build_prompt(self, state: Any) -> str:
        template = self._load_prompt_template()
        vars_ = self._build_variables(state)
        return render_prompt(template, vars_)
