# src/agents/utils.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Try importing LangChain components
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    import jsonschema
except ImportError:
    jsonschema = None


# -------------------------
# Exceptions
# -------------------------

class AgentError(Exception):
    pass


class PromptNotFound(AgentError):
    pass


class AgentOutputParseError(AgentError):
    pass


class AgentOutputValidationError(AgentError):
    pass


# -------------------------
# LLM Factory
# -------------------------

def get_llm(model_name: str, temperature: float = 0.0):
    """
    Creates and returns a configured ChatOpenAI client.
    Reads API credentials from environment variables.
    """
    # Load environment variables (e.g. from .env file)
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = "https://api.avalai.ir/v1"

    if not ChatOpenAI:
        raise ImportError("langchain_openai is not installed. Please install it to use the agents.")

    # Note: We rely on the library to handle missing API keys (or passed explicitly below)
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        base_url=base_url
    )


# -------------------------
# Helper Functions
# -------------------------

def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    """
    Replaces {{ variable }} placeholders in the template with values from the dictionary.
    Handles JSON serialization for dict/list values.
    """
    def repl(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in variables:
            # Leave the placeholder if variable is missing
            return match.group(0)
        
        v = variables[key]
        if isinstance(v, (dict, list)):
            return json.dumps(v, ensure_ascii=False, indent=2)
        return str(v)

    return re.sub(r"\{\{\s*([a-zA-Z0-9_\.]+)\s*\}\}", repl, template)


def parse_json_object(text: str) -> Dict[str, Any]:
    """
    Extracts and parses a JSON object from a string.
    Handles Markdown code blocks (```json ... ```) and surrounding text.
    """
    s = text.strip()
    
    # Remove Markdown code blocks if present
    if s.startswith("```"):
        newline_idx = s.find("\n")
        if newline_idx != -1:
            s = s[newline_idx+1:]
        if s.endswith("```"):
            s = s[:-3]
    s = s.strip()

    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            raise AgentOutputParseError("Expected a JSON object at top level.")
        return obj
    except Exception:
        # Fallback: try to find the first '{' and last '}'
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise AgentOutputParseError(f"Could not find a JSON object in output: {text[:100]}...")
        try:
            obj = json.loads(s[start : end + 1])
            if not isinstance(obj, dict):
                raise AgentOutputParseError("Extracted JSON is not an object.")
            return obj
        except Exception as e:
            raise AgentOutputParseError(f"Failed to parse JSON object: {e}") from e


def validate_with_jsonschema(payload: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> None:
    """
    Validates a dictionary against a JSON schema.
    Falls back to simple required field check if jsonschema lib is missing.
    """
    if schema is None:
        return

    if jsonschema is None:
        # Simple fallback: check required fields only
        required = schema.get("required", [])
        for k in required:
            if k not in payload:
                raise AgentOutputValidationError(f"Missing required field: {k}")
        return

    try:
        jsonschema.validate(instance=payload, schema=schema)
    except Exception as e:
        raise AgentOutputValidationError(str(e)) from e
def extract_python_code(text: str) -> str:
    """
    استخراج کد پایتون از بین تگ‌های مارک‌داون
    """
    # الگوی پیدا کردن متن بین ```python و ```
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # اگر تگ پیدا شد، محتوای داخلش را برگردان
        return match.group(1).strip()
    elif "```" in text:
        # اگر تگ python نداشت اما تگ کد داشت
        pattern = r"```(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    else:
        # اگر هیچ تگی نبود، کل متن را به عنوان کد برگردان (ریسک دارد اما لازم است)
        return text.strip()