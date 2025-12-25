# src/agents/base.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

# اضافه کردن کتابخانه‌های LangChain برای اتصال به هوش مصنوعی
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    # جهت جلوگیری از خطا در زمان ایمپورت اگر نصب نباشند
    ChatOpenAI = None

try:
    import jsonschema
except Exception:
    jsonschema = None


class AgentError(Exception):
    pass


class PromptNotFound(AgentError):
    pass


class AgentOutputParseError(AgentError):
    pass


class AgentOutputValidationError(AgentError):
    pass


@dataclass(frozen=True)
class AgentResponse:
    patch: Dict[str, Any]
    raw_text: Optional[str] = None


def _read_text_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        raise PromptNotFound(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, variables: Dict[str, Any]) -> str:
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
    s = text.strip()
    # حذف تگ‌های احتمالی Markdown مثل ```json
    if s.startswith("```"):
        # پیدا کردن اولین خط جدید
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
    if schema is None:
        return
    if jsonschema is None:
        required = schema.get("required", [])
        for k in required:
            if k not in payload:
                raise AgentOutputValidationError(f"Missing required field: {k}")
        return
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except Exception as e:
        raise AgentOutputValidationError(str(e)) from e


class BaseAgent:
    """
    کلاس پایه برای تمام ایجنت‌ها.
    اکنون از LangChain و پارامتر model پشتیبانی می‌کند.
    """

    name: str = "base_agent"
    prompt_file: Optional[str] = None
    default_prompt: str = ""
    output_schema: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        model: str = "gpt-4o-mini", 
        prompts_dir: str = "src/prompts",
    ):
        # بارگذاری مجدد دات‌انو در اینجا برای اطمینان
        from dotenv import load_dotenv
        import os
        load_dotenv()

        self.model_name = model
        self.prompts_dir = Path(prompts_dir)
        
        # خواندن مستقیم از متغیرهای محیطی
        api_key = os.getenv("OPENAI_API_KEY")
        
        base_url = "https://api.avalai.ir/v1"

        if ChatOpenAI:
            self.llm = ChatOpenAI(
                model=model,
                temperature=0,
                openai_api_key=api_key, # ارسال صریح کلید
                base_url=base_url       # ارسال صریح آدرس
            )
    def run(self, state: Any) -> Any:
        """
        متد اصلی که توسط UI فراخوانی می‌شود.
        اجرا می‌کند و تغییرات را روی State اعمال می‌کند.
        """
        response = self.invoke(state)
        
        # اگر state متد patch دارد (که در state.py جدید اضافه کردیم)، از آن استفاده کن
        if hasattr(state, "patch"):
            return state.patch(**response.patch)
        return state

    def invoke(self, state: Any) -> AgentResponse:
        """
        ساخت پرامپت، ارسال به LLM، و پردازش خروجی.
        """
        prompt_text = self._build_prompt(state)

        if self.llm is None:
            raise AgentError(f"LLM is not configured/installed for agent '{self.name}'.")

        # ارسال پیام به مدل
        messages = [HumanMessage(content=prompt_text)]
        ai_msg = self.llm.invoke(messages)
        raw = ai_msg.content

        # پردازش JSON
        payload = parse_json_object(raw)
        validate_with_jsonschema(payload, self.output_schema)
        patch = self._to_patch(payload, state)
        
        return AgentResponse(patch=patch, raw_text=raw)

    # -------------------------
    # متدهای داخلی
    # -------------------------

    def _build_variables(self, state: Any) -> Dict[str, Any]:
        return {}

    def _to_patch(self, payload: Dict[str, Any], state: Any) -> Dict[str, Any]:
        return payload

    def _load_prompt_template(self) -> str:
        if self.prompt_file:
            path = self.prompts_dir / self.prompt_file
            if path.exists():
                return _read_text_file(path)
        
        if self.default_prompt.strip():
            return self.default_prompt
            
        raise PromptNotFound(f"No prompt_file/default_prompt defined for agent '{self.name}'.")

    def _build_prompt(self, state: Any) -> str:
        template = self._load_prompt_template()
        vars_ = self._build_variables(state)
        return render_prompt(template, vars_)