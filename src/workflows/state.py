# src/workflows/state.py
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Literal


# -------------------------
# Typed payloads (JSON-friendly)
# -------------------------

Confidence = Literal["low", "medium", "high"]


class MappedColumn(TypedDict, total=False):
    column_name: str
    question_id: str
    question_text: str
    reason: str
    confidence: float
    confidence_label: Confidence
    inferred_role: str
    value_constraints: Dict[str, Any]
    privacy_level: str


class CodeReview(TypedDict, total=False):
    approved: bool
    feedback: str
    issues: List[Dict[str, Any]]
    safety: Dict[str, Any]
    score: float


class ExecutionResult(TypedDict, total=False):
    status: str
    stdout: str
    stderr: str
    results_json: Dict[str, Any]
    artifacts: List[Dict[str, Any]]
    started_at: str
    finished_at: str
    runtime_seconds: float


class QualityReview(TypedDict, total=False):
    approved: bool
    feedback: str
    issues: List[Dict[str, Any]]
    score: float
    sufficiency: Dict[str, Any]


ProfileSummary = Dict[str, Any]


# -------------------------
# Core State (single object passed around LangGraph)
# -------------------------

@dataclass
class WorkflowState:
    # Identity / request
    run_id: str
    user_question: str
    questionnaire_id: str  # <--- فیلد جدید اضافه شد
    schema_summary: Optional[List[str]] = None  # <--- فیلد جدید اضافه شد

    # Routing / mapping
    is_related: Optional[bool] = None
    mapped_columns: List[MappedColumn] = field(default_factory=list)

    # Data / planning
    data_profile: Optional[ProfileSummary] = None
    analysis_plan: Dict[str, Any] = field(default_factory=dict)
    stats_params: Dict[str, Any] = field(default_factory=dict)

    # Code loop
    code_draft: str = ""
    code_review: CodeReview = field(default_factory=dict)
    execution: ExecutionResult = field(default_factory=dict)

    # Quality + report
    quality_review: QualityReview = field(default_factory=dict)
    final_report: str = ""

    # Memory/state for NoteAgent
    notes: Dict[str, Any] = field(default_factory=dict)

    # Internal bookkeeping
    iteration: Dict[str, int] = field(default_factory=lambda: {"code": 0, "quality": 0})
    created_at: str = field(default_factory=lambda: datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().replace(microsecond=0).isoformat() + "Z")

    # -------------------------
    # Utilities (Dict Access Support)
    # -------------------------
    
    # اضافه کردن قابلیت دسترسی مثل دیکشنری (state["key"]) برای سازگاری با ایجنت‌ها
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        self.touch()

    def get(self, key, default=None):
        return getattr(self, key, default)

    def touch(self) -> None:
        self.updated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def patch(self, **updates: Any) -> "WorkflowState":
        for k, v in updates.items():
            if not hasattr(self, k):
                continue
            setattr(self, k, v)
        self.touch()
        return self

    def increment(self, key: str) -> None:
        self.iteration.setdefault(key, 0)
        self.iteration[key] += 1
        self.touch()

    # -------------------------
    # JSON serialization
    # -------------------------

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        return _json_sanitize(raw)

    def to_json(self, ensure_ascii: bool = False, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorkflowState":
        allowed = {f.name for f in WorkflowState.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in allowed}
        
        # چک کردن فیلدهای اجباری
        if "run_id" not in filtered or "user_question" not in filtered:
             # برای سازگاری با کدهای قدیمی یا تست، مقادیر پیش‌فرض می‌دهیم اگر نبودند
             pass 

        state = WorkflowState(**filtered)
        state.touch()
        return state

    @staticmethod
    def from_json(s: str) -> "WorkflowState":
        return WorkflowState.from_dict(json.loads(s))


def _json_sanitize(obj: Any) -> Any:
    if obj is None: return None
    if isinstance(obj, (str, int, float, bool)): return obj
    if isinstance(obj, datetime): return obj.replace(microsecond=0).isoformat() + "Z"
    if is_dataclass(obj): return _json_sanitize(asdict(obj))
    if isinstance(obj, dict): return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [_json_sanitize(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)