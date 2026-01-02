# src/workflows/state.py
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Literal, Union

# -------------------------
# 1. Type Definitions (Complex Objects)
# -------------------------
# این کلاس‌ها ساختار دیکشنری‌های داخلی را برای ایجنت‌ها شفاف می‌کنند.

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
    # فیلدهای اضافی احتمالی
    original_header: str

class CodeReview(TypedDict, total=False):
    approved: bool
    feedback: str
    issues: List[Dict[str, Any]]
    safety: Dict[str, Any]
    score: float

class ExecutionResult(TypedDict, total=False):
    success: bool      # وضعیت کلی اجرا
    status: str        # error / success
    stdout: str
    stderr: str
    results_json: Dict[str, Any]
    artifacts: List[str] # لیست مسیر فایل‌های تولید شده (تصاویر و ...)
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
# 2. Workflow State (The Core Data Class)
# -------------------------

@dataclass
class WorkflowState:
    """
    وضعیت مرکزی که بین تمام ایجنت‌ها دست‌به‌دست می‌شود.
    شامل شناسه اجرا، سوال کاربر، وضعیت دیتابیس و خروجی مراحل مختلف است.
    """
    
    # --- Core Identity ---
    run_id: str
    user_question: str
    questionnaire_id: Optional[str] = None  # شناسه فایل/پرسشنامه در دیتابیس
    
    # --- Metadata & Context ---
    schema_summary: List[str] = field(default_factory=list) # لیست نام ستون‌ها
    data_profile: Dict[str, Any] = field(default_factory=dict) # خلاصه آماری داده‌ها
    
    # --- Router / Mapper Stage ---
    is_related: Optional[bool] = None
    mapped_columns: List[MappedColumn] = field(default_factory=list)
    
    # --- Planning Stage ---
    analysis_plan: Dict[str, Any] = field(default_factory=dict)
    stats_params: Dict[str, Any] = field(default_factory=dict) # پارامترهای آماری استخراج شده

    # --- Coding & Execution Stage ---
    code_draft: str = ""
    code_review: CodeReview = field(default_factory=dict)
    execution: ExecutionResult = field(default_factory=dict)

    # --- Quality & Reporting Stage ---
    quality_review: QualityReview = field(default_factory=dict)
    final_report: str = ""

    # --- Shared Memory / Notes ---
    # ایجنت‌ها می‌توانند یادداشت‌های موقت یا استدلال‌های خود را اینجا بنویسند
    notes: Dict[str, Any] = field(default_factory=dict)

    # --- Bookkeeping ---
    iteration: Dict[str, int] = field(default_factory=lambda: {"code": 0, "quality": 0})
    created_at: str = field(default_factory=lambda: datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().replace(microsecond=0).isoformat() + "Z")

    # -------------------------
    # 3. Helper Methods
    # -------------------------

    def patch(self, **changes) -> "WorkflowState":
        """
        یک کپی جدید از استیت با مقادیر تغییر یافته برمی‌گرداند (Immutable Update).
        استفاده: new_state = state.patch(is_related=True, notes={...})
        """
        # اگر دیکشنری‌های تو در تو مثل notes را آپدیت می‌کنیم، بهتر است کپی بگیریم
        # اما برای سادگی و سرعت، از replace استاندارد استفاده می‌کنیم.
        # ایجنت‌ها باید دقت کنند که دیکشنری‌های Mutable را تغییر ندهند مگر اینکه قصدشان این باشد.
        updated = replace(self, **changes)
        updated.touch()
        return updated

    def touch(self) -> None:
        """زمان به‌روزرسانی را آپدیت می‌کند."""
        self.updated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # --- Dictionary Compatibility (برای سازگاری با کدهای قدیمی یا LangGraph ساده) ---
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)
        self.touch()

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
    
    def update(self, mapping: Dict[str, Any]) -> None:
        for k, v in mapping.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.touch()

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return _json_sanitize(asdict(self))

    def to_json(self, ensure_ascii: bool = False, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        # فیلتر کردن کلیدهای اضافی که در کلاس تعریف نشده‌اند (برای جلوگیری از خطای init)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    @classmethod
    def from_json(cls, json_str: str) -> "WorkflowState":
        return cls.from_dict(json.loads(json_str))


# -------------------------
# 4. Utility Functions
# -------------------------

def _json_sanitize(obj: Any) -> Any:
    """تبدیل اشیاء به فرمت قابل سریالایز JSON به صورت بازگشتی."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return _json_sanitize(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]
    # Fallback for unknown objects
    return str(obj)
