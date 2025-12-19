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
    # Column mapping result produced by ColumnMapperAgent
    column_name: str                 # schema column_name (stable key)
    question_id: str                 # resolved id from schema registry (optional but recommended)
    question_text: str               # human-readable label
    reason: str                      # why it was mapped
    confidence: float                # 0..1
    confidence_label: Confidence     # low/medium/high
    inferred_role: str               # e.g. "target", "group_by", "filter", "time"
    value_constraints: Dict[str, Any]  # e.g. {"allowed_values":[...]} or {"range":[...]}
    privacy_level: str               # normal/sensitive/restricted


class CodeReview(TypedDict, total=False):
    # Produced by CodeReviewerAgent
    approved: bool
    feedback: str
    issues: List[Dict[str, Any]]     # structured issues (optional)
    safety: Dict[str, Any]           # e.g. {"network": False, "fs_write": "limited"}
    score: float                     # 0..1


class ExecutionResult(TypedDict, total=False):
    # Produced by sandbox executor
    status: str                      # "success" | "failed"
    stdout: str
    stderr: str
    results_json: Dict[str, Any]     # numeric tables/metrics/etc.
    artifacts: List[Dict[str, Any]]  # e.g. [{"type":"chart","path":"..."}]
    started_at: str                  # ISO
    finished_at: str                 # ISO
    runtime_seconds: float


class QualityReview(TypedDict, total=False):
    # Produced by QualityReviewAgent
    approved: bool
    feedback: str
    issues: List[Dict[str, Any]]
    score: float                     # 0..1
    sufficiency: Dict[str, Any]      # e.g. {"n":123, "missing_rate":0.12}


# Note: ProfileSummary is returned as a plain dict by SQLiteEAVProfiler.profile()
# so we keep it as Dict[str, Any] here to guarantee JSON serializability.
ProfileSummary = Dict[str, Any]


# -------------------------
# Core State (single object passed around LangGraph)
# -------------------------

@dataclass
class WorkflowState:
    # Identity / request
    run_id: str
    user_question: str

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

    # Memory/state for NoteAgent (assumptions, decisions, versions, errors, etc.)
    notes: Dict[str, Any] = field(default_factory=dict)

    # Internal bookkeeping (optional but handy)
    iteration: Dict[str, int] = field(default_factory=lambda: {"code": 0, "quality": 0})
    created_at: str = field(default_factory=lambda: datetime.utcnow().replace(microsecond=0).isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().replace(microsecond=0).isoformat() + "Z")

    # -------------------------
    # State patching utilities
    # -------------------------

    def touch(self) -> None:
        # Update updated_at timestamp.
        self.updated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def patch(self, **updates: Any) -> "WorkflowState":
        """
        Apply a partial update (node output) to the state.
        This is useful because agents should return patches, not full states.
        """
        for k, v in updates.items():
            if not hasattr(self, k):
                continue
            setattr(self, k, v)
        self.touch()
        return self

    def increment(self, key: str) -> None:
        # Increment named iteration counter (e.g., "code", "quality").
        self.iteration.setdefault(key, 0)
        self.iteration[key] += 1
        self.touch()

    # -------------------------
    # JSON serialization
    # -------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a JSON-serializable dict.
        Ensures dataclasses inside notes or other fields are converted as well.
        """
        raw = asdict(self)
        return _json_sanitize(raw)

    def to_json(self, ensure_ascii: bool = False, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorkflowState":
        """
        Reconstruct the state from a dict (e.g., from notes_state.state_json).
        Unknown keys are ignored to be forward-compatible.
        """
        allowed = {f.name for f in WorkflowState.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in allowed}

        # Ensure required fields exist
        if "run_id" not in filtered or "user_question" not in filtered:
            raise ValueError("State dict must contain 'run_id' and 'user_question'.")

        state = WorkflowState(**filtered)  # type: ignore[arg-type]
        state.touch()
        return state

    @staticmethod
    def from_json(s: str) -> "WorkflowState":
        return WorkflowState.from_dict(json.loads(s))


# -------------------------
# Helpers
# -------------------------

def _json_sanitize(obj: Any) -> Any:
    """
    Convert an arbitrary object into something JSON-serializable.
    - dataclasses -> dict
    - datetime -> ISO string
    - set/tuple -> list
    - fallback -> str
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, datetime):
        return obj.replace(microsecond=0).isoformat() + "Z"

    if is_dataclass(obj):
        return _json_sanitize(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v) for v in obj]

    # Last resort: stringify
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def new_state(run_id: str, user_question: str) -> WorkflowState:
    """
    Convenience factory to create a fresh state.
    """
    return WorkflowState(run_id=run_id, user_question=user_question)
