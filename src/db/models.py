# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal


QuestionType = Literal["numeric", "categorical", "likert", "text", "date", "json"]
ValueType = Literal["numeric", "text", "date", "json"]


@dataclass(frozen=True)
class Questionnaire:
    questionnaire_id: str
    name: str
    version: str


@dataclass(frozen=True)
class QuestionDef:
    question_id: str
    questionnaire_id: str
    column_name: str
    question_text: str
    type: QuestionType
    allowed_values: Optional[Dict[str, Any]] = None
    missing_rules: Optional[Dict[str, Any]] = None
    privacy_level: str = "normal"
    order_index: int = 0
    is_active: bool = True


@dataclass(frozen=True)
class ResponseHeader:
    response_id: str
    questionnaire_id: str
    submitted_at: Optional[str] = None
    respondent_id: Optional[str] = None


@dataclass(frozen=True)
class ResponseValue:
    value_id: str
    response_id: str
    question_id: str
    value_type: ValueType
    value_text: Optional[str] = None
    value_num: Optional[float] = None
    value_date: Optional[str] = None
    value_json: Optional[str] = None


@dataclass(frozen=True)
class AnalysisRun:
    run_id: str
    user_question: str
    questionnaire_id: Optional[str] = None
    mapped_columns: Optional[Dict[str, Any]] = None
    analysis_plan: Optional[Dict[str, Any]] = None
    code: Optional[str] = None
    execution_status: Optional[str] = None
    result_artifacts: Optional[Dict[str, Any]] = None
    final_report: Optional[str] = None
