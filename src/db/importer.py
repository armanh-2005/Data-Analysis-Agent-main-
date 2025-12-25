# src/db/importer.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
from uuid import UUID, uuid4, uuid5

import pandas as pd

from .connection import connect
from .repository import SQLiteRepository, hash_respondent_id
from .models import Questionnaire, QuestionDef
from src.app.errors import ImporterError


_UUID_NAMESPACE = UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # stable namespace


@dataclass(frozen=True)
class ImportResult:
    questionnaire_id: str
    inserted_responses: int
    inserted_values: int
    skipped_rows: int
    registered_questions: int


class QuestionnaireImporter:
    def __init__(
        self,
        db_path: str,
        respondent_id_salt: str,
        default_privacy_level: str = "normal",
        meta_columns: Sequence[str] = ("respondent_id", "submitted_at"),
    ):
        self.db_path = db_path
        self.repo = SQLiteRepository(db_path)
        self.respondent_id_salt = respondent_id_salt
        self.default_privacy_level = default_privacy_level
        self.meta_columns = set(meta_columns)

    def import_csv(
        self,
        file_path: str,
        questionnaire_name: str,
        version: str,
        questionnaire_id: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> ImportResult:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
        except Exception as e:
            raise ImporterError(f"Failed to read CSV: {e}") from e

        return self._import_dataframe(
            df=df,
            questionnaire_name=questionnaire_name,
            version=version,
            questionnaire_id=questionnaire_id,
            source_hint=str(file_path),
        )

    def import_excel(
        self,
        file_path: str,
        questionnaire_name: str,
        version: str,
        sheet_name: Optional[str] = None,
        questionnaire_id: Optional[str] = None,
    ) -> ImportResult:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            raise ImporterError(f"Failed to read Excel: {e}") from e

        return self._import_dataframe(
            df=df,
            questionnaire_name=questionnaire_name,
            version=version,
            questionnaire_id=questionnaire_id,
            source_hint=f"{file_path}#{sheet_name or 'default'}",
        )

    def _import_dataframe(
        self,
        df: pd.DataFrame,
        questionnaire_name: str,
        version: str,
        questionnaire_id: Optional[str],
        source_hint: str,
    ) -> ImportResult:
        if df is None or df.empty:
            raise ImporterError("Input dataset is empty.")

        # Normalize column names to stable keys.
        df = df.copy()
        df.columns = [self._normalize_column_name(c) for c in df.columns]

        # Ensure questionnaire exists (or create).
        qid = questionnaire_id or self._get_or_create_questionnaire_id(questionnaire_name, version)
        self.repo.upsert_questionnaire(Questionnaire(questionnaire_id=qid, name=questionnaire_name, version=version))

        # --- تغییر ۱: پاکسازی داده‌های قبلی ---
        self.repo.clear_responses(qid)

        # Register/Update schema for non-meta columns.
        question_columns = [c for c in df.columns if c not in self.meta_columns]
        registered = self._register_schema(questionnaire_id=qid, df=df, question_columns=question_columns)

        # --- تغییر ۲: آماده‌سازی داده‌ها برای درج گروهی (Batch Insert) ---
        
        # تبدیل NaN های پانداز به None پایتون (چون SQLite با NaN مشکل دارد)
        # این کار برای کل دیتافریم انجام می‌شود
        records = df.where(pd.notnull(df), None).to_dict(orient='records')
        
        # فراخوانی متد سریع در Repository
        inserted_responses = self.repo.insert_responses_batch(
            questionnaire_id=qid,
            rows_data=records,
            respondent_id_salt=self.respondent_id_salt
        )
        
        # محاسبه تعداد مقادیر (تقریبی)
        inserted_values = df[question_columns].count().sum()

        return ImportResult(
            questionnaire_id=qid,
            inserted_responses=inserted_responses,
            inserted_values=int(inserted_values),
            skipped_rows=0, # در حالت Batch همه‌چیز با هم درج می‌شود
            registered_questions=registered,
        )

    def _register_schema(self, questionnaire_id: str, df: pd.DataFrame, question_columns: Sequence[str]) -> int:
        registered = 0
        order_index = 0

        for col in question_columns:
            order_index += 1
            series = df[col]
            q_type, allowed_values = self._infer_question_type_and_allowed_values(series)
            privacy_level = self._infer_privacy_level(col)

            question_id = self._stable_question_id(questionnaire_id, col)
            qd = QuestionDef(
                question_id=question_id,
                questionnaire_id=questionnaire_id,
                column_name=col,
                question_text=col,
                type=q_type,
                allowed_values=allowed_values,
                missing_rules={"allow_missing": True},
                privacy_level=privacy_level,
                order_index=order_index,
                is_active=True,
            )
            self.repo.upsert_question_def(qd)
            registered += 1

        return registered

    def _infer_question_type_and_allowed_values(self, s: pd.Series) -> Tuple[str, Optional[Dict[str, Any]]]:
        non_null = s.dropna()
        if non_null.empty:
            return "text", None

        try:
            numeric = pd.to_numeric(non_null, errors="coerce")
            numeric_ratio = float(numeric.notna().mean())
            if numeric_ratio >= 0.95:
                uniq = sorted(set(numeric.dropna().astype(float).tolist()))
                if len(uniq) <= 7 and all(float(int(x)) == float(x) for x in uniq):
                    return "likert", {"values": uniq}
                return "numeric", None
        except Exception:
            pass

        try:
            dt = pd.to_datetime(non_null, errors="coerce", utc=False)
            dt_ratio = float(dt.notna().mean())
            if dt_ratio >= 0.95:
                return "date", None
        except Exception:
            pass

        uniq_count = int(non_null.astype(str).nunique(dropna=True))
        n = int(len(non_null))
        if uniq_count <= 50 and uniq_count / max(n, 1) <= 0.2:
            top_values = non_null.astype(str).value_counts(dropna=True).head(50).index.tolist()
            return "categorical", {"values": top_values}

        return "text", None

    def _infer_privacy_level(self, column_name: str) -> str:
        col = column_name.lower()
        sensitive_patterns = [
            r"\bname\b", r"\bfirst_?name\b", r"\blast_?name\b",
            r"\bemail\b", r"\bphone\b", r"\bmobile\b",
            r"\baddress\b", r"\bssn\b", r"\bnational_?id\b",
            r"\bpassport\b", r"\bbirth\b", r"\bdob\b",
        ]
        for pat in sensitive_patterns:
            if re.search(pat, col):
                return "sensitive"
        return self.default_privacy_level

    def _stable_question_id(self, questionnaire_id: str, column_name: str) -> str:
        key = f"{questionnaire_id}:{column_name}"
        return str(uuid5(_UUID_NAMESPACE, key))

    def _get_or_create_questionnaire_id(self, name: str, version: str) -> str:
        conn = connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT questionnaire_id FROM questionnaires WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
            if row:
                return str(row["questionnaire_id"])

            qid = str(uuid4())
            conn.execute(
                "INSERT INTO questionnaires(questionnaire_id, name, version) VALUES (?, ?, ?)",
                (qid, name, version),
            )
            conn.commit()
            return qid
        finally:
            conn.close()

    def _normalize_column_name(self, raw: Any) -> str:
        """
        Unicode-safe column normalization.
        Keeps Persian/Arabic letters and other word characters.
        """
        s = str(raw).strip()
        s = s.replace("\u200c", " ")  # remove ZWNJ artifacts
        s = s.lower()
        s = re.sub(r"\s+", "_", s, flags=re.UNICODE)
        # Keep unicode word chars + underscore; drop punctuation.
        s = re.sub(r"[^\w_]+", "", s, flags=re.UNICODE)
        s = re.sub(r"_+", "_", s, flags=re.UNICODE).strip("_")
        return s or "col"
