# repository.py
from __future__ import annotations

import json
import hashlib
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

import pandas as pd

from .connection import connect
from .models import Questionnaire, QuestionDef, AnalysisRun


def _now_iso_sqlite() -> str:
    # SQLite will fill defaults, but sometimes we want explicit values.
    import datetime as _dt
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def hash_respondent_id(raw_id: str, salt: str) -> str:
    # Hash respondent IDs to avoid storing direct identifiers.
    h = hashlib.sha256()
    h.update((salt + "::" + raw_id).encode("utf-8"))
    return h.hexdigest()


class SQLiteRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def init_schema_from_sql(self, schema_sql: str) -> None:
        # Execute schema SQL in a single transaction.
        conn = connect(self.db_path)
        try:
            conn.executescript(schema_sql)
            conn.commit()
        finally:
            conn.close()

    # -------------------------
    # Questionnaire + schema
    # -------------------------
    def upsert_questionnaire(self, q: Questionnaire) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO questionnaires(questionnaire_id, name, version)
                VALUES (?, ?, ?)
                ON CONFLICT(questionnaire_id) DO UPDATE SET
                  name=excluded.name,
                  version=excluded.version
                """,
                (q.questionnaire_id, q.name, q.version),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_question_def(self, qd: QuestionDef) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO questionnaire_schema(
                  question_id, questionnaire_id, column_name, question_text,
                  type, allowed_values, missing_rules, privacy_level,
                  order_index, is_active, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(question_id) DO UPDATE SET
                  questionnaire_id=excluded.questionnaire_id,
                  column_name=excluded.column_name,
                  question_text=excluded.question_text,
                  type=excluded.type,
                  allowed_values=excluded.allowed_values,
                  missing_rules=excluded.missing_rules,
                  privacy_level=excluded.privacy_level,
                  order_index=excluded.order_index,
                  is_active=excluded.is_active,
                  updated_at=datetime('now')
                """,
                (
                    qd.question_id,
                    qd.questionnaire_id,
                    qd.column_name,
                    qd.question_text,
                    qd.type,
                    json.dumps(qd.allowed_values, ensure_ascii=False) if qd.allowed_values is not None else None,
                    json.dumps(qd.missing_rules, ensure_ascii=False) if qd.missing_rules is not None else None,
                    qd.privacy_level,
                    qd.order_index,
                    1 if qd.is_active else 0,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_schema(self, questionnaire_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT question_id, questionnaire_id, column_name, question_text, type,
                       allowed_values, missing_rules, privacy_level, order_index, is_active
                FROM questionnaire_schema
                WHERE questionnaire_id = ?
                """
                + (" AND is_active = 1" if active_only else "")
                + " ORDER BY order_index ASC",
                (questionnaire_id,),
            ).fetchall()

            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "question_id": r["question_id"],
                        "questionnaire_id": r["questionnaire_id"],
                        "column_name": r["column_name"],
                        "question_text": r["question_text"],
                        "type": r["type"],
                        "allowed_values": json.loads(r["allowed_values"]) if r["allowed_values"] else None,
                        "missing_rules": json.loads(r["missing_rules"]) if r["missing_rules"] else None,
                        "privacy_level": r["privacy_level"],
                        "order_index": r["order_index"],
                        "is_active": bool(r["is_active"]),
                    }
                )
            return out
        finally:
            conn.close()

    def resolve_question_ids(self, questionnaire_id: str, column_names: Sequence[str]) -> Dict[str, str]:
        # Map column_name -> question_id
        if not column_names:
            return {}

        placeholders = ",".join(["?"] * len(column_names))
        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                f"""
                SELECT column_name, question_id
                FROM questionnaire_schema
                WHERE questionnaire_id = ?
                  AND column_name IN ({placeholders})
                """,
                (questionnaire_id, *column_names),
            ).fetchall()
            return {r["column_name"]: r["question_id"] for r in rows}
        finally:
            conn.close()

    # -------------------------
    # Responses (EAV)
    # -------------------------
    def insert_response(
        self,
        questionnaire_id: str,
        answers_by_column: Dict[str, Any],
        respondent_id_hash: Optional[str] = None,
        submitted_at: Optional[str] = None,
    ) -> str:
        # Insert response header + values. Values are stored with a best-effort type mapping.
        response_id = str(uuid4())
        submitted_at = submitted_at or _now_iso_sqlite()

        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO responses(response_id, questionnaire_id, submitted_at, respondent_id)
                VALUES (?, ?, ?, ?)
                """,
                (response_id, questionnaire_id, submitted_at, respondent_id_hash),
            )

            # Resolve question IDs
            col_names = list(answers_by_column.keys())
            mapping = self.resolve_question_ids(questionnaire_id, col_names)

            for col, val in answers_by_column.items():
                question_id = mapping.get(col)
                if question_id is None:
                    # Unknown question column_name -> skip (or raise, depending on your policy)
                    continue

                value_id = str(uuid4())
                value_type, vt, vn, vd, vj = self._serialize_value(val)

                conn.execute(
                    """
                    INSERT INTO response_values(
                      value_id, response_id, question_id,
                      value_type, value_text, value_num, value_date, value_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(response_id, question_id) DO UPDATE SET
                      value_type=excluded.value_type,
                      value_text=excluded.value_text,
                      value_num=excluded.value_num,
                      value_date=excluded.value_date,
                      value_json=excluded.value_json
                    """,
                    (value_id, response_id, question_id, value_type, vt, vn, vd, vj),
                )

            conn.commit()
            return response_id
        finally:
            conn.close()

    def _serialize_value(self, val: Any) -> Tuple[str, Optional[str], Optional[float], Optional[str], Optional[str]]:
        # Returns: (value_type, value_text, value_num, value_date, value_json)
        if val is None:
            return ("text", None, None, None, None)

        # Numeric
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return ("numeric", None, float(val), None, None)

        # Date/datetime represented as ISO-like strings (you can strengthen this later)
        if isinstance(val, str):
            s = val.strip()
            # Simple heuristic: if looks like YYYY-MM-DD or YYYY-MM-DDTHH:MM
            if len(s) >= 10 and s[4:5] == "-" and s[7:8] == "-":
                return ("date", None, None, s, None)
            return ("text", s, None, None, None)

        # Dict/list -> JSON
        if isinstance(val, (dict, list)):
            return ("json", None, None, None, json.dumps(val, ensure_ascii=False))

        # Fallback: stringify
        return ("text", str(val), None, None, None)

    # -------------------------
    # Fetch for analysis (long + wide)
    # -------------------------
    def fetch_long_dataframe(
        self,
        questionnaire_id: str,
        column_names: Sequence[str],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        # Returns long format: one row per (response_id, column_name)
        mapping = self.resolve_question_ids(questionnaire_id, column_names)
        question_ids = [mapping[c] for c in column_names if c in mapping]
        if not question_ids:
            return pd.DataFrame(columns=["response_id", "submitted_at", "respondent_id", "column_name", "value"])

        placeholders = ",".join(["?"] * len(question_ids))
        params: List[Any] = [questionnaire_id, *question_ids]

        where_time = ""
        if date_from:
            where_time += " AND r.submitted_at >= ?"
            params.append(date_from)
        if date_to:
            where_time += " AND r.submitted_at < ?"
            params.append(date_to)

        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                f"""
                SELECT
                  r.response_id,
                  r.submitted_at,
                  r.respondent_id,
                  s.column_name,
                  v.value_type,
                  v.value_text,
                  v.value_num,
                  v.value_date,
                  v.value_json
                FROM responses r
                JOIN response_values v ON v.response_id = r.response_id
                JOIN questionnaire_schema s ON s.question_id = v.question_id
                WHERE r.questionnaire_id = ?
                  AND v.question_id IN ({placeholders})
                  {where_time}
                ORDER BY r.submitted_at ASC
                """,
                params,
            ).fetchall()

            data = []
            for r in rows:
                value = (
                    r["value_num"]
                    if r["value_type"] == "numeric"
                    else r["value_date"]
                    if r["value_type"] == "date"
                    else json.loads(r["value_json"])
                    if r["value_type"] == "json" and r["value_json"]
                    else r["value_text"]
                )
                data.append(
                    {
                        "response_id": r["response_id"],
                        "submitted_at": r["submitted_at"],
                        "respondent_id": r["respondent_id"],
                        "column_name": r["column_name"],
                        "value": value,
                    }
                )
            return pd.DataFrame(data)
        finally:
            conn.close()

    def fetch_wide_dataframe(
        self,
        questionnaire_id: str,
        column_names: Sequence[str],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        # Wide format via pivot in pandas (safer than dynamic SQL pivot).
        long_df = self.fetch_long_dataframe(
            questionnaire_id=questionnaire_id,
            column_names=column_names,
            date_from=date_from,
            date_to=date_to,
        )
        if long_df.empty:
            return long_df

        wide = long_df.pivot_table(
            index=["response_id", "submitted_at", "respondent_id"],
            columns="column_name",
            values="value",
            aggfunc="first",
        ).reset_index()

        # Make columns flat
        wide.columns.name = None
        return wide

    # -------------------------
    # Analysis runs + notes_state
    # -------------------------
    def create_analysis_run(self, run: AnalysisRun) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO analysis_runs(run_id, questionnaire_id, user_question, mapped_columns, analysis_plan,
                                          code, execution_status, result_artifacts, final_report)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                  questionnaire_id=excluded.questionnaire_id,
                  user_question=excluded.user_question,
                  mapped_columns=excluded.mapped_columns,
                  analysis_plan=excluded.analysis_plan,
                  code=excluded.code,
                  execution_status=excluded.execution_status,
                  result_artifacts=excluded.result_artifacts,
                  final_report=excluded.final_report
                """,
                (
                    run.run_id,
                    run.questionnaire_id,
                    run.user_question,
                    json.dumps(run.mapped_columns, ensure_ascii=False) if run.mapped_columns else None,
                    json.dumps(run.analysis_plan, ensure_ascii=False) if run.analysis_plan else None,
                    run.code,
                    run.execution_status,
                    json.dumps(run.result_artifacts, ensure_ascii=False) if run.result_artifacts else None,
                    run.final_report,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def patch_analysis_run(self, run_id: str, **fields: Any) -> None:
        # Update any subset of columns safely.
        allowed = {
            "questionnaire_id",
            "user_question",
            "mapped_columns",
            "analysis_plan",
            "code",
            "execution_status",
            "result_artifacts",
            "final_report",
        }
        set_parts = []
        params: List[Any] = []

        for k, v in fields.items():
            if k not in allowed:
                continue
            set_parts.append(f"{k} = ?")
            if k in {"mapped_columns", "analysis_plan", "result_artifacts"} and v is not None:
                params.append(json.dumps(v, ensure_ascii=False))
            else:
                params.append(v)

        if not set_parts:
            return

        params.append(run_id)
        conn = connect(self.db_path)
        try:
            conn.execute(
                f"UPDATE analysis_runs SET {', '.join(set_parts)} WHERE run_id = ?",
                params,
            )
            conn.commit()
        finally:
            conn.close()

    def save_notes_state(self, run_id: str, state: Dict[str, Any]) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO notes_state(run_id, state_json, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(run_id) DO UPDATE SET
                  state_json=excluded.state_json,
                  updated_at=datetime('now')
                """,
                (run_id, json.dumps(state, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    def load_notes_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT state_json FROM notes_state WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if not row:
                return None
            return json.loads(row["state_json"])
        finally:
            conn.close()
