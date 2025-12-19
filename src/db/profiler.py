# src/db/profiler.py
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .connection import connect


@dataclass(frozen=True)
class QuestionProfile:
    # Basic question metadata
    question_id: str
    column_name: str
    question_text: str
    q_type: str  # numeric/categorical/likert/text/date/json
    privacy_level: str
    allowed_values: Optional[Dict[str, Any]]

    # Completeness
    n_total: int
    n_answered: int
    n_missing: int
    missing_rate: float

    # Shape/summary
    unique_count: Optional[int] = None
    top_values: Optional[List[Dict[str, Any]]] = None  # [{value,count,share}]
    numeric: Optional[Dict[str, Any]] = None  # {mean,std,min,max,median,q1,q3}


@dataclass(frozen=True)
class ProfileSummary:
    questionnaire_id: str
    date_from: Optional[str]
    date_to: Optional[str]
    sample_size: Optional[int]

    n_total_responses: int
    questions: List[QuestionProfile]


class ProfilerError(Exception):
    pass


class SQLiteEAVProfiler:
    """
    Profiles an EAV-stored questionnaire in SQLite.

    Key design goals:
      - JSON-serializable output (safe to store in notes_state / analysis_runs).
      - Works with dynamic questionnaires (schema registry).
      - Optional sampling to avoid pulling huge tables for quick planning.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def profile(
        self,
        questionnaire_id: str,
        column_names: Optional[Sequence[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sample_size: Optional[int] = None,
        top_k: int = 20,
        include_sensitive_top_values: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns a JSON-serializable profile dict.

        Args:
            questionnaire_id: Target questionnaire.
            column_names: If provided, profile only these columns (by schema column_name).
            date_from/date_to: Filter on responses.submitted_at (string compare; keep ISO-like).
            sample_size: If provided, profile on most-recent N responses (within date filters).
            top_k: For categorical/likert/text: number of top values to return.
            include_sensitive_top_values: If False, top_values are suppressed for sensitive questions.
        """
        schema = self._load_schema(questionnaire_id, column_names=column_names)
        if not schema:
            raise ProfilerError("No schema found for the given questionnaire/columns.")

        # Count total responses (respecting date filters + sampling).
        response_ids = self._select_response_ids(
            questionnaire_id=questionnaire_id,
            date_from=date_from,
            date_to=date_to,
            sample_size=sample_size,
        )
        n_total = len(response_ids)

        # If there are no responses, return an empty-ish profile but with question metadata.
        if n_total == 0:
            summary = ProfileSummary(
                questionnaire_id=questionnaire_id,
                date_from=date_from,
                date_to=date_to,
                sample_size=sample_size,
                n_total_responses=0,
                questions=[
                    QuestionProfile(
                        question_id=q["question_id"],
                        column_name=q["column_name"],
                        question_text=q["question_text"],
                        q_type=q["type"],
                        privacy_level=q["privacy_level"],
                        allowed_values=q["allowed_values"],
                        n_total=0,
                        n_answered=0,
                        n_missing=0,
                        missing_rate=0.0,
                        unique_count=0,
                        top_values=None,
                        numeric=None,
                    )
                    for q in schema
                ],
            )
            return self._to_json_dict(summary)

        # Fetch all values for selected questions for selected responses (single query).
        values_rows = self._fetch_values_for_responses(
            response_ids=response_ids,
            question_ids=[q["question_id"] for q in schema],
        )

        # Group values by question_id.
        by_qid: Dict[str, List[Dict[str, Any]]] = {}
        for r in values_rows:
            by_qid.setdefault(r["question_id"], []).append(r)

        question_profiles: List[QuestionProfile] = []
        for q in schema:
            qid = q["question_id"]
            qtype = q["type"]
            privacy_level = q["privacy_level"]

            rows = by_qid.get(qid, [])
            # Note: Ideally each response has exactly one row per question (UNIQUE(response_id, question_id)),
            # but we handle missing rows too.
            present_by_response = {rr["response_id"]: rr for rr in rows}

            # Build per-response values so missingness is correctly computed.
            typed_values: List[Any] = []
            missing_count = 0

            for rid in response_ids:
                rr = present_by_response.get(rid)
                if rr is None:
                    missing_count += 1
                    continue

                val, is_missing = self._row_to_value(rr)
                if is_missing:
                    missing_count += 1
                else:
                    typed_values.append(val)

            n_answered = len(typed_values)
            n_missing = missing_count
            missing_rate = (n_missing / n_total) if n_total else 0.0

            # Summaries by type.
            unique_count: Optional[int] = None
            top_values: Optional[List[Dict[str, Any]]] = None
            numeric: Optional[Dict[str, Any]] = None

            if qtype in {"numeric"}:
                numeric = self._numeric_summary(typed_values)
                unique_count = numeric.get("unique_count") if numeric else None

            elif qtype in {"likert"}:
                # Likert might be stored numeric; we treat it as categorical distribution too.
                unique_count, top_values = self._top_values_summary(
                    typed_values=typed_values,
                    top_k=top_k,
                    suppress_top=(privacy_level == "sensitive" and not include_sensitive_top_values),
                )

            elif qtype in {"categorical", "text"}:
                # For text, top values can be noisy; still useful for quick planning (optionally suppress).
                unique_count, top_values = self._top_values_summary(
                    typed_values=typed_values,
                    top_k=top_k,
                    suppress_top=(privacy_level == "sensitive" and not include_sensitive_top_values),
                )

            elif qtype in {"date"}:
                # Dates: we can provide min/max as strings (ISO-like), plus unique_count.
                unique_count = len(set(str(v) for v in typed_values))
                if typed_values:
                    svals = sorted(str(v) for v in typed_values)
                    top_values = None
                    numeric = {
                        "min": svals[0],
                        "max": svals[-1],
                        "unique_count": unique_count,
                    }

            elif qtype in {"json"}:
                # JSON answers: report only answered count + unique_count by stringified value (best-effort).
                unique_count = len(set(json.dumps(v, ensure_ascii=False, sort_keys=True) for v in typed_values))
                top_values = None
                numeric = None

            qp = QuestionProfile(
                question_id=qid,
                column_name=q["column_name"],
                question_text=q["question_text"],
                q_type=qtype,
                privacy_level=privacy_level,
                allowed_values=q["allowed_values"],
                n_total=n_total,
                n_answered=n_answered,
                n_missing=n_missing,
                missing_rate=missing_rate,
                unique_count=unique_count,
                top_values=top_values,
                numeric=numeric,
            )
            question_profiles.append(qp)

        summary = ProfileSummary(
            questionnaire_id=questionnaire_id,
            date_from=date_from,
            date_to=date_to,
            sample_size=sample_size,
            n_total_responses=n_total,
            questions=question_profiles,
        )
        return self._to_json_dict(summary)

    # -------------------------
    # DB access
    # -------------------------

    def _load_schema(self, questionnaire_id: str, column_names: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
        conn = connect(self.db_path)
        try:
            params: List[Any] = [questionnaire_id]
            where_cols = ""
            if column_names:
                placeholders = ",".join(["?"] * len(column_names))
                where_cols = f" AND column_name IN ({placeholders})"
                params.extend(list(column_names))

            rows = conn.execute(
                f"""
                SELECT question_id, column_name, question_text, type, allowed_values, privacy_level
                FROM questionnaire_schema
                WHERE questionnaire_id = ?
                  AND is_active = 1
                  {where_cols}
                ORDER BY order_index ASC
                """,
                params,
            ).fetchall()

            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "question_id": r["question_id"],
                        "column_name": r["column_name"],
                        "question_text": r["question_text"],
                        "type": r["type"],
                        "allowed_values": json.loads(r["allowed_values"]) if r["allowed_values"] else None,
                        "privacy_level": r["privacy_level"],
                    }
                )
            return out
        finally:
            conn.close()

    def _select_response_ids(
        self,
        questionnaire_id: str,
        date_from: Optional[str],
        date_to: Optional[str],
        sample_size: Optional[int],
    ) -> List[str]:
        conn = connect(self.db_path)
        try:
            where_time = ""
            params: List[Any] = [questionnaire_id]
            if date_from:
                where_time += " AND submitted_at >= ?"
                params.append(date_from)
            if date_to:
                where_time += " AND submitted_at < ?"
                params.append(date_to)

            limit_clause = ""
            if sample_size is not None and sample_size > 0:
                limit_clause = f" LIMIT {int(sample_size)}"

            rows = conn.execute(
                f"""
                SELECT response_id
                FROM responses
                WHERE questionnaire_id = ?
                  {where_time}
                ORDER BY submitted_at DESC
                {limit_clause}
                """,
                params,
            ).fetchall()
            return [str(r["response_id"]) for r in rows]
        finally:
            conn.close()

    def _fetch_values_for_responses(self, response_ids: Sequence[str], question_ids: Sequence[str]) -> List[Dict[str, Any]]:
        if not response_ids or not question_ids:
            return []

        rid_ph = ",".join(["?"] * len(response_ids))
        qid_ph = ",".join(["?"] * len(question_ids))

        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                f"""
                SELECT
                  v.response_id,
                  v.question_id,
                  v.value_type,
                  v.value_text,
                  v.value_num,
                  v.value_date,
                  v.value_json
                FROM response_values v
                WHERE v.response_id IN ({rid_ph})
                  AND v.question_id IN ({qid_ph})
                """,
                [*response_ids, *question_ids],
            ).fetchall()

            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "response_id": str(r["response_id"]),
                        "question_id": str(r["question_id"]),
                        "value_type": r["value_type"],
                        "value_text": r["value_text"],
                        "value_num": r["value_num"],
                        "value_date": r["value_date"],
                        "value_json": r["value_json"],
                    }
                )
            return out
        finally:
            conn.close()

    # -------------------------
    # Value decoding + missingness
    # -------------------------

    def _row_to_value(self, r: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Returns (value, is_missing).
        Missing logic is conservative:
          - NULL -> missing
          - empty/whitespace string -> missing
        """
        vtype = r.get("value_type")

        if vtype == "numeric":
            vn = r.get("value_num")
            if vn is None or (isinstance(vn, float) and math.isnan(vn)):
                return None, True
            return float(vn), False

        if vtype == "date":
            vd = r.get("value_date")
            if vd is None:
                return None, True
            s = str(vd).strip()
            return (s, s == "")

        if vtype == "json":
            vj = r.get("value_json")
            if vj is None:
                return None, True
            s = str(vj).strip()
            if s == "":
                return None, True
            try:
                return json.loads(s), False
            except Exception:
                # If JSON is corrupted, treat as missing to avoid misleading distributions.
                return None, True

        # text (default)
        vt = r.get("value_text")
        if vt is None:
            return None, True
        s = str(vt).strip()
        return (s, s == "")

    # -------------------------
    # Summaries
    # -------------------------

    def _numeric_summary(self, values: List[Any]) -> Optional[Dict[str, Any]]:
        nums: List[float] = []
        for v in values:
            try:
                if v is None:
                    continue
                nums.append(float(v))
            except Exception:
                continue

        if not nums:
            return None

        nums_sorted = sorted(nums)
        n = len(nums_sorted)

        mean = sum(nums_sorted) / n
        std = statistics.pstdev(nums_sorted) if n >= 2 else 0.0

        # Median and quartiles (simple, deterministic method)
        median = statistics.median(nums_sorted)
        q1 = statistics.median(nums_sorted[: n // 2]) if n >= 4 else None
        q3 = statistics.median(nums_sorted[(n + 1) // 2 :]) if n >= 4 else None

        return {
            "n": n,
            "mean": mean,
            "std": std,
            "min": nums_sorted[0],
            "max": nums_sorted[-1],
            "median": median,
            "q1": q1,
            "q3": q3,
            "unique_count": len(set(nums_sorted)),
        }

    def _top_values_summary(
        self,
        typed_values: List[Any],
        top_k: int,
        suppress_top: bool,
    ) -> Tuple[int, Optional[List[Dict[str, Any]]]]:
        svals = [str(v) for v in typed_values if v is not None]
        unique_count = len(set(svals))

        if suppress_top or not svals:
            return unique_count, None

        counts: Dict[str, int] = {}
        for s in svals:
            counts[s] = counts.get(s, 0) + 1

        total = len(svals)
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: max(int(top_k), 1)]

        top_values = [
            {"value": k, "count": c, "share": (c / total) if total else 0.0}
            for k, c in items
        ]
        return unique_count, top_values

    # -------------------------
    # Serialization
    # -------------------------

    def _to_json_dict(self, summary: ProfileSummary) -> Dict[str, Any]:
        # Convert dataclasses to plain JSON-serializable dict.
        d = asdict(summary)
        return d
