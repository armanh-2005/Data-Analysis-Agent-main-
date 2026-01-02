# src/db/profiler.py
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .connection import connect


@dataclass(frozen=True)
class QuestionProfile:
    question_id: str
    column_name: str
    question_text: str
    q_type: str
    privacy_level: str
    allowed_values: Optional[Dict[str, Any]]
    n_total: int
    n_answered: int
    n_missing: int
    missing_rate: float
    unique_count: Optional[int] = None
    top_values: Optional[List[Dict[str, Any]]] = None
    numeric: Optional[Dict[str, Any]] = None
    sample_values: Optional[List[Any]] = None


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
    def __init__(self, db_path: str):
        self.db_path = db_path

    def profile(
        self,
        questionnaire_id: str,
        column_names: Optional[Sequence[str]] = None, # وقتی column_mapper اجرا میشه، لیست ستون‌ها اینجا میاد
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sample_size: Optional[int] = None,
        top_k: int = 20,
        include_sensitive_top_values: bool = False,
    ) -> Dict[str, Any]:
        
        # 1. Load Schema (اینجا خودش فیلتر میشه روی column_names)
        schema = self._load_schema(questionnaire_id, column_names=column_names)
        if not schema:
            # اگر هیچ ستونی انتخاب نشده باشه یا آیدی غلط باشه
            if column_names: 
                # ممکنه column_mapper چیزی انتخاب کرده که تو دیتابیس نیست، پس ارور نمیدیم، دیکشنری خالی میدیم
                return self._create_empty_profile(questionnaire_id, [], date_from, date_to, sample_size)
            raise ProfilerError(f"No schema found for questionnaire_id: {questionnaire_id}")

        # 2. Select Response IDs
        response_ids = self._select_response_ids(
            questionnaire_id=questionnaire_id,
            date_from=date_from,
            date_to=date_to,
            sample_size=sample_size,
        )
        n_total = len(response_ids)

        # 3. Handle Empty Dataset
        if n_total == 0:
            return self._create_empty_profile(
                questionnaire_id, schema, date_from, date_to, sample_size
            )

        # 4. Fetch Data
        values_rows = self._fetch_values_for_responses(
            response_ids=response_ids,
            question_ids=[q["question_id"] for q in schema],
        )

        by_qid: Dict[str, List[Dict[str, Any]]] = {}
        for r in values_rows:
            by_qid.setdefault(r["question_id"], []).append(r)

        # 5. Calculate Statistics
        question_profiles: List[QuestionProfile] = []
        
        for q in schema:
            qid = q["question_id"]
            qtype = q["type"]
            privacy_level = q["privacy_level"]
            rows = by_qid.get(qid, [])
            present_by_response = {rr["response_id"]: rr for rr in rows}

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
            missing_rate = (n_missing / n_total) if n_total > 0 else 0.0

            unique_count: Optional[int] = None
            top_values: Optional[List[Dict[str, Any]]] = None
            numeric: Optional[Dict[str, Any]] = None
            sample_values: Optional[List[Any]] = None # متغیر جدید

            if qtype in {"numeric", "integer", "float"}:
                numeric = self._numeric_summary(typed_values)
                unique_count = numeric.get("unique_count") if numeric else 0

            elif qtype in {"likert", "categorical", "text", "boolean"}:
                # محاسبه آمار برای نمودارها
                unique_count, top_values = self._top_values_summary(
                    typed_values=typed_values,
                    top_k=top_k,
                    suppress_top=(privacy_level == "sensitive" and not include_sensitive_top_values),
                )
                
                # --- بخش جدید: استخراج مقادیر نمونه برای ایجنت ---
                # فقط لیستی از استرینگ‌های یونیک (بدون تعداد) تا ایجنت بفهمه چی توشه
                if typed_values:
                    # تبدیل همه به استرینگ و حذف تکراری‌ها
                    svals = list(set(str(v) for v in typed_values if v is not None))
                    # مرتب‌سازی و برداشتن 20 تای اول (برای جلوگیری از شلوغی بیش از حد)
                    sample_values = sorted(svals)[:20]
                else:
                    sample_values = []

            elif qtype in {"date", "datetime"}:
                svals = [str(v) for v in typed_values]
                unique_count = len(set(svals))
                if svals:
                    sorted_vals = sorted(svals)
                    numeric = {
                        "min": sorted_vals[0],
                        "max": sorted_vals[-1],
                        "unique_count": unique_count,
                    }

            elif qtype in {"json"}:
                unique_count = len(set(json.dumps(v, sort_keys=True) for v in typed_values))

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
                missing_rate=round(missing_rate, 4),
                unique_count=unique_count,
                top_values=top_values,
                numeric=numeric,
                sample_values=sample_values, # ارسال به دیتاکلاس
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
    # Helper Methods
    # -------------------------

    def _create_empty_profile(self, questionnaire_id, schema, date_from, date_to, sample_size):
        questions = []
        for q in schema:
            questions.append(QuestionProfile(
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
            ))
        
        summary = ProfileSummary(
            questionnaire_id=questionnaire_id,
            date_from=date_from,
            date_to=date_to,
            sample_size=sample_size,
            n_total_responses=0,
            questions=questions
        )
        return self._to_json_dict(summary)

    def _to_json_dict(self, summary: Any) -> Dict[str, Any]:
        if is_dataclass(summary):
            return asdict(summary)
        return summary

    # -------------------------
    # DB Access Methods
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

            query = f"""
                SELECT question_id, column_name, question_text, type, allowed_values, privacy_level
                FROM questionnaire_schema
                WHERE questionnaire_id = ?
                  AND is_active = 1
                  {where_cols}
                ORDER BY order_index ASC
            """
            
            cursor = conn.cursor()
            rows = cursor.execute(query, params).fetchall()

            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append({
                    "question_id": r[0],
                    "column_name": r[1],
                    "question_text": r[2],
                    "type": r[3],
                    "allowed_values": json.loads(r[4]) if r[4] else None,
                    "privacy_level": r[5],
                })
            return out
        except Exception as e:
            print(f"Error loading schema: {e}")
            return []
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

            query = f"""
                SELECT response_id
                FROM responses
                WHERE questionnaire_id = ?
                  {where_time}
                ORDER BY submitted_at DESC
                {limit_clause}
            """
            
            cursor = conn.cursor()
            rows = cursor.execute(query, params).fetchall()
            return [str(r[0]) for r in rows]
        finally:
            conn.close()

    def _fetch_values_for_responses(self, response_ids: Sequence[str], question_ids: Sequence[str]) -> List[Dict[str, Any]]:
        if not response_ids or not question_ids:
            return []

        rid_ph = ",".join(["?"] * len(response_ids))
        qid_ph = ",".join(["?"] * len(question_ids))

        conn = connect(self.db_path)
        try:
            # FIX: Changed value_numeric to value_num based on your DB schema
            query = f"""
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
            """
            
            params = list(response_ids) + list(question_ids)
            cursor = conn.cursor()
            rows = cursor.execute(query, params).fetchall()

            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append({
                    "response_id": str(r[0]),
                    "question_id": str(r[1]),
                    "value_type": r[2],
                    "value_text": r[3],
                    "value_num": r[4], 
                    "value_date": r[5],
                    "value_json": r[6],
                })
            return out
        finally:
            conn.close()

    # -------------------------
    # Value Decoding
    # -------------------------

    def _row_to_value(self, r: Dict[str, Any]) -> Tuple[Any, bool]:
        vtype = r.get("value_type")

        # Numeric
        if vtype in ("numeric", "float", "integer"):
            vn = r.get("value_num")
            if vn is None:
                return None, True
            try:
                f_val = float(vn)
                if math.isnan(f_val):
                    return None, True
                return f_val, False
            except (ValueError, TypeError):
                return None, True

        # Date
        if vtype == "date":
            vd = r.get("value_date")
            if vd is None:
                return None, True
            s = str(vd).strip()
            return (s, s == "")

        # JSON
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
                return None, True

        # Text
        vt = r.get("value_text")
        if vt is None:
            return None, True
        s = str(vt).strip()
        return (s, s == "")

    # -------------------------
    # Math & Stats Logic
    # -------------------------

    def _numeric_summary(self, values: List[Any]) -> Optional[Dict[str, Any]]:
        nums: List[float] = []
        for v in values:
            try:
                if v is not None:
                    nums.append(float(v))
            except (ValueError, TypeError):
                continue

        if not nums:
            return None

        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        mean_val = sum(nums_sorted) / n
        std_val = statistics.pstdev(nums_sorted) if n > 1 else 0.0
        median_val = statistics.median(nums_sorted)
        
        half_idx = n // 2
        lower_half = nums_sorted[:half_idx]
        upper_half = nums_sorted[half_idx + (1 if n % 2 != 0 else 0):]
        q1 = statistics.median(lower_half) if lower_half else None
        q3 = statistics.median(upper_half) if upper_half else None

        return {
            "n": n,
            "mean": mean_val,
            "std": std_val,
            "min": nums_sorted[0],
            "max": nums_sorted[-1],
            "median": median_val,
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
        sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top_items = sorted_items[: max(int(top_k), 1)]

        top_values_list = [
            {
                "value": k, 
                "count": c, 
                "share": round((c / total), 4) if total > 0 else 0.0
            }
            for k, c in top_items
        ]
        return unique_count, top_values_list