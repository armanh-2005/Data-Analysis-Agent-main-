import sys
import os
import json
import sqlite3
import io
import contextlib
import traceback
import warnings
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from uuid import uuid4
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

# --- تنظیمات جلوگیری از کرش ---
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# --- تنظیم مسیرها ---
root_dir = Path(__file__).resolve().parents[2]
load_dotenv(root_dir / ".env")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- ماژول‌های پروژه ---
try:
    from src.app.config import Settings
    from src.db.repository import SQLiteRepository
    from src.db.importer import QuestionnaireImporter
    from src.db.profiler import SQLiteEAVProfiler
    from src.workflows.state import WorkflowState
    from src.agents.router_mapper_agent import RouterMapperAgent
    from src.agents.planner_agent import PlannerAgent
    from src.agents.code_writer_agent import CodeWriterAgent
    from src.agents.code_reviewer_agent import CodeReviewerAgent
    from src.agents.quality_review_agent import QualityReviewAgent
    from src.agents.report_writer_agent import ReportWriterAgent
    from src.tools import political, stats, viz
except ImportError as e:
    st.error(f"خطا در بارگذاری ماژول‌ها: {e}")
    st.stop()

# --- تنظیمات اولیه ---
st.set_page_config(page_title="دستیار تحلیل داده", page_icon="📊", layout="wide")
st.markdown("""<style>.stTextInput, .stMarkdown, .stButton { direction: rtl; text-align: right; }</style>""", unsafe_allow_html=True)

@st.cache_resource
def get_settings():
    return Settings.from_env()

settings = get_settings()
os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
if hasattr(settings, 'artifacts_dir'):
    os.makedirs(settings.artifacts_dir, exist_ok=True)

# --- تابع بررسی دیتابیس (جدید) ---
def debug_database_schema(db_path, q_id):
    """بررسی می‌کند آیا واقعاً ستون‌ها در دیتابیس ذخیره شده‌اند؟"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # شمارش سوالات (ستون‌ها) برای این فایل
            cursor.execute("SELECT count(*) FROM questions WHERE questionnaire_id = ?", (q_id,))
            q_count = cursor.fetchone()[0]
            
            # شمارش پاسخ‌ها (داده‌ها)
            cursor.execute("SELECT count(*) FROM response_values WHERE questionnaire_id = ?", (q_id,))
            r_count = cursor.fetchone()[0]
            
            return q_count, r_count
    except Exception as e:
        return -1, -1

# --- Executor ---
def execute_generated_code(code: str, db_path: str, artifacts_dir: str) -> Dict[str, Any]:
    buffer = io.StringIO()
    success = False
    output_text = ""
    generated_images = []
    
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    local_scope = {
        "pd": pd, "np": np, "sqlite3": sqlite3, "plt": plt, "json": json, "os": os,
        "political": political, "stats": stats, "viz": viz,
        "is_dataclass": is_dataclass, "asdict": asdict,
        "fetch_wide_dataframe": lambda qid: SQLiteRepository(db_path).fetch_wide_dataframe(qid),
        "RESULTS": {}, "ARTIFACTS": []   
    }

    try:
        plt.clf()
        plt.close('all')
        with contextlib.redirect_stdout(buffer):
            exec(code, {}, local_scope)
        output_text = buffer.getvalue()
        success = True
        for file in os.listdir(artifacts_dir):
            if file.lower().endswith(('.png', '.jpg')):
                generated_images.append(os.path.join(artifacts_dir, file))
    except Exception as e:
        output_text = f"خطا در اجرا: {str(e)}\n{traceback.format_exc()}"
        success = False

    return {"success": success, "output": output_text, "artifacts": generated_images}

# --- State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "current_questionnaire_id" not in st.session_state: st.session_state.current_questionnaire_id = None
if "profile_summary" not in st.session_state: st.session_state.profile_summary = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 مدیریت داده‌ها")
    uploaded_file = st.file_uploader("آپلود فایل", type=["csv", "xlsx"])
    
    if uploaded_file and not st.session_state.current_questionnaire_id:
        with st.status("در حال پردازش...", expanded=True) as status:
            try:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                
                if uploaded_file.name.endswith('.csv'): df = pd.read_csv(temp_path)
                else: df = pd.read_excel(temp_path)
                
                status.write(f"✅ فایل خوانده شد: {df.shape[0]} سطر, {df.shape[1]} ستون")
                
                importer = QuestionnaireImporter(settings.db_path, settings.respondent_id_salt)
                res = importer._import_dataframe(df, questionnaire_name=uploaded_file.name, version="v1", questionnaire_id=None, source_hint="upload")
                st.session_state.current_questionnaire_id = res.questionnaire_id
                
                # بررسی دیتابیس بلافاصله بعد از ایمپورت
                q_count, r_count = debug_database_schema(settings.db_path, res.questionnaire_id)
                status.write(f"📊 وضعیت دیتابیس: {q_count} ستون ذخیره شد.")

                if q_count == 0:
                    status.update(label="خطا: هیچ ستونی ذخیره نشد!", state="error")
                    st.error("مشکل مهم: فایل خوانده شد اما ستون‌ها در دیتابیس ذخیره نشدند. فرمت فایل اکسل را چک کنید.")
                else:
                    profiler = SQLiteEAVProfiler(settings.db_path)
                    profile = profiler.profile(res.questionnaire_id)
                    st.session_state.profile_summary = profile
                    status.write("✅ پروفایل داده‌ها ایجاد شد.")
                    status.update(label="آماده!", state="complete", expanded=False)
                
                os.remove(temp_path)
            except Exception as e:
                status.update(label="خطا", state="error")
                st.error(str(e))
                st.code(traceback.format_exc())

    if st.session_state.profile_summary:
        st.divider()
        summary = st.session_state.profile_summary
        # تبدیل امن به دیکشنری
        if is_dataclass(summary): summary = asdict(summary)
        
        q_list = summary.get('questions', [])
        st.info(f"رکوردها: {summary.get('n_total_responses', 0)}")
        
        if not q_list:
            st.warning("⚠️ لیست ستون‌ها خالی است!")
        else:
            cols = [q['column_name'] for q in q_list]
            st.text(f"ستون‌ها ({len(cols)}):")
            st.code("\n".join(cols[:10]) + "...", language="text")

    if st.button("شروع مجدد / پاکسازی"):
        st.session_state.clear()
        st.rerun()

# --- MAIN CHAT ---
st.title("🤖 دستیار تحلیلگر داده")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "artifacts" in msg:
            for img in msg["artifacts"]: st.image(img)

if prompt := st.chat_input("سوال شما..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if not st.session_state.current_questionnaire_id:
        st.error("لطفاً ابتدا فایل آپلود کنید.")
    else:
        with st.chat_message("assistant"):
            status_container = st.status("🤖 تحلیل درخواست...", expanded=True)
            try:
                # بازیابی پروفایل
                profile_data = st.session_state.get("profile_summary")
                if not profile_data:
                    profiler = SQLiteEAVProfiler(settings.db_path)
                    profile_data = profiler.profile(st.session_state.current_questionnaire_id)
                    st.session_state.profile_summary = profile_data

                # تبدیل امن برای استفاده در لاجیک
                if is_dataclass(profile_data): profile_data = asdict(profile_data)
                
                questions_list = profile_data.get('questions', [])
                if not questions_list:
                    status_container.update(label="خطای داده", state="error")
                    st.error("⛔ خطا: اسکیمای داده خالی است (No Schema).")
                    st.warning("""
                    به نظر می‌رسد فایل شما ستونی ندارد یا درست شناسایی نشده است.
                    ۱. مطمئن شوید فایل اکسل هدر (سرستون) دارد.
                    ۲. دکمه 'شروع مجدد' را بزنید و دوباره آپلود کنید.
                    ۳. اگر در سایدبار 'لیست ستون‌ها' خالی است، فایل مشکل دارد.
                    """)
                    st.stop()

                state = WorkflowState(
                    run_id=f"run_{uuid4().hex[:8]}",
                    questionnaire_id=st.session_state.current_questionnaire_id,
                    user_question=prompt,
                    schema_summary=[q['column_name'] for q in questions_list],
                    data_profile=profile_data
                )

                # Router Agent
                status_container.write("🔍 شناسایی ستون‌ها...")
                router = RouterMapperAgent(model=settings.router_model, db_path=settings.db_path)
                state = router.run(state)
                
                if not state.is_related:
                    reason = state.notes.get('router_reason', 'نامشخص')
                    final_msg = f"⛔ سوال نامرتبط تشخیص داده شد.\n**دلیل:** {reason}"
                    status_container.update(label="توقف", state="error")
                else:
                    # Planner
                    status_container.write("📝 برنامه‌ریزی...")
                    planner = PlannerAgent(model=settings.planner_model)
                    state = planner.run(state)
                    
                    # Coding Loop
                    coder = CodeWriterAgent(model=settings.code_writer_model)
                    reviewer = CodeReviewerAgent(model=settings.code_reviewer_model)
                    quality = QualityReviewAgent(model=settings.quality_review_model)
                    
                    for i in range(settings.max_code_iterations):
                        status_container.write(f"💻 کدنویسی (تلاش {i+1})...")
                        state = coder.run(state)
                        state = reviewer.run(state)
                        
                        if not state.code_review.get("approved"):
                            continue

                        status_container.write("⚙️ اجرا...")
                        exec_res = execute_generated_code(state.code_draft, settings.db_path, settings.artifacts_dir)
                        state.execution = exec_res

                        if not exec_res["success"]:
                            continue

                        status_container.write("🧐 بررسی کیفیت...")
                        state = quality.run(state)
                        if state.quality_review.get("approved"):
                            break
                    
                    # Report
                    status_container.write("✍️ گزارش نهایی...")
                    reporter = ReportWriterAgent(model=settings.report_writer_model)
                    state = reporter.run(state)
                    
                    final_msg = state.final_report
                    status_container.update(label="تمام شد!", state="complete")

                st.markdown(final_msg)
                if state.execution and state.execution.get("artifacts"):
                    cols = st.columns(len(state.execution["artifacts"]))
                    for idx, img in enumerate(state.execution["artifacts"]):
                        cols[idx].image(img)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_msg, 
                    "artifacts": state.execution.get("artifacts", []) if state.execution else []
                })

            except Exception as e:
                status_container.update(label="خطای سیستمی", state="error")
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())