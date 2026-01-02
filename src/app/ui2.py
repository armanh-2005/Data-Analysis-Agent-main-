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
    from src.agents.report_writer_agent import ReportWriterAgent
    from src.tools import political, stats, viz
except ImportError as e:
    st.error(f"خطا در بارگذاری ماژول‌ها: {e}")
    st.stop()

# --- تنظیمات اولیه ---
st.set_page_config(page_title="دستیار تحلیل داده", page_icon="📊", layout="wide")
st.markdown("""
<style>
    .stTextInput, .stMarkdown, .stButton { direction: rtl; text-align: right; }
    .stCode { direction: ltr; }
    div[data-testid="stStatus"] { direction: rtl; }
    .stTabs [data-baseweb="tab-list"] { justify-content: flex-end; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_settings():
    return Settings.from_env()

settings = get_settings()
os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
if hasattr(settings, 'artifacts_dir'):
    os.makedirs(settings.artifacts_dir, exist_ok=True)

# --- تابع بررسی دیتابیس (کانتر ستون و ردیف) ---
def debug_database_schema(db_path, q_id):
    """بررسی می‌کند آیا واقعاً ستون‌ها در دیتابیس ذخیره شده‌اند؟"""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM questionnaire_schema WHERE questionnaire_id = ?", (q_id,))
            q_count = cursor.fetchone()[0]
            cursor.execute("SELECT count(*) FROM responses WHERE questionnaire_id = ?", (q_id,))
            r_count = cursor.fetchone()[0]
            return q_count, r_count
    except Exception:
        return 0, 0

# --- Executor (با تزریق دیتافریم) ---
def execute_generated_code(code: str, db_path: str, artifacts_dir: str, questionnaire_id: str = None) -> Dict[str, Any]:
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    # 1. تابع کمکی برای دریافت دیتا از دیتابیس
    def _fetch_helper(qid=None):
        target_id = qid or questionnaire_id
        if not target_id:
            raise ValueError("Questionnaire ID not found in environment.")
        repo = SQLiteRepository(db_path)
        return repo.fetch_wide_dataframe(target_id)

    # 2. لود کردن دیتافریم (Pre-loading)
    # این کار باعث می‌شود df همیشه وجود داشته باشد
    try:
        preloaded_df = _fetch_helper(questionnaire_id)
        # print(f"DEBUG: Dataframe loaded successfully with shape: {preloaded_df.shape}")
    except Exception as e:
        # print(f"DEBUG: Failed to preload dataframe: {e}")
        preloaded_df = pd.DataFrame() 

    # 3. ساخت محیط اجرا (Local Scope)
    local_scope = {
        "pd": pd, "np": np, "sqlite3": sqlite3, "plt": plt, "json": json, "os": os,
        "political": political, "stats": stats, "viz": viz,
        "is_dataclass": is_dataclass, "asdict": asdict,
        "fetch_wide_dataframe": _fetch_helper,
        "questionnaire_id": questionnaire_id,
        "artifacts_dir": artifacts_dir,
        "RESULTS": {}, 
        "ARTIFACTS": [],
        
        # >>> تزریق متغیر df <<<
        "df": preloaded_df 
    }
    
    plt.clf()
    plt.close('all')

    cells = code.split('# %%')
    full_output_log = []
    generated_images = []
    has_error = False

    for i, cell_code in enumerate(cells):
        cell_code = cell_code.strip()
        if not cell_code: continue
            
        cell_output = io.StringIO()
        cell_header = f"\n--- [CELL {i+1}] ---\n"
        
        try:
            with contextlib.redirect_stdout(cell_output):
                with contextlib.redirect_stderr(cell_output):
                    exec(cell_code, {}, local_scope)
            
            output_str = cell_output.getvalue()
            full_output_log.append(f"{cell_header}{output_str if output_str.strip() else '(Executed successfully)'}")

        except Exception:
            has_error = True
            error_trace = traceback.format_exc()
            full_output_log.append(f"{cell_header}❌ ERROR:\n{error_trace}")

    # جمع‌آوری تصاویر
    for file in os.listdir(artifacts_dir):
        if file.lower().endswith(('.png', '.jpg')):
            generated_images.append(os.path.join(artifacts_dir, file))
    
    if "ARTIFACTS" in local_scope and isinstance(local_scope["ARTIFACTS"], list):
         for art in local_scope["ARTIFACTS"]:
             if art not in generated_images and os.path.exists(art):
                 generated_images.append(art)

    return {
        "success": not has_error,
        "output": "\n".join(full_output_log),
        "artifacts": generated_images
    }

# --- State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "current_questionnaire_id" not in st.session_state: st.session_state.current_questionnaire_id = None
if "profile_summary" not in st.session_state: st.session_state.profile_summary = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 مدیریت داده‌ها")
    uploaded_file = st.file_uploader("آپلود فایل (Excel/CSV)", type=["csv", "xlsx"])
    
    if uploaded_file:
        if st.session_state.current_questionnaire_id is None:
            with st.status("در حال پردازش فایل...", expanded=True) as status:
                try:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    
                    importer = QuestionnaireImporter(settings.db_path, settings.respondent_id_salt)
                    
                    if uploaded_file.name.endswith('.csv'): 
                        res = importer.import_csv(temp_path, questionnaire_name=uploaded_file.name, version="v1")
                    else: 
                        res = importer.import_excel(temp_path, questionnaire_name=uploaded_file.name, version="v1")
                    
                    st.session_state.current_questionnaire_id = res.questionnaire_id
                    
                    q_count, r_count = debug_database_schema(settings.db_path, res.questionnaire_id)
                    
                    if q_count > 0:
                        status.write(f"✅ **{q_count} ستون** و **{r_count} ردیف** ذخیره شد.")
                        profiler = SQLiteEAVProfiler(settings.db_path)
                        profile = profiler.profile(res.questionnaire_id)
                        st.session_state.profile_summary = profile
                        status.update(label="آماده!", state="complete", expanded=False)
                    else:
                        status.update(label="خطا در ذخیره!", state="error")
                        st.error(f"دیتابیس خالی است! (ستون: {q_count}، ردیف: {r_count})")
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    status.update(label="خطا", state="error")
                    st.error(f"Error: {str(e)}")

    if st.session_state.profile_summary:
        st.divider()
        summary = st.session_state.profile_summary
        if is_dataclass(summary): summary = asdict(summary)
        q_list = summary.get('questions', [])
        st.info(f"📊 رکوردها: {summary.get('n_total_responses', 0)}")
        if q_list:
            cols = [q['column_name'] for q in q_list]
            st.text(f"ستون‌ها ({len(cols)}):")
            st.code("\n".join(cols[:10]) + ("..." if len(cols)>10 else ""), language="text")

    if st.button("شروع مجدد / پاکسازی"):
        st.session_state.clear()
        st.rerun()

# --- MAIN CHAT ---
st.title("🤖 دستیار تحلیلگر داده")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            tab1, tab2, tab3 = st.tabs(["📝 گزارش", "📊 نمودارها", "💻 کد و لاگ"])
            with tab1: st.markdown(msg["content"])
            with tab2:
                if msg.get("artifacts"):
                    cols = st.columns(min(len(msg["artifacts"]), 2))
                    for idx, img in enumerate(msg["artifacts"]):
                        cols[idx % 2].image(img, use_column_width=True)
                else:
                    st.info("نموداری تولید نشده است.")
            with tab3:
                if msg.get("code"):
                    st.markdown("**کد تولید شده:**")
                    st.code(msg["code"], language="python")
                if msg.get("log"):
                    st.markdown("**خروجی اجرا:**")
                    st.code(msg["log"], language="text")

if prompt := st.chat_input("سوال تحلیلی خود را بپرسید..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if not st.session_state.current_questionnaire_id:
        st.error("لطفاً ابتدا فایل آپلود کنید.")
    else:
        with st.chat_message("assistant"):
            status_box = st.status("🤖 شروع تحلیل...", expanded=True)
            
            final_msg = ""
            final_artifacts = []
            final_code = ""
            final_log = ""
            
            try:
                # 1. بازیابی پروفایل
                profile_data = st.session_state.get("profile_summary")
                if not profile_data:
                    profiler = SQLiteEAVProfiler(settings.db_path)
                    profile_data = profiler.profile(st.session_state.current_questionnaire_id)
                    st.session_state.profile_summary = profile_data

                if is_dataclass(profile_data): profile_data = asdict(profile_data)
                questions_list = profile_data.get('questions', [])

                # 2. State
                state = WorkflowState(
                    run_id=f"run_{uuid4().hex[:8]}",
                    questionnaire_id=st.session_state.current_questionnaire_id,
                    user_question=prompt,
                    schema_summary=[q['column_name'] for q in questions_list],
                    data_profile=profile_data
                )

                # 3. Router
                status_box.write("🔍 تحلیل سوال و مسیریابی...")
                router = RouterMapperAgent(model=settings.router_model, db_path=settings.db_path)
                state = router.run(state)
                
                if not state.is_related:
                    final_msg = f"⛔ سوال نامرتبط است: {state.notes.get('router_reason')}"
                    status_box.update(label="توقف", state="error")
                else:
                    # 4. Planner
                    status_box.write("📝 برنامه‌ریزی تحلیل...")
                    planner = PlannerAgent(model=settings.planner_model)
                    state = planner.run(state)
                    
                    # --- فاز ۱: تحلیل عددی (Execution-Based Loop) ---
                    analyst = CodeWriterAgent(model=settings.code_writer_model, mode="analysis")
                    analysis_success = False
                    
                    # پاک کردن خطاهای قبلی
                    state.execution = {} 

                    for i in range(settings.max_code_iterations):
                        status_box.write(f"🧮 تحلیل عددی (تلاش {i+1})...")
                        
                        # 1. تولید کد
                        state = analyst.run(state)
                        
                        with status_box:
                            with st.expander(f"Analysis Code {i+1}", expanded=False):
                                st.code(state.code_draft, language="python")

                        # 2. اجرا
                        exec_res = execute_generated_code(
                            state.code_draft, settings.db_path, settings.artifacts_dir, state.questionnaire_id
                        )
                        
                        with status_box:
                            with st.expander(f"Analysis Log {i+1}", expanded=False):
                                st.text(exec_res["output"])

                        # 3. بررسی نتیجه
                        if exec_res["success"]:
                            # بررسی خروجی خالی
                            if not exec_res["output"].strip() and not exec_res["artifacts"]:
                                status_box.write("⚠️ اجرا شد ولی خروجی نداشت. تلاش مجدد...")
                                state.execution = {"error_trace": "Code executed successfully but printed NOTHING. Please use print() to show results."}
                            else:
                                analysis_success = True
                                final_code += f"\n# --- ANALYSIS ---\n{state.code_draft}\n"
                                final_log += f"\n--- ANALYSIS LOG ---\n{exec_res['output']}\n"
                                state.analysis_output = exec_res["output"]
                                break
                        else:
                            status_box.write("❌ خطا در اجرا. اصلاح خودکار...")
                            state.execution = {"error_trace": exec_res["output"]}

                    if not analysis_success:
                        raise RuntimeError("تحلیل عددی با شکست مواجه شد.")

                    # --- فاز ۲: ترسیم نمودار (Execution-Based Loop) ---
                    visualizer = CodeWriterAgent(model=settings.code_writer_model, mode="visualization")
                    viz_success = False
                    state.execution = {} # ریست خطاها برای فاز جدید
                    
                    for i in range(settings.max_code_iterations):
                        status_box.write(f"🎨 ترسیم نمودار (تلاش {i+1})...")
                        state = visualizer.run(state)
                        
                        with status_box:
                            with st.expander(f"Viz Code {i+1}", expanded=False):
                                st.code(state.viz_code, language="python")

                        exec_res = execute_generated_code(
                            state.viz_code, settings.db_path, settings.artifacts_dir, state.questionnaire_id
                        )
                        state.viz_artifacts = exec_res["artifacts"]

                        if exec_res["success"]:
                            viz_success = True
                            final_artifacts = exec_res["artifacts"]
                            final_code += f"\n# --- VISUALIZATION ---\n{state.viz_code}\n"
                            break
                        else:
                             status_box.write("❌ خطا در رسم نمودار. اصلاح خودکار...")
                             state.execution = {"error_trace": exec_res["output"]}

                    # 6. Report
                    status_box.write("✍️ تنظیم گزارش نهایی...")
                    reporter = ReportWriterAgent(model=settings.report_writer_model)
                    state = reporter.run(state)
                    
                    final_msg = state.final_report
                    status_box.update(label="تمام شد!", state="complete", expanded=False)

                    # --- نمایش نهایی ---
                    tab1, tab2, tab3 = st.tabs(["📝 گزارش", "📊 نمودارها", "💻 کد و لاگ"])
                    with tab1: st.markdown(final_msg)
                    with tab2:
                        if final_artifacts:
                            cols = st.columns(min(len(final_artifacts), 2))
                            for idx, img in enumerate(final_artifacts):
                                cols[idx % 2].image(img, caption=f"Chart {idx+1}", use_column_width=True)
                        else:
                            st.info("هیچ نموداری تولید نشد.")
                    with tab3:
                        st.markdown("### کدهای اجرا شده")
                        st.code(final_code, language="python")
                        st.divider()
                        st.markdown("### لاگ کامل")
                        st.code(final_log, language="text")

                    # ذخیره در سشن
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_msg, 
                        "artifacts": final_artifacts,
                        "code": final_code,
                        "log": final_log
                    })

            except Exception as e:
                status_box.update(label="خطای سیستمی", state="error")
                st.error(f"Error: {str(e)}")
                st.code(traceback.format_exc())