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

# --- تنظیمات جلوگیری از کرش‌های گرافیکی ---
# استفاده از بک‌اند غیرتعاملی برای جلوگیری از خطای Thread در استریم‌لیت
matplotlib.use('Agg')
# نادیده گرفتن هشدارهای غیر-بحرانی پانداس در خروجی
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
    st.error(f"خطا در بارگذاری ماژول‌های پروژه. لطفاً مسیرها را بررسی کنید: {e}")
    st.stop()

# --- تابع راه‌اندازی دیتابیس ---
def init_database(db_path: str):
    schema_path = os.path.join(os.path.dirname(__file__), "../db/schema.sql")
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='questionnaires';")
            if cursor.fetchone():
                return
    except Exception:
        pass

    print(f"⚠️ Initializing database schema from {schema_path}...")
    try:
        SQLiteRepository(db_path, schema_sql_path=schema_path)
        print("✅ Database tables created successfully.")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")

# --- تنظیمات اولیه صفحه ---
st.set_page_config(
    page_title="دستیار تحلیل داده هوشمند",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stTextInput, .stMarkdown, .stSelectbox, .stButton { direction: rtl; text-align: right; }
    h1, h2, h3, h4 { text-align: right; }
    .stChatMessage { direction: rtl; text-align: right; }
    p { text-align: right; }
    div[data-testid="stStatusWidget"] { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

# --- بارگذاری تنظیمات ---
@st.cache_resource
def get_settings():
    return Settings.from_env()

settings = get_settings()

# اطمینان از وجود دایرکتوری‌ها
os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
if hasattr(settings, 'artifacts_dir'):
    os.makedirs(settings.artifacts_dir, exist_ok=True)

init_database(settings.db_path)

# --- توابع کمکی اجرا (Executor) ---
def execute_generated_code(code: str, db_path: str, artifacts_dir: str) -> Dict[str, Any]:
    """
    کد تولید شده توسط ایجنت را اجرا می‌کند.
    """
    buffer = io.StringIO()
    success = False
    output_text = ""
    generated_images = []

    # ساخت پوشه اگر نباشد
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    # تعریف محیط محلی (Local Scope)
    # تمام ابزارهایی که ایجنت ممکن است نیاز داشته باشد اینجا تزریق می‌شوند
    local_scope = {
        "pd": pd,
        "np": np,
        "sqlite3": sqlite3,
        "plt": plt,
        "json": json,
        "os": os,
        "political": political,
        "stats": stats,
        "viz": viz,
        "is_dataclass": is_dataclass,  # <--- FIX: جلوگیری از خطای is_dataclass
        "asdict": asdict,              # <--- FIX: جلوگیری از خطای asdict
        "fetch_wide_dataframe": lambda qid: SQLiteRepository(db_path).fetch_wide_dataframe(qid),
        "RESULTS": {},    
        "ARTIFACTS": []   
    }

    try:
        # پاکسازی نمودارهای قبلی
        plt.clf()
        plt.close('all')
        
        # اجرای کد
        with contextlib.redirect_stdout(buffer):
            exec(code, {}, local_scope)
        
        output_text = buffer.getvalue()
        success = True
        
        # جمع‌آوری خروجی‌های تصویری
        # فرض بر این است که کد ایجنت تصاویر را در مسیر artifacts_dir ذخیره می‌کند
        # یا می‌توانیم تصاویر جدید ایجاد شده را شناسایی کنیم
        for file in os.listdir(artifacts_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                generated_images.append(os.path.join(artifacts_dir, file))
                
    except Exception as e:
        # چاپ کامل خطا برای دیباگ
        output_text = f"خطا در اجرا: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        success = False

    return {
        "success": success,
        "output": output_text,
        "artifacts": generated_images
    }

# --- مدیریت وضعیت (Session State) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_questionnaire_id" not in st.session_state:
    st.session_state.current_questionnaire_id = None
if "profile_summary" not in st.session_state:
    st.session_state.profile_summary = None

# --- سایدبار ---
with st.sidebar:
    st.header("📂 مدیریت داده‌ها")
    
    uploaded_file = st.file_uploader("آپلود فایل اکسل/CSV", type=["csv", "xlsx"])
    
    if uploaded_file and not st.session_state.current_questionnaire_id:
        with st.status("در حال پردازش فایل...", expanded=True) as status:
            try:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # خواندن فایل
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(temp_path)
                else:
                    df = pd.read_excel(temp_path)
                
                status.write("✅ فایل خوانده شد.")
                
                # ایمپورت
                importer = QuestionnaireImporter(settings.db_path, settings.respondent_id_salt)
                q_name = uploaded_file.name.split('.')[0]
                res = importer._import_dataframe(
                    df, questionnaire_name=q_name, version="v1", questionnaire_id=None, source_hint="upload"
                )
                st.session_state.current_questionnaire_id = res.questionnaire_id
                status.write(f"✅ داده‌ها وارد دیتابیس شدند ({res.inserted_responses} رکورد).")
                
                # پروفایلینگ
                profiler = SQLiteEAVProfiler(settings.db_path)
                profile = profiler.profile(res.questionnaire_id)
                st.session_state.profile_summary = profile
                status.write("✅ آنالیز آماری اولیه انجام شد.")
                
                status.update(label="آماده‌سازی تکمیل شد!", state="complete", expanded=False)
                os.remove(temp_path)
                
            except Exception as e:
                status.update(label="خطا در پردازش", state="error")
                st.error(f"Error: {e}")
                st.code(traceback.format_exc())

    if st.session_state.profile_summary:
        st.divider()
        st.subheader("📊 خلاصه داده‌ها")
        summary = st.session_state.profile_summary
        st.info(f"تعداد رکوردها: {summary.get('n_total_responses', 0)}")
        cols = [q['column_name'] for q in summary.get('questions', [])]
        st.text("ستون‌های شناسایی شده:")
        st.code("\n".join(cols[:10]) + ("..." if len(cols)>10 else ""), language="text")

    if st.button("شروع مجدد / پاکسازی"):
        st.session_state.messages = []
        st.session_state.current_questionnaire_id = None
        st.session_state.profile_summary = None
        st.rerun()

# --- چت روم ---
st.title("🤖 دستیار تحلیلگر داده")

# نمایش پیام‌های قبلی
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "artifacts" in msg and msg["artifacts"]:
            for img in msg["artifacts"]:
                st.image(img)

# دریافت ورودی کاربر
if prompt := st.chat_input("سوال خود را بپرسید..."):
    # افزودن پیام کاربر
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # بررسی وجود فایل
    if not st.session_state.current_questionnaire_id:
        st.error("لطفاً ابتدا یک فایل داده آپلود کنید.")
    else:
        # شروع پردازش دستیار
        with st.chat_message("assistant"):
            status_container = st.status("🤖 در حال تحلیل درخواست شما...", expanded=True)
            final_msg = ""
            artifacts = []

            try:
                # 1. بازیابی پروفایل (با مکانیزم امن)
                profile_data = st.session_state.get("profile_summary")
                
                if not profile_data and st.session_state.current_questionnaire_id:
                    # تلاش مجدد برای خواندن از دیتابیس
                    profiler = SQLiteEAVProfiler(settings.db_path)
                    profile_data = profiler.profile(st.session_state.current_questionnaire_id)
                    st.session_state.profile_summary = profile_data
                
                if not profile_data:
                    status_container.update(label="خطا در داده‌ها", state="error")
                    st.error("مشکلی در خواندن پروفایل داده‌ها پیش آمد. لطفاً فایل را مجدد آپلود کنید.")
                    st.stop()

                # 2. ایجاد وضعیت اولیه
                state = WorkflowState(
                    run_id=f"run_{uuid4().hex[:8]}",
                    questionnaire_id=st.session_state.current_questionnaire_id,
                    user_question=prompt,
                    schema_summary=[q['column_name'] for q in profile_data.get('questions', [])],
                    data_profile=profile_data
                )

                # 3. مسیریابی (Router Agent)
                status_container.write("🔍 بررسی سوال و ستون‌ها...")
                router = RouterMapperAgent(model=settings.router_model, db_path=settings.db_path)
                state = router.run(state)
                
                if not state.is_related:
                    # اگر سوال نامرتبط بود
                    reason = state.notes.get('router_reason', 'دلیل مشخص نیست')
                    final_msg = f"⛔ سوال شما نامرتبط تشخیص داده شد.\n\n**دلیل:** {reason}"
                    status_container.update(label="توقف تحلیل", state="error", expanded=False)
                else:
                    # 4. برنامه‌ریزی (Planner Agent)
                    status_container.write("📝 طراحی استراتژی تحلیل...")
                    planner = PlannerAgent(model=settings.planner_model)
                    state = planner.run(state)
                    
                    # 5. حلقه کدنویسی (Code Loop)
                    coder = CodeWriterAgent(model=settings.code_writer_model)
                    reviewer = CodeReviewerAgent(model=settings.code_reviewer_model)
                    quality_agent = QualityReviewAgent(model=settings.quality_review_model)
                    
                    for i in range(settings.max_code_iterations):
                        status_container.write(f"💻 کدنویسی و اجرا (تلاش {i+1})...")
                        
                        # نوشتن کد
                        state = coder.run(state)
                        
                        # بررسی کد
                        state = reviewer.run(state)
                        review_result = state.code_review or {}
                        
                        if not review_result.get("approved", False):
                            status_container.write(f"⚠️ نیاز به اصلاح کد: {review_result.get('feedback')}")
                            # ادامه لوپ برای اصلاح کد توسط CodeWriter در دور بعدی
                            continue 

                        # اجرای کد
                        status_container.write("⚙️ اجرای کد پایتون...")
                        exec_res = execute_generated_code(state.code_draft, settings.db_path, settings.artifacts_dir)
                        state.execution = exec_res

                        if not exec_res["success"]:
                            # اگر اجرا خطا داد، به ایجنت برمی‌گردیم تا اصلاح کند
                            status_container.write(f"❌ خطا در اجرا: {exec_res['output'][:100]}...")
                            continue 

                        # ارزیابی کیفیت خروجی
                        status_container.write("🧐 بررسی کیفیت پاسخ...")
                        state = quality_agent.run(state)
                        
                        if state.quality_review.get("approved"):
                            status_container.write("✅ پاسخ تایید شد.")
                            break
                        else:
                            status_container.write("⚠️ بهبود پاسخ نهایی...")
                    
                    # 6. گزارش‌نویسی (Report Agent)
                    status_container.write("✍️ تنظیم گزارش نهایی...")
                    reporter = ReportWriterAgent(model=settings.report_writer_model)
                    state = reporter.run(state)
                    
                    final_msg = state.final_report
                    if state.execution:
                        artifacts = state.execution.get("artifacts", [])
                    
                    status_container.update(label="تحلیل کامل شد!", state="complete", expanded=False)

                # نمایش خروجی نهایی
                st.markdown(final_msg)
                
                if artifacts:
                    cols = st.columns(len(artifacts))
                    for idx, img_path in enumerate(artifacts):
                        with cols[idx]:
                            st.image(img_path, caption=f"نمودار {idx+1}", use_container_width=True)
                        
                # ذخیره در تاریخچه
                msg_data = {"role": "assistant", "content": final_msg}
                if artifacts:
                    msg_data["artifacts"] = artifacts
                st.session_state.messages.append(msg_data)

            except Exception as e:
                status_container.update(label="خطای سیستمی", state="error")
                st.error(f"یک خطای پیش‌بینی نشده رخ داد: {str(e)}")
                st.code(traceback.format_exc())