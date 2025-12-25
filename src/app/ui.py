import sys
import os

# اضافه کردن مسیر اصلی پروژه به Path برای ایمپورت صحیح ماژول‌ها
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import io
import contextlib
from pathlib import Path
from typing import Dict, Any, List

# --- ماژول‌های پروژه ---
from src.app.config import Settings
from src.db.connection import connect
from src.db.repository import SQLiteRepository
from src.db.importer import QuestionnaireImporter
from src.db.profiler import SQLiteEAVProfiler
from src.workflows.state import WorkflowState
from src.agents.router_agent import RouterAgent
from src.agents.column_mapper_agent import ColumnMapperAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.stats_params_agent import StatsParamsAgent
from src.agents.code_writer_agent import CodeWriterAgent
from src.agents.code_reviewer_agent import CodeReviewerAgent
from src.agents.quality_review_agent import QualityReviewAgent
from src.agents.report_writer_agent import ReportWriterAgent



# --- تابع راه‌اندازی دیتابیس (جدید) ---
def init_database(db_path: str):
    """
    بررسی می‌کند که آیا دیتابیس مقداردهی اولیه شده است یا خیر.
    اگر جداول وجود نداشته باشند، فایل schema.sql را اجرا می‌کند.
    """
    # پیدا کردن مسیر فایل schema.sql نسبت به فایل فعلی
    # فایل ui.py در src/app است و schema.sql در src/db
    schema_path = os.path.join(os.path.dirname(__file__), "../db/schema.sql")
    
    try:
        # اتصال موقت برای چک کردن وجود جدول
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='questionnaires';")
            if cursor.fetchone():
                # جدول وجود دارد، نیازی به کاری نیست
                return
    except Exception:
        pass

    # اگر جدول وجود نداشت، با استفاده از Repository اسکیما را اعمال کن
    print(f"⚠️ Initializing database schema from {schema_path}...")
    # نکته: کلاس SQLiteRepository باید پارامتر schema_sql_path را در __init__ پشتیبانی کند (که در کدهای قبلی اضافه کردیم)
    SQLiteRepository(db_path, schema_sql_path=schema_path)
    print("✅ Database tables created successfully.")


# --- تنظیمات اولیه صفحه ---
st.set_page_config(
    page_title="دستیار تحلیل داده هوشمند",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- استایل‌دهی راست‌چین (RTL) ---
st.markdown("""
<style>
    .stTextInput, .stMarkdown, .stSelectbox, .stButton { direction: rtl; text-align: right; }
    h1, h2, h3, h4 { text-align: right; }
    .stChatMessage { direction: rtl; text-align: right; }
    p { text-align: right; }
</style>
""", unsafe_allow_html=True)

# --- بارگذاری تنظیمات ---
@st.cache_resource
def get_settings():
    return Settings.from_env()

settings = get_settings()

init_database(settings.db_path)

# --- توابع کمکی اجرا (Executor) ---
def execute_generated_code(code: str, db_path: str, artifacts_dir: str) -> Dict[str, Any]:
    """
    کد تولید شده توسط ایجنت را در یک محیط کنترل شده اجرا می‌کند.
    """
    buffer = io.StringIO()
    success = False
    output_text = ""
    generated_images = []

    # تعریف محیط محلی برای اجرا
    # ما توابع کمکی مثل fetch_wide_dataframe را اینجا تعریف می‌کنیم تا کد ایجنت ساده‌تر باشد
    def fetch_wide_dataframe(questionnaire_id: str):
        repo = SQLiteRepository(db_path)
        return repo.fetch_wide_dataframe(questionnaire_id)
    
    local_scope = {
        "pd": pd,
        "sqlite3": sqlite3,
        "plt": plt,
        "db_connection": db_path,
        "fetch_wide_dataframe": fetch_wide_dataframe,
    }

    try:
        # تغییر مسیر stdout برای گرفتن پرینت‌ها
        with contextlib.redirect_stdout(buffer):
            exec(code, {}, local_scope)
        
        output_text = buffer.getvalue()
        success = True
        
        # پیدا کردن عکس‌های تولید شده
        # فرض بر این است که ایجنت عکس‌ها را در artifacts_dir ذخیره کرده
        for file in os.listdir(artifacts_dir):
            if file.endswith(".png") or file.endswith(".jpg"):
                # چک کنیم که فایل جدید باشد (اختیاری)
                generated_images.append(os.path.join(artifacts_dir, file))
                
    except Exception as e:
        output_text = f"خطا در اجرا: {str(e)}"
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

# --- سایدبار: آپلود و تنظیمات ---
with st.sidebar:
    st.header("📂 مدیریت داده‌ها")
    
    uploaded_file = st.file_uploader("آپلود فایل اکسل/CSV", type=["csv", "xlsx"])
    
    if uploaded_file and not st.session_state.current_questionnaire_id:
        with st.status("در حال پردازش فایل...", expanded=True) as status:
            try:
                # ۱. ذخیره فایل موقت
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # ۲. خواندن فایل
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(temp_path)
                else:
                    df = pd.read_excel(temp_path)
                
                status.write("✅ فایل خوانده شد.")
                
                # ۳. ایمپورت به دیتابیس
                importer = QuestionnaireImporter(
                    settings.db_path, 
                    settings.respondent_id_salt
                )
                q_name = uploaded_file.name.split('.')[0]
                res = importer._import_dataframe(
                    df, 
                    questionnaire_name=q_name, 
                    version="v1", 
                    questionnaire_id=None, 
                    source_hint="upload"
                )
                st.session_state.current_questionnaire_id = res.questionnaire_id
                status.write(f"✅ داده‌ها وارد دیتابیس شدند ({res.inserted_responses} رکورد).")
                
                # ۴. پروفایلینگ
                profiler = SQLiteEAVProfiler(settings.db_path)
                profile = profiler.profile(res.questionnaire_id)
                st.session_state.profile_summary = profile
                status.write("✅ آنالیز آماری اولیه انجام شد.")
                
                status.update(label="آماده‌سازی تکمیل شد!", state="complete", expanded=False)
                os.remove(temp_path)
                
            except Exception as e:
                status.update(label="خطا در پردازش", state="error")
                st.error(f"Error: {e}")

    # نمایش خلاصه پروفایل
    if st.session_state.profile_summary:
        st.divider()
        st.subheader("📊 خلاصه داده‌ها")
        summary = st.session_state.profile_summary
        st.info(f"تعداد رکوردها: {summary.get('n_total_responses', 0)}")
        
        # لیست ستون‌ها
        cols = [q['column_name'] for q in summary.get('questions', [])]
        st.text("ستون‌های شناسایی شده:")
        st.code("\n".join(cols[:10]) + ("..." if len(cols)>10 else ""), language="text")

    # دکمه ریست
    if st.button("شروع مجدد / پاکسازی"):
        st.session_state.messages = []
        st.session_state.current_questionnaire_id = None
        st.session_state.profile_summary = None
        st.rerun()

# --- پنل اصلی: چت ---
st.title("🤖 دستیار تحلیلگر داده")
st.markdown("سوال خود را به فارسی بپرسید (مثلاً: *میانگین سن افرادی که از محصول راضی بودند چقدر است؟*)")

# نمایش تاریخچه چت
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "artifacts" in msg:
            for img in msg["artifacts"]:
                st.image(img)

# ورودی کاربر
if prompt := st.chat_input("سوال شما..."):
    # نمایش پیام کاربر
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # اگر داده‌ای بارگذاری نشده باشد
    if not st.session_state.current_questionnaire_id:
        st.error("لطفاً ابتدا یک فایل داده آپلود کنید.")
    else:
        # --- شروع فرآیند تحلیل (Agent Workflow) ---
        # --- شروع فرآیند تحلیل (Agent Workflow) ---
        with st.chat_message("assistant"):
            
            # کانتینر وضعیت زنده (Live Status)
            status_container = st.status("🤖 در حال تحلیل درخواست شما...", expanded=True)
            
            try:
                # 0. آماده‌سازی وضعیت اولیه
                # نکته: مطمئن شوید فایل src/workflows/state.py را آپدیت کرده‌اید
                state = WorkflowState(
                    run_id=f"run_{uuid4().hex[:8]}",
                    questionnaire_id=st.session_state.current_questionnaire_id,
                    user_question=prompt,
                    schema_summary=[q['column_name'] for q in st.session_state.profile_summary.get('questions', [])]
                )

                # 1. Router Agent (آیا سوال مرتبط است؟)
                status_container.write("🔍 بررسی ماهیت سوال...")
                router = RouterAgent(model=settings.router_model)
                state = router.run(state)
                
                if state.get("router_decision") == "reject":
                    final_msg = "⛔ سوال شما نامرتبط با داده‌های موجود تشخیص داده شد."
                    status_container.update(label="توقف تحلیل", state="error", expanded=False)
                
                else:
                    # 2. Mapper Agent (شناسایی ستون‌ها)
                    status_container.write("🔗 شناسایی ستون‌های مرتبط...")
                    mapper = ColumnMapperAgent(model=settings.mapper_model)
                    # پاس دادن لیست کامل ستون‌ها به مپر
                    state["all_column_names"] = state["schema_summary"]
                    state = mapper.run(state)
                    status_container.write(f"ستون‌های انتخاب شده: `{state.get('mapped_columns')}`")

                    # 3. Planner Agent (نقشه راه)
                    status_container.write("📝 طراحی استراتژی تحلیل...")
                    planner = PlannerAgent(model=settings.planner_model)
                    state = planner.run(state)
                    
                    # 4. حلقه تولید و اصلاح کد (Coding Loop)
                    max_retries = settings.max_code_iterations
                    coder = CodeWriterAgent(model=settings.code_writer_model)
                    reviewer = CodeReviewerAgent(model=settings.code_reviewer_model)
                    quality_agent = QualityReviewAgent(model=settings.quality_review_model)
                    
                    for i in range(max_retries):
                        iteration_label = f"(تلاش {i+1}/{max_retries})"
                        
                        # الف) نوشتن کد
                        status_container.write(f"💻 نوشتن کد پایتون {iteration_label}...")
                        state = coder.run(state)
                        
                        # ب) بررسی امنیت
                        status_container.write(f"🛡️ بررسی امنیت کد {iteration_label}...")
                        state = reviewer.run(state)
                        
                        if not state.get("code_is_safe", True):
                            status_container.write("⚠️ کد ناامن تشخیص داده شد. تلاش برای اصلاح...")
                            # فیدبک امنیتی در state['quality_feedback'] توسط Reviewer ذخیره شده است
                            continue # بازگشت به ابتدای حلقه برای اصلاح کد

                        # ج) اجرای کد
                        status_container.write(f"⚙️ اجرای کد {iteration_label}...")
                        code = state.get("code", "")
                        exec_res = execute_generated_code(code, settings.db_path, settings.artifacts_dir)
                        
                        state["execution_success"] = exec_res["success"]
                        state["execution_output"] = exec_res["output"]
                        state["execution_artifacts"] = exec_res["artifacts"]

                        if not exec_res["success"]:
                            state["quality_feedback"] = f"Runtime Error: {exec_res['output']}"
                            status_container.write(f"❌ خطا در اجرا: {exec_res['output'][:100]}...")
                            continue # بازگشت برای رفع باگ

                        # د) بررسی کیفیت پاسخ (Quality Review)
                        status_container.write(f"🧐 ارزیابی کیفیت پاسخ {iteration_label}...")
                        state = quality_agent.run(state)
                        
                        feedback = state.get("quality_feedback")
                        if feedback:
                            status_container.write(f"⚠️ پاسخ کامل نبود: {feedback}")
                            # حلقه ادامه می‌یابد تا کد اصلاح شود
                        else:
                            # کیفیت تایید شد
                            status_container.write("✅ کیفیت پاسخ تایید شد.")
                            break
                    else:
                        # اگر حلقه تمام شد و هنوز به نتیجه نرسیدیم
                        status_container.write("⚠️ حداکثر تلاش انجام شد. گزارش نهایی ممکن است کامل نباشد.")

                    # نمایش کد نهایی (برای کاربران فنی)
                    with st.expander("مشاهده کد نهایی پایتون"):
                        st.code(state.get("code", ""), language="python")
                        st.text(f"خروجی اجرا:\n{state.get('execution_output', '')}")

                    # 5. Report Writer (گزارش نهایی)
                    status_container.write("✍️ نگارش گزارش نهایی...")
                    reporter = ReportWriterAgent(model=settings.report_writer_model)
                    state = reporter.run(state)
                    
                    final_msg = state.get("final_report", "گزارشی تولید نشد.")
                    status_container.update(label="تحلیل کامل شد!", state="complete", expanded=False)

                # --- نمایش خروجی نهایی ---
                st.markdown(final_msg)
                
                # مدیریت فایل‌های خروجی (نمودارها)
                artifacts = state.get("execution_artifacts", [])
                if artifacts:
                    cols = st.columns(len(artifacts))
                    for idx, img_path in enumerate(artifacts):
                        with cols[idx]:
                            st.image(img_path, caption=f"نمودار {idx+1}", use_column_width=True)
                        
                # ذخیره در حافظه چت
                msg_data = {"role": "assistant", "content": final_msg}
                if artifacts:
                    msg_data["artifacts"] = artifacts
                st.session_state.messages.append(msg_data)

            except Exception as e:
                status_container.update(label="خطای سیستمی", state="error")
                st.error(f"یک خطای پیش‌بینی نشده رخ داد: {str(e)}")
                # برای دیباگ، پرینت کامل خطا در کنسول
                import traceback
                traceback.print_exc()